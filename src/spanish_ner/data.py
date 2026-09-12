"""Loading and validation of the CoNLL-2002 Spanish NER corpus.

The corpus ships as plain text in the classic CoNLL column format::

    Melbourne NP B-LOC
    (         Fpa O
    Australia NP B-LOC

One token per line, blank line between sentences, columns separated by a
single space: ``TOKEN  POS  NER_TAG``. Tags follow the IOB2 scheme over four
entity types: PER, ORG, LOC, MISC.

We parse the raw files ourselves rather than relying on a remote loading
script. This keeps the pipeline reproducible even as dataset hosting changes,
and lets us verify the byte content of every split with a checksum.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from spanish_ner.utils import PROJECT_ROOT

# The label set is hard-coded, not inferred from whatever happens to appear in
# a given file. An inferred order would silently change the id -> label mapping
# between runs and quietly corrupt a saved model's config.
ENTITY_TYPES: tuple[str, ...] = ("LOC", "MISC", "ORG", "PER")
LABELS: tuple[str, ...] = ("O",) + tuple(
    f"{prefix}-{ent}" for ent in ENTITY_TYPES for prefix in ("B", "I")
)
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}


@dataclass(frozen=True)
class Sentence:
    """One tokenised sentence with its part-of-speech and NER annotations."""

    tokens: list[str]
    pos_tags: list[str]
    ner_tags: list[str]

    def __post_init__(self) -> None:
        n = len(self.tokens)
        if not (len(self.pos_tags) == len(self.ner_tags) == n):
            raise ValueError(
                f"Ragged sentence: {n} tokens, {len(self.pos_tags)} POS, "
                f"{len(self.ner_tags)} NER tags"
            )

    def __len__(self) -> int:
        return len(self.tokens)


class DataIntegrityError(RuntimeError):
    """Raised when a raw file is missing, corrupt, or has an unexpected checksum."""


def sha256(path: str | Path) -> str:
    """Stream a file through SHA-256 so large corpora never load into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_conll(path: str | Path) -> list[Sentence]:
    """Parse a CoNLL-2002 column file into a list of :class:`Sentence`.

    Malformed lines raise rather than being skipped: a silently dropped token
    would shift every downstream label and produce a wrong-but-plausible score.
    """
    sentences: list[Sentence] = []
    tokens: list[str] = []
    pos: list[str] = []
    ner: list[str] = []

    # The distributed files are UTF-8 despite the corpus's latin-1 heritage.
    # Reading them as latin-1 yields silent mojibake ("informacion" -> "informaciÃ³n")
    # on ~27k tokens, so we assert strict UTF-8 and let a mismatch fail loudly.
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                if tokens:
                    sentences.append(Sentence(tokens, pos, ner))
                    tokens, pos, ner = [], [], []
                continue
            parts = line.split()
            if len(parts) != 3:
                raise DataIntegrityError(
                    f"{Path(path).name}:{lineno}: expected 3 columns, got {len(parts)}: {line!r}"
                )
            token, pos_tag, ner_tag = parts
            if ner_tag not in LABEL2ID:
                raise DataIntegrityError(
                    f"{Path(path).name}:{lineno}: unknown NER tag {ner_tag!r}"
                )
            tokens.append(token)
            pos.append(pos_tag)
            ner.append(ner_tag)

    if tokens:  # file not terminated by a blank line
        sentences.append(Sentence(tokens, pos, ner))
    return sentences


def load_split(
    split: str,
    config: dict,
    verify_checksum: bool = True,
) -> list[Sentence]:
    """Load one official split ('train', 'dev' or 'test') with integrity checks."""
    data_cfg = config["data"]
    if split not in data_cfg["files"]:
        raise KeyError(f"Unknown split {split!r}; expected one of {list(data_cfg['files'])}")

    filename = data_cfg["files"][split]
    path = PROJECT_ROOT / data_cfg["raw_dir"] / filename
    if not path.exists():
        raise DataIntegrityError(
            f"Missing raw file {path}. Run `python scripts/download_data.py` first."
        )

    if verify_checksum:
        expected = data_cfg["checksums"][filename]
        actual = sha256(path)
        if actual != expected:
            raise DataIntegrityError(
                f"Checksum mismatch for {filename}.\n  expected {expected}\n  actual   {actual}\n"
                "The raw data changed; every metric in reports/ is now stale."
            )

    return parse_conll(path)


def load_all(config: dict, include_test: bool = False) -> dict[str, list[Sentence]]:
    """Load train and dev; the test split is opt-in.

    Making `test` opt-in is a deliberate guard rail: reading the official test
    split (`esp.testb`) requires asking for it by name, so it cannot happen by
    accident.

    Two scripts ask. `scripts/evaluate_test.py` produces the reported scores,
    and `scripts/error_analysis.py` describes the errors of the already-chosen
    model. An earlier version of this docstring said the test split was "read
    exactly once, by the final evaluation script", and that the error analysis
    ran on dev. Both were wrong, and grep says so in one line.

    The property that actually matters is intact and is narrower than what was
    claimed: **nothing is selected on the test split.** Baselines,
    hyper-parameters and early stopping all use dev. The error analysis runs
    after the final evaluation and changed no model.
    """
    splits = {name: load_split(name, config) for name in ("train", "dev")}
    if include_test:
        splits["test"] = load_split("test", config)
    return splits


def iter_entities(tags: list[str]):
    """Decode an IOB2 tag sequence into ``(entity_type, start, end)`` spans.

    ``end`` is exclusive. An ``I-`` tag that opens a span without a preceding
    ``B-`` of the same type is treated as the start of a new entity, matching
    seqeval's default (lenient) behaviour.
    """
    start: int | None = None
    ent_type: str | None = None

    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-") or (
            tag.startswith("I-") and (ent_type is None or tag[2:] != ent_type)
        ):
            if start is not None:
                yield ent_type, start, i
            start, ent_type = i, tag[2:]
        elif tag == "O":
            if start is not None:
                yield ent_type, start, i
            start, ent_type = None, None


def corpus_stats(sentences: list[Sentence]) -> dict:
    """Summary statistics used by the EDA notebook and the README table."""
    from collections import Counter

    ent_counts: Counter[str] = Counter()
    for sent in sentences:
        for ent_type, _, _ in iter_entities(sent.ner_tags):
            ent_counts[ent_type] += 1

    n_tokens = sum(len(s) for s in sentences)
    return {
        "n_sentences": len(sentences),
        "n_tokens": n_tokens,
        "avg_sentence_length": round(n_tokens / len(sentences), 2) if sentences else 0.0,
        "n_entities": sum(ent_counts.values()),
        "entities_by_type": dict(sorted(ent_counts.items())),
    }


def sentence_key(sentence: Sentence) -> str:
    """Surface form used to detect verbatim duplicates across splits."""
    return " ".join(sentence.tokens)


def split_overlap(a: list[Sentence], b: list[Sentence]) -> set[str]:
    """Sentences appearing verbatim in both splits."""
    return {sentence_key(s) for s in a} & {sentence_key(s) for s in b}


def deduplicate_against(
    target: list[Sentence], reference: list[Sentence]
) -> list[Sentence]:
    """Drop sentences of `target` that appear verbatim in `reference`.

    The official CoNLL-2002 splits are not disjoint: 309 of the 1,517 test
    sentences (20.4%) also occur in training, mostly newswire boilerplate such
    as the single token "-" or datelines like "Madrid , 23 may ( EFE ) .".
    Those duplicates hold 336 of the 3,559 test entities (9.4%).

    We cannot remove them from the official benchmark without losing
    comparability with published results, so evaluation reports both numbers:
    the full test set (comparable) and this deduplicated subset (a stricter
    estimate of generalisation to unseen text).
    """
    seen = {sentence_key(s) for s in reference}
    return [s for s in target if sentence_key(s) not in seen]
