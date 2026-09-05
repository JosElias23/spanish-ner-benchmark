"""Non-neural baselines.

Every transformer result needs something to be compared against. Without a
baseline, "F1 = 0.87" is an uninterpretable number: it could be state of the
art or it could be worse than a lookup table.

Two baselines are implemented:

1. ``GazetteerTagger`` -- memorise every entity surface form seen in training
   and greedily longest-match them at inference. This is the floor. It also
   quantifies how much of the task is pure memorisation, which is the honest
   context for interpreting any learned model's score.

2. ``CRFTagger`` -- a linear-chain conditional random field over hand-designed
   orthographic and contextual features. This was the state of the art for
   sequence labelling before contextual embeddings, so it is the meaningful
   bar a transformer has to clear.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import sklearn_crfsuite

from spanish_ner.data import Sentence, iter_entities

# ---------------------------------------------------------------------------
# 1. Gazetteer / dictionary lookup
# ---------------------------------------------------------------------------


class GazetteerTagger:
    """Longest-match tagger over entity surface forms memorised from training.

    Ambiguity (the same string annotated as different types in different
    contexts, e.g. "Barcelona" as LOC and as ORG) is resolved by majority vote,
    since the model has no context to disambiguate with. That limitation is the
    point: it shows precisely what contextual models buy us.
    """

    def __init__(self, max_span_length: int = 8) -> None:
        self.max_span_length = max_span_length
        self.gazetteer: dict[tuple[str, ...], str] = {}

    def fit(self, sentences: list[Sentence]) -> GazetteerTagger:
        surface_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        for sent in sentences:
            for ent_type, start, end in iter_entities(sent.ner_tags):
                surface_counts[tuple(sent.tokens[start:end])][ent_type] += 1

        self.gazetteer = {
            surface: counts.most_common(1)[0][0] for surface, counts in surface_counts.items()
        }
        return self

    def predict_sentence(self, tokens: list[str]) -> list[str]:
        tags = ["O"] * len(tokens)
        i = 0
        while i < len(tokens):
            # Greedy longest match: prefer "Real Madrid" over "Real".
            for span in range(min(self.max_span_length, len(tokens) - i), 0, -1):
                candidate = tuple(tokens[i : i + span])
                ent_type = self.gazetteer.get(candidate)
                if ent_type is not None:
                    tags[i] = f"B-{ent_type}"
                    for j in range(i + 1, i + span):
                        tags[j] = f"I-{ent_type}"
                    i += span
                    break
            else:
                i += 1
        return tags

    def predict(self, sentences: list[Sentence]) -> list[list[str]]:
        return [self.predict_sentence(s.tokens) for s in sentences]


# ---------------------------------------------------------------------------
# 2. Conditional random field
# ---------------------------------------------------------------------------


def word_features(sent: Sentence, i: int) -> dict:
    """Orthographic, morphological and contextual features for one token.

    Spanish-specific choices worth defending in an interview:
    - Suffixes up to 3 characters capture inflectional morphology, which is far
      more informative in Spanish than in English.
    - The POS tag is used as-is *and* truncated to its first two characters.
      The corpus uses the fine-grained PAROLE tagset (NP, NC, AQ, VMI...), and
      the coarse prefix generalises better on rare tags.
    - Capitalisation is the single strongest signal for Spanish NER, so it is
      encoded for the token and both of its neighbours.
    """
    word = sent.tokens[i]
    pos = sent.pos_tags[i]

    features = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[:3]": word[:3],
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.hasdigit": any(c.isdigit() for c in word),
        "word.hashyphen": "-" in word,
        "word.length": min(len(word), 12),
        "pos": pos,
        "pos[:2]": pos[:2],
    }

    for offset in (-2, -1, 1, 2):
        j = i + offset
        prefix = f"{offset:+d}:"
        if 0 <= j < len(sent):
            neighbour = sent.tokens[j]
            features.update({
                f"{prefix}word.lower": neighbour.lower(),
                f"{prefix}word.istitle": neighbour.istitle(),
                f"{prefix}word.isupper": neighbour.isupper(),
                f"{prefix}pos": sent.pos_tags[j],
                f"{prefix}pos[:2]": sent.pos_tags[j][:2],
            })
        elif j < 0:
            features["BOS"] = True
        else:
            features["EOS"] = True

    return features


def sentence_features(sent: Sentence) -> list[dict]:
    return [word_features(sent, i) for i in range(len(sent))]


class CRFTagger:
    """Linear-chain CRF over hand-engineered features.

    A CRF beats independent per-token classification because it scores the
    whole tag sequence jointly, so it can learn structural constraints such as
    "I-PER never follows B-ORG" directly from the transition weights.
    """

    def __init__(
        self,
        algorithm: str = "lbfgs",
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 200,
    ) -> None:
        self.model = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            c1=c1,  # L1 -> drives useless features to exactly zero
            c2=c2,  # L2 -> keeps weights small and stable
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )

    def fit(self, sentences: list[Sentence]) -> CRFTagger:
        x = [sentence_features(s) for s in sentences]
        y = [s.ner_tags for s in sentences]
        self.model.fit(x, y)
        return self

    def predict(self, sentences: list[Sentence]) -> list[list[str]]:
        return self.model.predict([sentence_features(s) for s in sentences])

    def top_transitions(self, n: int = 10) -> list[tuple[tuple[str, str], float]]:
        """Highest-weighted tag transitions -- useful for the error analysis notebook."""
        return Counter(self.model.transition_features_).most_common(n)
