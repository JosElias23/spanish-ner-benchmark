"""Transformer fine-tuning for token classification.

The central problem this module solves is *label alignment*. Our corpus is
annotated per word, but BERT tokenises into sub-words:

    words      : ["Telefónica", "invirtió"]
    labels     : ["B-ORG",      "O"       ]
    sub-tokens : ["Telef", "##ónica", "invir", "##tió"]

There is no longer a one-to-one correspondence, so the labels must be
re-projected onto sub-tokens before training. Getting this wrong is the classic
silent bug in NER fine-tuning: the model trains happily and scores badly.

We label the *first* sub-token of each word and assign -100 to the rest.
PyTorch's cross-entropy ignores -100, so continuation sub-tokens contribute no
loss and no gradient. The alternative -- repeating the label on every sub-token
-- over-weights long words in the loss and makes decoding ambiguous when
sub-tokens of the same word disagree. Both strategies are implemented so the
choice can be justified with an experiment rather than an assertion.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from spanish_ner.data import ID2LABEL, LABEL2ID, Sentence

IGNORE_INDEX = -100


def encode_sentence(
    sentence: Sentence,
    tokenizer,
    max_length: int,
    label_all_subtokens: bool = False,
    with_labels: bool = True,
) -> dict:
    """Tokenise one pre-split sentence and project word labels onto sub-tokens."""
    encoding = tokenizer(
        sentence.tokens,
        is_split_into_words=True,  # the corpus is already tokenised; do not re-split
        truncation=True,
        max_length=max_length,
    )

    if not with_labels:
        return dict(encoding)

    word_ids = encoding.word_ids()
    labels: list[int] = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # [CLS], [SEP] and padding carry no label.
            labels.append(IGNORE_INDEX)
        elif word_id != previous_word_id:
            # First sub-token of a word: carries the word's label.
            labels.append(LABEL2ID[sentence.ner_tags[word_id]])
        elif label_all_subtokens:
            # Continuation sub-token. A B- label must become I- so the decoded
            # sequence stays valid IOB2.
            tag = sentence.ner_tags[word_id]
            labels.append(LABEL2ID["I" + tag[1:] if tag.startswith("B-") else tag])
        else:
            labels.append(IGNORE_INDEX)
        previous_word_id = word_id

    encoding = dict(encoding)
    encoding["labels"] = labels
    return encoding


class TokenClassificationDataset(Dataset):
    """Thin wrapper so the Trainer can consume our Sentence objects directly.

    We keep the original Sentence list alongside the encodings: at prediction
    time we need each sub-token's word id to map logits back to words, and we
    need the true word count to detect sentences lost to truncation.
    """

    def __init__(
        self,
        sentences: list[Sentence],
        tokenizer,
        max_length: int,
        label_all_subtokens: bool = False,
        with_labels: bool = True,
    ) -> None:
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = [
            encode_sentence(s, tokenizer, max_length, label_all_subtokens, with_labels)
            for s in sentences
        ]

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict:
        return self.encodings[idx]

    def truncation_report(self) -> dict:
        """How many words were cut off by max_length.

        Truncated words are unrecoverable: their gold entities can never be
        predicted, which caps recall. This must be reported, not assumed to be
        zero.
        """
        lost_sentences = 0
        lost_words = 0
        for sent in self.sentences:
            encoding = self.tokenizer(
                sent.tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
            )
            covered = {w for w in encoding.word_ids() if w is not None}
            missing = len(sent) - len(covered)
            if missing > 0:
                lost_sentences += 1
                lost_words += missing
        return {
            "sentences_truncated": lost_sentences,
            "words_lost": lost_words,
            "max_length": self.max_length,
        }


def align_predictions(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert sub-token logits back into word-level IOB2 tag sequences.

    Positions labelled -100 (special tokens and continuation sub-tokens) are
    dropped, which leaves exactly one prediction per original word.
    """
    predictions = np.argmax(logits, axis=-1)
    true_tags, pred_tags = [], []

    for pred_row, label_row in zip(predictions, labels, strict=True):
        keep = label_row != IGNORE_INDEX
        true_tags.append([ID2LABEL[int(i)] for i in label_row[keep]])
        pred_tags.append([ID2LABEL[int(i)] for i in pred_row[keep]])

    return true_tags, pred_tags


def predict_sentences(
    sentences: list[Sentence],
    model,
    tokenizer,
    max_length: int,
    batch_size: int = 64,
    device: str | None = None,
) -> list[list[str]]:
    """Predict word-level tags, guaranteeing one tag per input word.

    Words beyond max_length receive "O". They are unpredictable by
    construction, and silently returning a short sequence would crash the
    evaluator or, worse, misalign it.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    all_tags: list[list[str]] = []

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        encodings = [
            tokenizer(s.tokens, is_split_into_words=True, truncation=True, max_length=max_length)
            for s in batch
        ]
        padded = tokenizer.pad(encodings, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**padded).logits.cpu().numpy()

        for sent, enc, row in zip(batch, encodings, logits, strict=True):
            tags = ["O"] * len(sent)
            seen: set[int] = set()
            for position, word_id in enumerate(enc.word_ids()):
                if word_id is not None and word_id not in seen:
                    seen.add(word_id)
                    tags[word_id] = ID2LABEL[int(np.argmax(row[position]))]
            all_tags.append(tags)

    return all_tags
