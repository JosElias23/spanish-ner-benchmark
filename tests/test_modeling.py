"""Tests for sub-word label alignment.

Misaligned labels are the classic silent failure in NER fine-tuning: training
converges, loss goes down, and the model is quietly learning the wrong target.
These tests pin the alignment contract.
"""

import numpy as np
import pytest
from transformers import AutoTokenizer

from spanish_ner.data import ID2LABEL, LABEL2ID, Sentence
from spanish_ner.modeling import (
    IGNORE_INDEX,
    align_predictions,
    encode_sentence,
)

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture
def sentence():
    # "Telefónica" splits into several sub-tokens; "en" and "Madrid" do not.
    return Sentence(
        tokens=["Telefónica", "invirtió", "en", "Madrid"],
        pos_tags=["NP", "VMI", "SP", "NP"],
        ner_tags=["B-ORG", "O", "O", "B-LOC"],
    )


class TestEncodeSentence:
    def test_one_label_per_input_token(self, sentence, tokenizer):
        enc = encode_sentence(sentence, tokenizer, max_length=128)
        labelled = [lbl for lbl in enc["labels"] if lbl != IGNORE_INDEX]
        assert len(labelled) == len(sentence.tokens)

    def test_labels_are_in_original_order(self, sentence, tokenizer):
        enc = encode_sentence(sentence, tokenizer, max_length=128)
        recovered = [ID2LABEL[i] for i in enc["labels"] if i != IGNORE_INDEX]
        assert recovered == sentence.ner_tags

    def test_labels_and_input_ids_have_equal_length(self, sentence, tokenizer):
        enc = encode_sentence(sentence, tokenizer, max_length=128)
        assert len(enc["labels"]) == len(enc["input_ids"])

    def test_special_tokens_are_ignored(self, sentence, tokenizer):
        enc = encode_sentence(sentence, tokenizer, max_length=128)
        # [CLS] at the start and [SEP] at the end must never carry a label.
        assert enc["labels"][0] == IGNORE_INDEX
        assert enc["labels"][-1] == IGNORE_INDEX

    def test_the_word_actually_splits_into_subtokens(self, sentence, tokenizer):
        """Guard the premise of these tests: if nothing splits, they prove nothing."""
        enc = encode_sentence(sentence, tokenizer, max_length=128)
        word_ids = [w for w in tokenizer(
            sentence.tokens, is_split_into_words=True
        ).word_ids() if w is not None]
        assert len(word_ids) > len(sentence.tokens)
        assert sum(1 for lbl in enc["labels"] if lbl == IGNORE_INDEX) > 2

    def test_label_all_subtokens_converts_b_to_i(self, sentence, tokenizer):
        """Continuation sub-tokens of a B- word must become I-, not stay B-.

        Two adjacent B-ORG tags decode as two separate entities, which would
        corrupt the target sequence.
        """
        enc = encode_sentence(sentence, tokenizer, max_length=128, label_all_subtokens=True)
        tags = [ID2LABEL[i] for i in enc["labels"] if i != IGNORE_INDEX]
        assert tags[0] == "B-ORG"
        assert all(t == "I-ORG" for t in tags[1 : tags.index("O")])
        assert tags.count("B-ORG") == 1

    def test_label_all_subtokens_labels_every_subtoken(self, sentence, tokenizer):
        enc = encode_sentence(sentence, tokenizer, max_length=128, label_all_subtokens=True)
        # Only [CLS] and [SEP] remain ignored.
        assert sum(1 for lbl in enc["labels"] if lbl == IGNORE_INDEX) == 2

    def test_truncation_never_produces_ragged_output(self, sentence, tokenizer):
        enc = encode_sentence(sentence, tokenizer, max_length=4)
        assert len(enc["labels"]) == len(enc["input_ids"]) <= 4


class TestAlignPredictions:
    def test_ignored_positions_are_dropped(self):
        labels = np.array([[IGNORE_INDEX, LABEL2ID["B-PER"], IGNORE_INDEX, LABEL2ID["O"]]])
        logits = np.zeros((1, 4, len(ID2LABEL)))
        logits[0, 1, LABEL2ID["B-PER"]] = 10.0
        logits[0, 3, LABEL2ID["O"]] = 10.0
        # A confident but ignored position must not leak into the output.
        logits[0, 0, LABEL2ID["B-LOC"]] = 99.0

        true_tags, pred_tags = align_predictions(logits, labels)
        assert true_tags == [["B-PER", "O"]]
        assert pred_tags == [["B-PER", "O"]]

    def test_gold_and_predicted_sequences_have_equal_length(self):
        labels = np.array([[IGNORE_INDEX, 1, 2, IGNORE_INDEX, 0]])
        logits = np.random.RandomState(0).randn(1, 5, len(ID2LABEL))
        true_tags, pred_tags = align_predictions(logits, labels)
        assert len(true_tags[0]) == len(pred_tags[0]) == 3
