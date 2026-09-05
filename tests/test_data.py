"""Integrity tests for the CoNLL-2002 Spanish corpus pipeline.

These are not decorative. Each one guards a failure mode that silently
produces a plausible-but-wrong metric.
"""

import pytest

from spanish_ner.data import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    Sentence,
    deduplicate_against,
    iter_entities,
    sentence_key,
    split_overlap,
)

# Reference statistics from Tjong Kim Sang (2002), "Introduction to the
# CoNLL-2002 Shared Task". If our parser disagrees with the paper, the parser
# is wrong.
EXPECTED_SENTENCES = {"train": 8323, "dev": 1915, "test": 1517}
EXPECTED_ENTITIES = {"train": 18798, "dev": 4352, "test": 3559}


class TestLabelSchema:
    def test_label_set_is_exactly_nine_labels(self):
        assert len(LABELS) == 9  # O + 4 entity types x {B, I}

    def test_outside_label_is_index_zero(self):
        # Many downstream tools assume O == 0; pin it explicitly.
        assert LABEL2ID["O"] == 0

    def test_mapping_is_bijective(self):
        assert len(LABEL2ID) == len(ID2LABEL) == len(LABELS)
        assert all(ID2LABEL[LABEL2ID[lbl]] == lbl for lbl in LABELS)

    def test_label_order_is_deterministic(self):
        # A model saved with one ordering and loaded with another would produce
        # confidently wrong predictions, so the order is frozen by this test.
        assert LABELS == (
            "O",
            "B-LOC", "I-LOC",
            "B-MISC", "I-MISC",
            "B-ORG", "I-ORG",
            "B-PER", "I-PER",
        )


class TestParsing:
    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_sentence_count_matches_published_corpus(self, splits, split):
        assert len(splits[split]) == EXPECTED_SENTENCES[split]

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_entity_count_matches_published_corpus(self, splits, split):
        total = sum(len(list(iter_entities(s.ner_tags))) for s in splits[split])
        assert total == EXPECTED_ENTITIES[split]

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_no_ragged_sentences(self, splits, split):
        for sent in splits[split]:
            assert len(sent.tokens) == len(sent.pos_tags) == len(sent.ner_tags)

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_no_empty_sentences(self, splits, split):
        assert all(len(s) > 0 for s in splits[split])

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_all_tags_are_in_the_schema(self, splits, split):
        for sent in splits[split]:
            for tag in sent.ner_tags:
                assert tag in LABEL2ID

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_encoding_is_utf8_not_mojibake(self, splits, split):
        """Reading the files as latin-1 corrupts ~27k accented tokens.

        The corruption is silent: training still runs, the score is just worse.
        This test is the tripwire.
        """
        corrupt = [
            tok
            for sent in splits[split]
            for tok in sent.tokens
            if "Ã" in tok or "\ufffd" in tok
        ]
        assert not corrupt, f"{len(corrupt)} mojibake tokens, e.g. {corrupt[:5]}"

    def test_ragged_sentence_is_rejected(self):
        with pytest.raises(ValueError):
            Sentence(tokens=["a", "b"], pos_tags=["NP"], ner_tags=["O", "O"])


class TestSplitOverlap:
    """The official CoNLL-2002 splits are NOT disjoint.

    309 of 1,517 test sentences occur verbatim in training. Most are newswire
    boilerplate, but they carry 9.4% of the test entities. We cannot fix the
    benchmark without losing comparability with published results, so instead
    we pin the overlap here: if these numbers ever move, the raw data changed
    and every metric in reports/ must be regenerated.
    """

    def test_train_test_overlap_is_the_known_amount(self, splits):
        overlap = split_overlap(splits["train"], splits["test"])
        duplicated = [s for s in splits["test"] if sentence_key(s) in overlap]
        assert len(duplicated) == 309

    def test_overlap_is_dominated_by_short_boilerplate(self, splits):
        overlap = split_overlap(splits["train"], splits["test"])
        duplicated = [s for s in splits["test"] if sentence_key(s) in overlap]
        substantive = [s for s in duplicated if len(s) >= 10]
        assert len(substantive) == 86

    def test_deduplicated_test_set_has_the_expected_size(self, splits):
        clean = deduplicate_against(splits["test"], splits["train"])
        assert len(clean) == 1517 - 309

    def test_deduplication_removes_no_entity_types_entirely(self, splits):
        """A stricter test set is only useful if it still covers every class."""
        clean = deduplicate_against(splits["test"], splits["train"])
        types = {t for s in clean for t, _, _ in iter_entities(s.ner_tags)}
        assert types == {"LOC", "MISC", "ORG", "PER"}

    def test_deduplication_is_idempotent(self, splits):
        once = deduplicate_against(splits["test"], splits["train"])
        twice = deduplicate_against(once, splits["train"])
        assert len(once) == len(twice)


class TestEntityDecoding:
    def test_simple_span(self):
        tags = ["O", "B-PER", "I-PER", "O"]
        assert list(iter_entities(tags)) == [("PER", 1, 3)]

    def test_two_adjacent_entities_of_same_type(self):
        # B- immediately after I- must open a NEW entity, not extend the old one.
        tags = ["B-ORG", "I-ORG", "B-ORG"]
        assert list(iter_entities(tags)) == [("ORG", 0, 2), ("ORG", 2, 3)]

    def test_entity_at_end_of_sequence_is_closed(self):
        assert list(iter_entities(["O", "B-LOC"])) == [("LOC", 1, 2)]

    def test_type_change_without_b_prefix_splits_the_span(self):
        assert list(iter_entities(["B-PER", "I-LOC"])) == [("PER", 0, 1), ("LOC", 1, 2)]

    def test_all_outside_yields_nothing(self):
        assert list(iter_entities(["O", "O", "O"])) == []

    def test_empty_sequence_yields_nothing(self):
        assert list(iter_entities([])) == []
