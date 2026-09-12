"""Tests for the significance protocol, after an audit found it selected on test.

`scripts/evaluate_test.py` used to rank the five models by their **test** F1 and
then compare every other model against the winner. The reference arm was
therefore chosen by its score on the same data the p-values come from, and the
four resulting comparisons were printed as though each were a single
pre-registered test. One of them, mBERT vs XLM-R, cleared 0.05 by 0.008 and was
marked "Significant: yes"; under Holm it is 0.126.

Two things replaced it and both are tested here: a confirmatory comparison named
in the source before any score is seen, and a Holm correction over the family of
exploratory ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spanish_ner.evaluate import holm_adjust

ROOT = Path(__file__).resolve().parents[1]


class TestHolmAdjust:
    def test_a_single_test_is_unchanged(self):
        """Nothing to correct for when nothing else was run."""
        assert holm_adjust({"a": 0.0312}) == {"a": 0.0312}

    def test_an_empty_family_is_empty(self):
        assert holm_adjust({}) == {}

    def test_the_smallest_p_is_multiplied_by_the_family_size(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.2, "c": 0.5, "d": 0.9})
        assert adjusted["a"] == pytest.approx(0.04)

    def test_adjusted_p_values_never_decrease_along_the_ordering(self):
        """The step-down monotonicity rule, which is what makes Holm valid."""
        raw = {"a": 0.001, "b": 0.02, "c": 0.021, "d": 0.3, "e": 0.31}
        adjusted = holm_adjust(raw)
        in_raw_order = [adjusted[k] for k in sorted(raw, key=lambda k: raw[k])]
        assert in_raw_order == sorted(in_raw_order)

    def test_no_adjusted_p_is_ever_below_its_raw_value(self):
        raw = {"a": 0.04, "b": 0.001, "c": 0.6, "d": 0.22, "e": 0.05}
        adjusted = holm_adjust(raw)
        assert all(adjusted[k] >= raw[k] - 1e-12 for k in raw)

    def test_it_is_never_more_conservative_than_bonferroni(self):
        """Holm is used instead of Bonferroni because it is uniformly better."""
        raw = {"a": 0.004, "b": 0.011, "c": 0.03, "d": 0.4}
        adjusted = holm_adjust(raw)
        assert all(adjusted[k] <= min(1.0, len(raw) * raw[k]) + 1e-12 for k in raw)

    def test_the_published_borderline_result_does_not_survive(self):
        """The comparison this correction exists for.

        mBERT vs XLM-R was p = 0.042 in a family of ten and was published as
        significant. Ten comparisons at 0.05 reject something about 40% of the
        time when every null is true.
        """
        family = {
            "beto_vs_crf": 0.0, "beto_vs_gazetteer": 0.0, "crf_vs_gazetteer": 0.0,
            "crf_vs_mbert": 0.0, "crf_vs_xlmr": 0.0, "gazetteer_vs_mbert": 0.0,
            "gazetteer_vs_xlmr": 0.0, "mbert_vs_xlmr": 0.042,
            "beto_vs_xlmr": 0.1514, "beto_vs_mbert": 0.7518,
        }
        adjusted = holm_adjust(family)
        assert adjusted["mbert_vs_xlmr"] > 0.05, (
            "the borderline comparison must not survive the correction"
        )
        assert adjusted["beto_vs_crf"] < 0.05, (
            "the large, unambiguous differences must still survive it"
        )


class TestTheProtocolIsNotChosenFromTheResults:
    """The defect was structural, so the guards are structural too."""

    def test_the_confirmatory_comparison_is_a_literal_in_the_source(self):
        """Named before any score is seen, not derived from one."""
        source = (ROOT / "scripts" / "evaluate_test.py").read_text(encoding="utf-8")
        assert 'CONFIRMATORY = ("bert-base-spanish-wwm-cased", "mbert")' in source

    def test_no_arm_is_selected_by_sorting_on_a_score(self):
        """The exact shape of the original defect, kept out by assertion.

        `sorted(results, key=... f1 ...)` picking a reference arm is what made
        every published p-value optimistic. If it comes back, so does the bias.
        """
        source = (ROOT / "scripts" / "evaluate_test.py").read_text(encoding="utf-8")
        offending = [
            line for line in source.splitlines()
            if "sorted(" in line and "f1" in line and not line.strip().startswith("#")
        ]
        assert not offending, (
            "a model is being ranked by score to choose a comparison arm: "
            f"{offending}"
        )


class TestTheReportRecordsWhatWasCompared:
    """Skipped without reports/, so a fresh clone can still run pytest."""

    @pytest.fixture(scope="class")
    def metrics(self):
        path = ROOT / "reports" / "metrics_test.json"
        if not path.exists():
            pytest.skip("reports/metrics_test.json not generated yet")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_the_confirmatory_comparison_is_marked_as_prespecified(self, metrics):
        confirmatory = metrics.get("confirmatory")
        if confirmatory is None:
            pytest.skip("run evaluate_test.py with all three transformers")
        assert confirmatory["prespecified"] is True

    def test_every_exploratory_comparison_carries_an_adjusted_p(self, metrics):
        for key, result in metrics["significance"].items():
            assert "p_value_holm" in result, f"{key} has no Holm-adjusted p-value"

    def test_significance_is_decided_on_the_adjusted_p(self, metrics):
        for key, result in metrics["significance"].items():
            assert result["significant"] == (result["p_value_holm"] < 0.05), (
                f"{key} is flagged from its unadjusted p-value")

    def test_the_three_split_sizes_are_all_recorded(self, metrics):
        """1,517 total, 1,208 absent from training, 1,166 distinct and absent.

        The prose used to quote the middle one while the code scored the last.
        """
        sizes = metrics["split_sizes"]
        assert sizes["test_sentences"] > sizes["absent_from_training"]
        assert sizes["absent_from_training"] > sizes["distinct_and_absent_scored"]
