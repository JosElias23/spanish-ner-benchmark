"""Entity-level evaluation.

NER is scored at the *entity* level, not the token level. A prediction counts
as correct only when the entity type and both boundaries match exactly.
Token-level accuracy is a misleading metric here: roughly 88% of tokens in
CoNLL-2002 are outside any entity, so a model that predicts "O" everywhere
already scores ~0.88 accuracy while finding nothing at all.

seqeval implements the exact scoring of the CoNLL shared-task script, which
keeps our numbers comparable with published results.
"""

from __future__ import annotations

import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from spanish_ner.data import Sentence


def evaluate_predictions(
    y_true: list[list[str]],
    y_pred: list[list[str]],
) -> dict:
    """Compute entity-level precision, recall and F1, overall and per class.

    Returns a plain dict so it can be serialised straight into reports/.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} gold sequences vs {len(y_pred)} predicted")
    for i, (t, p) in enumerate(zip(y_true, y_pred, strict=True)):
        if len(t) != len(p):
            raise ValueError(f"Sequence {i}: {len(t)} gold tags vs {len(p)} predicted tags")

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    per_type = {
        label: {
            "precision": round(scores["precision"], 4),
            "recall": round(scores["recall"], 4),
            "f1": round(scores["f1-score"], 4),
            "support": int(scores["support"]),
        }
        for label, scores in report.items()
        if label not in {"micro avg", "macro avg", "weighted avg"}
    }

    return {
        "overall": {
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        },
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "per_type": dict(sorted(per_type.items())),
        "n_sentences": len(y_true),
        "n_gold_entities": sum(v["support"] for v in per_type.values()),
    }


def gold_tags(sentences: list[Sentence]) -> list[list[str]]:
    return [s.ner_tags for s in sentences]


def format_table(results: dict, title: str = "") -> str:
    """Render a results dict as a Markdown table, ready to paste into a README."""
    lines = []
    if title:
        lines += [f"**{title}**", ""]
    lines += [
        "| Entity | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, s in results["per_type"].items():
        lines.append(
            f"| {label} | {s['precision']:.4f} | {s['recall']:.4f} | "
            f"{s['f1']:.4f} | {s['support']} |"
        )
    o = results["overall"]
    lines.append(
        f"| **Overall (micro)** | **{o['precision']:.4f}** | **{o['recall']:.4f}** | "
        f"**{o['f1']:.4f}** | **{results['n_gold_entities']}** |"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------


def entity_counts_per_sentence(
    y_true: list[list[str]], y_pred: list[list[str]]
) -> np.ndarray:
    """Per-sentence (true positives, predicted count, gold count).

    Micro-averaged precision, recall and F1 are ratios of sums over these three
    quantities, so a bootstrap resample only needs to re-sum the rows. That
    turns 10,000 resamples from hours of seqeval calls into milliseconds.
    """
    from spanish_ner.data import iter_entities

    counts = np.zeros((len(y_true), 3), dtype=np.int64)
    for i, (gold, pred) in enumerate(zip(y_true, y_pred, strict=True)):
        gold_spans = set(iter_entities(gold))
        pred_spans = set(iter_entities(pred))
        counts[i] = (len(gold_spans & pred_spans), len(pred_spans), len(gold_spans))
    return counts


def _micro_f1(counts: np.ndarray) -> float:
    tp, n_pred, n_gold = counts.sum(axis=0)
    if n_pred == 0 or n_gold == 0:
        return 0.0
    precision = tp / n_pred
    recall = tp / n_gold
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def holm_adjust(p_values: dict) -> dict:
    """Holm-Bonferroni step-down, for a family of comparisons run at once.

    Ten pairwise tests at alpha = 0.05 reject something about 40% of the time
    when every null is true. Reporting the smallest of them as if it were a
    single pre-registered test is the most common way a model comparison
    overstates itself, and it is the mistake this repository made: it ran four
    comparisons against a reference arm and printed "Significant: yes" on the
    one that cleared 0.05 by 0.0003.

    Holm is used rather than Bonferroni because it is uniformly more powerful
    and no less valid: sort ascending, multiply the k-th smallest by (m - k),
    then enforce monotonicity so an adjusted p can never fall below one that
    precedes it.

    Returns {key: adjusted_p} over the same keys it was given.
    """
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted, running = {}, 0.0
    for rank, (key, raw) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * raw))
        adjusted[key] = round(running, 4)
    return adjusted


def paired_bootstrap(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Paired bootstrap test for the F1 difference between two models.

    Both models are scored on the *same* resampled sentences, which controls for
    the fact that some sentences are simply harder than others. The reported
    p-value is the fraction of resamples in which model A fails to beat model B
    -- i.e. how often the observed ranking could flip on a different sample of
    the same size.

    A 1-point F1 gap on 1,500 sentences is frequently not significant. Reporting
    the interval rather than the point estimate is what makes a model comparison
    honest.
    """
    if counts_a.shape != counts_b.shape:
        raise ValueError("Both models must be scored on the same sentences")

    rng = np.random.default_rng(seed)
    n = len(counts_a)
    observed = _micro_f1(counts_a) - _micro_f1(counts_b)

    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = _micro_f1(counts_a[idx]) - _micro_f1(counts_b[idx])

    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(diffs, [alpha, 1 - alpha])
    tail = np.mean(diffs <= 0) if observed > 0 else np.mean(diffs >= 0)

    return {
        "f1_a": round(_micro_f1(counts_a), 4),
        "f1_b": round(_micro_f1(counts_b), 4),
        "observed_difference": round(observed, 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
        "confidence": confidence,
        # Two-sided: how often the difference changes sign or vanishes.
        "p_value": round(min(1.0, float(tail) * 2), 4),
        "significant": bool(lower > 0 or upper < 0),
        "n_resamples": n_resamples,
    }
