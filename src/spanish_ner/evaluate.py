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
