"""Error analysis and figures for the README.

Produces:
  reports/figures/model_comparison.png   F1 by model, full vs deduplicated test
  reports/figures/f1_by_entity_type.png  per-class F1 across models
  reports/figures/error_breakdown.png    what the deployed model actually gets wrong
  reports/error_analysis.json            the same numbers in machine-readable form

Aggregate F1 says a model is good. It does not say *how* it fails, which is the
only thing that tells you what to fix next. Every error is bucketed into one of
four categories:

  boundary   right type, wrong span     (predicting "el Banco Central" for "Banco Central")
  type       right span, wrong type     (tagging Barcelona as LOC when the gold says ORG)
  spurious   an entity predicted where the gold annotation has none
  missed     a gold entity not predicted at all

Usage:
    python scripts/error_analysis.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402

from spanish_ner.data import deduplicate_against, iter_entities, load_all  # noqa: E402
from spanish_ner.modeling import predict_sentences  # noqa: E402
from spanish_ner.utils import (  # noqa: E402
    PROJECT_ROOT,
    load_config,
    save_json,
    set_seed,
    setup_logging,
)

PALETTE = {
    "gazetteer": "#9aa0a6",
    "crf": "#5f6caf",
    "bert-base-spanish-wwm-cased": "#d1495b",
    "mbert": "#00798c",
    "xlmr": "#edae49",
}
# Ordered by increasing modelling power, so the figures read left to right as
# the story of the project rather than as an alphabetical list.
MODEL_ORDER = ["gazetteer", "crf", "bert-base-spanish-wwm-cased", "mbert", "xlmr"]

PRETTY = {
    "gazetteer": "Gazetteer",
    "crf": "CRF",
    "bert-base-spanish-wwm-cased": "BETO",
    "mbert": "mBERT",
    "xlmr": "XLM-R",
}


def classify_errors(sentences, predictions) -> tuple[Counter, list[dict]]:
    """Bucket every disagreement between gold and predicted entity spans."""
    categories: Counter[str] = Counter()
    examples: list[dict] = []

    for sent, pred_tags in zip(sentences, predictions, strict=True):
        gold = {(s, e): t for t, s, e in iter_entities(sent.ner_tags)}
        pred = {(s, e): t for t, s, e in iter_entities(pred_tags)}

        for span, gold_type in gold.items():
            if span in pred:
                if pred[span] != gold_type:
                    categories["type"] += 1
                    examples.append({
                        "category": "type",
                        "text": " ".join(sent.tokens[span[0]:span[1]]),
                        "gold": gold_type,
                        "predicted": pred[span],
                        "context": " ".join(sent.tokens[max(0, span[0] - 5):span[1] + 5]),
                    })
                continue

            # Overlap with a predicted span means the model found something
            # there but drew the boundaries differently.
            overlapping = [p for p in pred if not (p[1] <= span[0] or p[0] >= span[1])]
            if overlapping:
                categories["boundary"] += 1
                p = overlapping[0]
                examples.append({
                    "category": "boundary",
                    "gold_text": " ".join(sent.tokens[span[0]:span[1]]),
                    "predicted_text": " ".join(sent.tokens[p[0]:p[1]]),
                    "gold": gold_type,
                    "predicted": pred[p],
                })
            else:
                categories["missed"] += 1
                examples.append({
                    "category": "missed",
                    "text": " ".join(sent.tokens[span[0]:span[1]]),
                    "gold": gold_type,
                    "context": " ".join(sent.tokens[max(0, span[0] - 5):span[1] + 5]),
                })

        for span, pred_type in pred.items():
            if span in gold:
                continue
            if any(not (g[1] <= span[0] or g[0] >= span[1]) for g in gold):
                continue  # already counted as a boundary error
            categories["spurious"] += 1
            examples.append({
                "category": "spurious",
                "text": " ".join(sent.tokens[span[0]:span[1]]),
                "predicted": pred_type,
                "context": " ".join(sent.tokens[max(0, span[0] - 5):span[1] + 5]),
            })

    return categories, examples


def ordered_models(metrics: dict) -> list[str]:
    present = metrics["per_model"]
    known = [m for m in MODEL_ORDER if m in present]
    return known + [m for m in present if m not in MODEL_ORDER]


def plot_model_comparison(metrics: dict, out: Path) -> None:
    models = ordered_models(metrics)
    full = [metrics["per_model"][m]["full"]["overall"]["f1"] for m in models]
    dedup = [metrics["per_model"][m]["deduplicated"]["overall"]["f1"] for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(models))
    width = 0.38
    colors = [PALETTE.get(m, "#888888") for m in models]
    ax.bar([i - width / 2 for i in x], full, width, color=colors)
    ax.bar([i + width / 2 for i in x], dedup, width, color=colors, alpha=0.5)

    for i, (f, d) in enumerate(zip(full, dedup, strict=True)):
        ax.text(i - width / 2, f + 0.013, f"{f:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, d + 0.013, f"{d:.3f}", ha="center", fontsize=9, alpha=0.7)

    ax.set_xticks(list(x))
    ax.set_xticklabels([PRETTY.get(m, m) for m in models])
    ax.set_ylabel("Entity-level F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Spanish NER on CoNLL-2002 held-out test set (esp.testb)")
    # Each model keeps its own colour, so the legend explains the two shades
    # rather than repeating a single model's colour twice.
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#555555", label="Full test set (1,517 sentences)"),
            Patch(facecolor="#555555", alpha=0.5,
                  label="Deduplicated (1,208 unseen in training)"),
        ],
        loc="lower right",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_f1_by_type(metrics: dict, out: Path) -> None:
    models = ordered_models(metrics)
    types = ["LOC", "MISC", "ORG", "PER"]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(models)
    for j, model in enumerate(models):
        per_type = metrics["per_model"][model]["full"]["per_type"]
        scores = [per_type.get(t, {}).get("f1", 0.0) for t in types]
        offset = (j - (len(models) - 1) / 2) * width
        ax.bar([i + offset for i in range(len(types))], scores, width,
               label=PRETTY.get(model, model), color=PALETTE.get(model, "#888888"))

    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types)
    ax.set_ylabel("Entity-level F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-entity-type F1 on the test set: MISC is hardest for every model")
    ax.legend(ncol=len(models), fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_error_breakdown(categories: Counter, model_label: str, out: Path) -> None:
    order = ["missed", "spurious", "boundary", "type"]
    labels = {
        "missed": "Missed\nentity not found",
        "spurious": "Spurious\nentity invented",
        "boundary": "Boundary\nwrong span",
        "type": "Type\nwrong class",
    }
    values = [categories.get(c, 0) for c in order]
    total = sum(values) or 1

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh([labels[c] for c in order], values, color="#d1495b", alpha=0.85)
    for bar, v in zip(bars, values, strict=True):
        ax.text(bar.get_width() + total * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{v}   {v / total:.0%}", va="center", fontsize=10)
    ax.set_xlabel("Errors on the test set")
    ax.set_xlim(0, max(values) * 1.28)
    ax.set_title(f"What {model_label} gets wrong ({total} errors in total)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    log = setup_logging()
    config = load_config()
    set_seed(config["seed"])

    metrics_path = PROJECT_ROOT / "reports" / "metrics_test.json"
    if not metrics_path.exists():
        log.error("reports/metrics_test.json not found. Run scripts/evaluate_test.py first.")
        return 1
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    figures = PROJECT_ROOT / config["paths"]["figures_dir"]
    figures.mkdir(parents=True, exist_ok=True)

    plot_model_comparison(metrics, figures / "model_comparison.png")
    plot_f1_by_type(metrics, figures / "f1_by_entity_type.png")
    log.info("Comparison figures written to %s", figures)

    splits = load_all(config, include_test=True)
    test = splits["test"]
    checkpoint = (
        PROJECT_ROOT / "models" / config["transformer"]["model_name"].split("/")[-1] / "best"
    )
    if not checkpoint.exists():
        log.error("Checkpoint %s not found. Run scripts/train_transformer.py first.", checkpoint)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    predictions = predict_sentences(
        test,
        model,
        tokenizer,
        config["transformer"]["max_length"],
        config["transformer"]["eval_batch_size"],
    )

    categories, examples = classify_errors(test, predictions)
    plot_error_breakdown(categories, "BETO", figures / "error_breakdown.png")

    total_errors = sum(categories.values())
    log.info("BETO test errors: %s (total %d)", dict(categories), total_errors)

    save_json(
        {
            "model": config["transformer"]["model_name"],
            "split": "test (esp.testb)",
            "total_errors": total_errors,
            "by_category": dict(categories),
            "by_category_pct": {
                k: round(100 * v / total_errors, 1) for k, v in categories.items()
            },
            "n_test_sentences": len(test),
            "n_deduplicated": len(deduplicate_against(test, splits["train"])),
            "sample_errors": examples[:40],
        },
        "reports/error_analysis.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
