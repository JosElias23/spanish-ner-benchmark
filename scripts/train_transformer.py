"""Fine-tune a transformer for Spanish NER and evaluate on the development set.

Usage:
    python scripts/train_transformer.py                          # BETO (default)
    python scripts/train_transformer.py --model bert-base-multilingual-cased
    python scripts/train_transformer.py --model xlm-roberta-base --tag xlmr

The same script trains every model in the comparison table, so no result can
differ because of an accidental change in the training procedure. Only the
checkpoint name changes.

The test split is never loaded here. Model selection happens on dev.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from spanish_ner.data import ID2LABEL, LABEL2ID, LABELS, load_all  # noqa: E402
from spanish_ner.evaluate import evaluate_predictions, format_table, gold_tags  # noqa: E402
from spanish_ner.modeling import (  # noqa: E402
    TokenClassificationDataset,
    align_predictions,
    predict_sentences,
)
from spanish_ner.utils import (  # noqa: E402
    PROJECT_ROOT,
    load_config,
    save_json,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="HF checkpoint (default: config value)")
    parser.add_argument("--tag", default=None, help="Short name used in output paths")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--label-all-subtokens",
        action="store_true",
        help="Label every sub-word instead of only the first one (ablation)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging()
    config = load_config()
    set_seed(config["seed"])

    cfg = config["transformer"]
    model_name = args.model or cfg["model_name"]
    tag = args.tag or model_name.split("/")[-1]
    epochs = args.epochs or cfg["num_epochs"]
    lr = args.lr or cfg["learning_rate"]
    batch_size = args.batch_size or cfg["batch_size"]
    label_all = args.label_all_subtokens or cfg["label_all_subtokens"]
    if label_all:
        tag = f"{tag}-allsub"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Model: %s | device: %s | seed: %d", model_name, device, config["seed"])
    if device == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    # --- data ------------------------------------------------------------
    splits = load_all(config)
    train_sents, dev_sents = splits["train"], splits["dev"]

    # add_prefix_space is required by byte-level BPE tokenizers (RoBERTa family)
    # when the input is already split into words. It is ignored by WordPiece.
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    train_ds = TokenClassificationDataset(
        train_sents, tokenizer, cfg["max_length"], label_all_subtokens=label_all
    )
    dev_ds = TokenClassificationDataset(
        dev_sents, tokenizer, cfg["max_length"], label_all_subtokens=label_all
    )

    truncation = train_ds.truncation_report()
    log.info("Truncation at max_length=%d: %d sentences, %d words lost",
             truncation["max_length"], truncation["sentences_truncated"],
             truncation["words_lost"])

    # --- model -----------------------------------------------------------
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,   # baked into the checkpoint so inference cannot
        label2id=LABEL2ID,   # silently use a different mapping
    )

    output_dir = PROJECT_ROOT / config["paths"]["models_dir"] / tag

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        y_true, y_pred = align_predictions(logits, labels)
        res = evaluate_predictions(y_true, y_pred)
        return {
            "precision": res["overall"]["precision"],
            "recall": res["overall"]["recall"],
            "f1": res["overall"]["f1"],
            "macro_f1": res["macro_f1"],
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        seed=config["seed"],
        data_seed=config["seed"],
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=cfg["eval_batch_size"],
        num_train_epochs=epochs,
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        # Keep the epoch that maximises dev F1, not the last one. Without this,
        # a model that peaks at epoch 2 and overfits afterwards gets reported at
        # its worst.
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        bf16=(device == "cuda"),
        report_to="none",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    t0 = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - t0
    log.info("Training finished in %.1fs (%.1f min)", train_seconds, train_seconds / 60)

    # --- evaluate --------------------------------------------------------
    # Re-predict through the same code path the demo app uses, rather than
    # trusting the Trainer's internal loop. If the two ever disagree, the
    # deployed model is not the one that was measured.
    y_pred = predict_sentences(
        dev_sents, trainer.model, tokenizer, cfg["max_length"], cfg["eval_batch_size"]
    )
    results = evaluate_predictions(gold_tags(dev_sents), y_pred)
    results["model_name"] = model_name
    results["train_seconds"] = round(train_seconds, 1)
    results["epochs"] = epochs
    results["learning_rate"] = lr
    results["batch_size"] = batch_size
    results["label_all_subtokens"] = label_all
    results["truncation"] = truncation
    results["n_parameters"] = sum(p.numel() for p in trainer.model.parameters())
    results["epoch_history"] = [
        {k: v for k, v in entry.items() if k.startswith("eval_") or k == "epoch"}
        for entry in trainer.state.log_history
        if "eval_f1" in entry
    ]

    save_json(results, f"reports/metrics_{tag}_dev.json")

    final_dir = output_dir / "best"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    log.info("Best model saved to %s", final_dir)

    print()
    print(format_table(results, title=f"{tag} -- development set (esp.testa)"))
    print()
    print("Dev F1 by epoch:", json.dumps(
        [round(e["eval_f1"], 4) for e in results["epoch_history"]]
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
