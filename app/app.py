"""Gradio demo for Spanish Named Entity Recognition.

Deployed on Hugging Face Spaces (free CPU tier). The app deliberately reuses
`predict_sentences` from the training package rather than the `pipeline` helper,
so the demo runs exactly the same inference code that produced the reported
metrics. A demo that disagrees with the evaluation is worse than no demo.

Run locally:
    python app/app.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402

from spanish_ner.data import Sentence, iter_entities  # noqa: E402
from spanish_ner.modeling import predict_sentences  # noqa: E402

# On Spaces this points at the model repo; locally at the trained checkpoint.
MODEL_PATH = os.environ.get(
    "NER_MODEL_PATH",
    str(Path(__file__).resolve().parents[1] / "models" / "bert-base-spanish-wwm-cased" / "best"),
)
MAX_LENGTH = 256

ENTITY_NAMES = {
    "PER": "Person",
    "ORG": "Organisation",
    "LOC": "Location",
    "MISC": "Miscellaneous",
}

# Matches the corpus tokenisation closely enough for free text: words, numbers
# and individual punctuation marks are separate tokens.
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

print(f"Loading model from {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Model ready.")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def recognise(text: str):
    """Return Gradio HighlightedText spans plus a plain-text entity summary."""
    text = (text or "").strip()
    if not text:
        return [], "Enter some Spanish text to begin."

    tokens = tokenize(text)
    if not tokens:
        return [], "No tokens found."

    sentence = Sentence(tokens=tokens, pos_tags=["X"] * len(tokens), ner_tags=["O"] * len(tokens))
    tags = predict_sentences([sentence], model, tokenizer, MAX_LENGTH, batch_size=1)[0]

    # Build display spans, marking non-entity stretches with None.
    spans: list[tuple[str, str | None]] = []
    entities: list[tuple[str, str]] = []
    cursor = 0
    for ent_type, start, end in iter_entities(tags):
        if start > cursor:
            spans.append((" ".join(tokens[cursor:start]) + " ", None))
        surface = " ".join(tokens[start:end])
        spans.append((surface, ent_type))
        entities.append((surface, ent_type))
        cursor = end
    if cursor < len(tokens):
        spans.append((" ".join(tokens[cursor:]), None))

    if not entities:
        summary = "No entities found."
    else:
        lines = [f"Found {len(entities)} entities:", ""]
        lines += [
            f"  {surface}  ->  {ENTITY_NAMES.get(t, t)} ({t})" for surface, t in entities
        ]
        summary = "\n".join(lines)

    return spans, summary


EXAMPLES = [
    "El Banco Central de Chile anuncio hoy en Santiago que Rosanna Costa presidira la reunion.",
    "Telefonica invirtio 500 millones de euros en Madrid durante la Copa America.",
    "Gabriel Boric se reunio con representantes de la Union Europea en Bruselas.",
    "La Universidad Andres Bello y CONICYT firmaron un convenio en Vina del Mar.",
]

with gr.Blocks(title="Spanish NER") as demo:
    gr.Markdown(
        """
        # Spanish Named Entity Recognition

        A BETO (`dccuchile/bert-base-spanish-wwm-cased`) model fine-tuned on
        CoNLL-2002 Spanish. It tags four entity types: **Person**,
        **Organisation**, **Location** and **Miscellaneous**.

        Trained and evaluated with a strict protocol: the test split was read
        once, at the end. Full results, baselines and known limitations are in
        the [GitHub repository](https://github.com/JoseElias23/spanish-ner-beto).
        """
    )

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Spanish text",
                placeholder="Escribe o pega un texto en espanol...",
                lines=5,
            )
            submit = gr.Button("Recognise entities", variant="primary")
        with gr.Column():
            highlighted = gr.HighlightedText(label="Tagged text", combine_adjacent=True)
            summary = gr.Textbox(label="Entities found", lines=8)

    gr.Examples(examples=EXAMPLES, inputs=text_input)

    submit.click(recognise, inputs=text_input, outputs=[highlighted, summary])
    text_input.submit(recognise, inputs=text_input, outputs=[highlighted, summary])

if __name__ == "__main__":
    demo.launch()
