"""Tests for the inference service.

These are contract tests, not model tests. They check that the service returns
what its schema promises, that its probes mean what an orchestrator assumes
they mean, and that the awkward inputs a real caller sends do not crash it.

The whole suite is skipped when no trained checkpoint is present, so a fresh
clone runs green before anything has been trained.
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "models" / "bert-base-spanish-wwm-cased" / "best"
)

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="No trained checkpoint; run scripts/train_transformer.py first",
)


@pytest.fixture(scope="module")
def client():
    os.environ.setdefault("NER_MODEL_PATH", str(CHECKPOINT))
    from serve.api import app

    # The context manager triggers the lifespan handler, which loads the model.
    with TestClient(app) as test_client:
        yield test_client


class TestProbes:
    def test_health_is_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ready_reports_the_loaded_model(self, client):
        body = client.get("/ready").json()
        assert body["ready"] is True
        assert body["device"] in {"cpu", "cuda"}

    def test_every_response_carries_a_timing_header(self, client):
        response = client.get("/health")
        assert float(response.headers["X-Process-Time-Ms"]) >= 0


class TestExtraction:
    def test_finds_the_obvious_entities(self, client):
        response = client.post("/extract", json={
            "texts": ["El Banco Central de Chile anuncio hoy en Santiago."]
        })
        assert response.status_code == 200

        entities = response.json()["documents"][0]["entities"]
        types = {e["type"] for e in entities}
        assert "ORG" in types
        assert "LOC" in types

    def test_one_document_in_one_document_out(self, client):
        texts = ["Telefonica invirtio en Madrid.", "Gabriel Boric viajo a Bruselas.", "Hola."]
        body = client.post("/extract", json={"texts": texts}).json()
        assert len(body["documents"]) == len(texts)
        assert [d["text"] for d in body["documents"]] == texts

    def test_entity_offsets_are_ordered_and_non_empty(self, client):
        body = client.post("/extract", json={
            "texts": ["Rosanna Costa preside el Banco Central de Chile en Santiago."]
        }).json()
        for entity in body["documents"][0]["entities"]:
            assert entity["start_token"] < entity["end_token"]
            assert entity["text"].strip()

    def test_latency_is_reported(self, client):
        body = client.post("/extract", json={"texts": ["Madrid."]}).json()
        assert body["latency_ms"] > 0

    def test_batching_returns_the_same_entities_as_one_at_a_time(self, client):
        """Batching is a throughput optimisation and must not change the answer.

        Padding a short sequence alongside a long one is exactly where an
        attention-mask bug shows up, and it would show up as slightly worse
        output rather than as an error.
        """
        texts = [
            "El Banco Central de Chile anuncio hoy en Santiago.",
            "Corto.",
            "Gabriel Boric se reunio con la Union Europea en Bruselas ayer por la tarde.",
        ]
        batched = client.post("/extract", json={"texts": texts}).json()["documents"]
        individually = [
            client.post("/extract", json={"texts": [t]}).json()["documents"][0]
            for t in texts
        ]

        for a, b in zip(batched, individually, strict=True):
            assert [(e["text"], e["type"]) for e in a["entities"]] == [
                (e["text"], e["type"]) for e in b["entities"]
            ]


class TestAwkwardInput:
    def test_empty_batch_is_rejected(self, client):
        assert client.post("/extract", json={"texts": []}).status_code == 422

    def test_oversized_batch_is_rejected(self, client):
        response = client.post("/extract", json={"texts": ["hola"] * 500})
        assert response.status_code == 422

    def test_whitespace_only_text_does_not_crash(self, client):
        response = client.post("/extract", json={"texts": ["   "]})
        assert response.status_code == 200
        assert response.json()["documents"][0]["entities"] == []

    def test_emoji_and_punctuation_do_not_crash(self, client):
        response = client.post("/extract", json={"texts": ["!!! ¿¿ 🙂 ###"]})
        assert response.status_code == 200

    def test_very_long_document_is_truncated_not_rejected(self, client):
        """Words past max_length are unpredictable, but the call must still work."""
        response = client.post("/extract", json={"texts": ["Madrid " * 3000]})
        assert response.status_code == 200

    def test_missing_field_is_rejected(self, client):
        assert client.post("/extract", json={}).status_code == 422


class TestMetrics:
    def test_metrics_expose_counters_and_quantiles(self, client):
        client.post("/extract", json={"texts": ["Santiago."]})
        text = client.get("/metrics").text
        assert "ner_requests_total" in text
        assert 'ner_request_latency_ms{quantile="0.95"}' in text

    def test_document_counter_advances_by_the_batch_size(self, client):
        def documents_total() -> int:
            for line in client.get("/metrics").text.splitlines():
                if line.startswith("ner_documents_total "):
                    return int(line.split()[1])
            raise AssertionError("counter missing")

        before = documents_total()
        client.post("/extract", json={"texts": ["uno", "dos", "tres"]})
        assert documents_total() == before + 3
