# Reproduce every number in the README, in order.
#   make all   runs the full pipeline (~15 min on a consumer GPU)

PYTHON ?= python

.PHONY: help install data test baselines transformers evaluate figures app all clean

help:
	@echo "install      install the package and dev dependencies"
	@echo "data         download and checksum-verify the CoNLL-2002 corpus"
	@echo "test         run the test suite (44 tests)"
	@echo "baselines    train the gazetteer and CRF, evaluate on dev"
	@echo "transformers fine-tune BETO, mBERT and XLM-R"
	@echo "evaluate     single evaluation on the held-out test split"
	@echo "figures      regenerate figures and the error analysis"
	@echo "app          serve the Gradio demo locally"
	@echo "all          data -> test -> baselines -> transformers -> evaluate -> figures"

install:
	$(PYTHON) -m pip install -e ".[dev,app]"

data:
	$(PYTHON) scripts/download_data.py

test:
	$(PYTHON) -m pytest

baselines:
	$(PYTHON) scripts/train_baselines.py

transformers:
	$(PYTHON) scripts/train_transformer.py
	$(PYTHON) scripts/train_transformer.py --model bert-base-multilingual-cased --tag mbert
	$(PYTHON) scripts/train_transformer.py --model xlm-roberta-base --tag xlmr

evaluate:
	$(PYTHON) scripts/evaluate_test.py \
		--models models/bert-base-spanish-wwm-cased/best models/mbert/best models/xlmr/best

figures:
	$(PYTHON) scripts/error_analysis.py

app:
	$(PYTHON) app/app.py

all: data test baselines transformers evaluate figures

clean:
	rm -rf models outputs .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
