.PHONY: setup ingest features train evaluate serve

VENV_PY=.\.venv\Scripts\python
VENV_PIP=.\.venv\Scripts\pip

setup:
	python -m venv .venv && $(VENV_PY) && pip install -U pip && \
	$(VENV_PIP) install -e .

ingest:
	$(VENV_PY) -m src.ingest

features:
	$(VENV_PY) -m src.features

train:
	$(VENV_PY) -m src.train

evaluate:
	$(VENV_PY) -m src.evaluate

serve:
	$(VENV_PY) -m uvicorn src.serve:app --reload

frontend:
	$(VENV_PY) -m src.frontend.app