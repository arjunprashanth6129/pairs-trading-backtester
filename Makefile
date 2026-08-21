VENV := .venv
PY   := $(VENV)/bin/python
export PYTHONPATH := src

.PHONY: all setup run test clean

all: run

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

run:
	$(PY) -m pairs_backtester.run

test:
	$(VENV)/bin/pytest -q

clean:
	rm -rf results/* docs/figures/* .pytest_cache
