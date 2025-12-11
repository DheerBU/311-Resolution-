# Makefile for Boston 311 project

PYTHON := python
VENV   := .venv
PIP    := $(VENV)/bin/pip

.PHONY: all setup venv install run test clean

all: run

venv:
	@test -d $(VENV) || ($(PYTHON) -m venv $(VENV))

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: install
	$(VENV)/bin/python midterm_pipeline.py

test: install
	$(VENV)/bin/python -m pytest

clean:
	rm -rf $(VENV)
	find . -name "__pycache__" -type d -exec rm -rf {} +
