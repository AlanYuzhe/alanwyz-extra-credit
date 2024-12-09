PYTHON = python3.13
PIP = $(PYTHON) -m pip
VENV_DIR = .venv

all: setup run

setup:
	@echo "Setting up virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/$(PIP) install --upgrade pip
	$(VENV_DIR)/bin/$(PIP) install -r requirements.txt

run:
	@echo "Running the program..."
	$(VENV_DIR)/bin/$(PYTHON) test.py
