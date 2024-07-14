# Makefile

# Define the default target that runs all commands in sequence
.PHONY: all
all: frozen gitcount changelog

# Define individual commands
.PHONY: frozen
frozen:
	@echo "Running frozen.py..."
	python frozen.py

.PHONY: gitcount
gitcount: 
	@echo "Running gitcount.py..."
	python gitcount.py

.PHONY: changelog
changelog: 
	@echo "Running change_log.py..."
	python change_log.py
