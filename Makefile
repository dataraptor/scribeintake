# POSIX task runner. Windows/PowerShell equivalents live in tasks.ps1.
# Canonical commands use `python -m ...` so they work without `make`.

.PHONY: install lint fmt test test-live

install:
	pip install -e "./core[dev]"

lint:
	ruff check

fmt:
	ruff format

test:
	python -m pytest core/tests -m "not live"

test-live:
	python -m pytest core/tests -m live
