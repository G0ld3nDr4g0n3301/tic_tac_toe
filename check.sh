#!/bin/bash
uv run mypy . --explicit-package-bases && uv run ruff check . --fix && uv run ruff format .
