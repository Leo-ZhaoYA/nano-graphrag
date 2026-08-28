# Repository Guidelines

## Project Structure & Module Organization

The main Python package is `nano_graphrag/`. `graphrag.py` and `base.py` expose the core GraphRAG workflow; storage adapters live in `nano_graphrag/_storage/`, entity extraction in `nano_graphrag/entity_extraction/`, and prompts/utilities in the package root. `docs/` contains design and contribution notes, `documents/` contains sample inputs and outputs, `config/` holds YAML configuration, and `.github/workflows/` contains CI. Top-level scripts such as `server.py`, `client.py`, and `evaluation.py` support serving and evaluation. Add automated tests under `tests/` when extending core behavior.

## Build, Test, and Development Commands

Use an editable install while developing:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Run the CI-equivalent checks with:

```powershell
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
$env:NANO_GRAPHRAG_TEST_IGNORE_NEO4J='true'; python -m pytest -o log_cli=true -o log_cli_level=INFO -v ./
```

Use `python server.py` or the documented examples in `readme.md` for local integration checks. Keep API keys and connection settings in environment variables or untracked local configuration.

## Coding Style & Naming Conventions

Use Python 4-space indentation, clear type-aware interfaces, `snake_case` for modules/functions/variables, `PascalCase` for classes, and descriptive `UPPER_SNAKE_CASE` constants. Preserve the existing asynchronous patterns and storage abstractions. Keep imports and implementation changes focused; run the CI flake8 command before submitting.

## Testing Guidelines

Use `pytest` and `pytest-asyncio` for tests; name files `test_*.py` and test functions `test_*`. Core-code changes should include regression tests, documentation/docstrings, or runnable examples. Tests requiring Neo4j must honor `NANO_GRAPHRAG_TEST_IGNORE_NEO4J` so the default suite remains portable. No repository-wide coverage threshold is currently configured.

## Commit & Pull Request Guidelines

This checkout has no readable Git history, so follow the existing contribution policy: use focused commits with imperative subjects (for example, `Fix chunk retrieval ordering`). Pull requests should explain the behavior change, include tests and relevant documentation/examples, identify configuration or dependency changes, and mention any external services required. Keep dependencies minimal and link related issues when applicable.
