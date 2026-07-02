# Install everything (including ML deps — first sync downloads several GB of wheels)
setup:
    uv sync

# Dev/test environment without torch/vibevoice (extract/combine/tests work fine)
setup-lite:
    uv sync --no-group ml

test *ARGS:
    uv run --no-sync pytest {{ARGS}}

lint:
    uv run --no-sync ruff check src tests

fmt:
    uv run --no-sync ruff format src tests
