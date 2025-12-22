# Setup Guide

## Quick Start

### 1. Python Environment

You need Python 3.9 or higher. Using pyenv:

```bash
# Set Python version (if using pyenv)
pyenv local 3.12.2  # or any 3.9+
python --version  # Verify
```

### 2. Install Poetry (if not installed)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 3. Install Dependencies

```bash
poetry install
```

This installs all dependencies from `pyproject.toml` into a Poetry-managed virtual environment.

### 4. Set Up API Keys

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=sk-...
# GEMINI_API_KEY=...
# CLAUDE_API_KEY=sk-ant-...
# XAI_API_KEY=xai-...
```

**Note:** You only need API keys for the models you want to use. You don't need all of them.

### 5. Verify Setup

Run unit tests (no API keys required):

```bash
poetry run pytest tests/
```

Run smoke test (skips gracefully if no API keys):

```bash
poetry run python scripts/smoke_test_minimal.py
```

**Note:** Always use `poetry run` before Python commands to ensure you're using the Poetry-managed environment.

If tests pass, you're ready for Week 2!

## What Gets Installed

From `pyproject.toml`:

- **Core dependencies:**

  - `pandas` - Data handling
  - `python-dotenv` - Environment variable loading
  - `openai` - OpenAI API client
  - `anthropic` - Claude API client
  - `google-genai` - Gemini API client

- **Optional dev dependencies** (for development):
  - `pytest` - Testing
  - `jupyter` - Notebooks
  - `black`, `ruff` - Code formatting/linting

## Troubleshooting

### "No models available"

- Check that your `.env` file exists and has API keys
- Verify API keys are correct (no extra spaces)
- Make sure you're in the project root directory

### "Module not found"

- Make sure you ran `poetry install`
- Check that you're using `poetry run` before Python commands
- Verify you're in the project root directory

### pyenv issues

If you see "shell integration not enabled":

```bash
# Add to your ~/.zshrc:
eval "$(pyenv init -)"

# Then reload:
source ~/.zshrc
```

## Next Steps

Once setup is complete:

1. Run `poetry run pytest tests/` to verify unit tests
2. Run `poetry run python scripts/smoke_test_minimal.py` to verify extraction pipeline
3. Start Week 2: `poetry run python scripts/run_phase1.py --data data/study1_venues_20250117.csv --sample-size 10`
