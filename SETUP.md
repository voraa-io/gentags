# Setup Guide

## Quick Start

### 1. Python Environment

You need Python 3.9 or higher. Using pyenv:

```bash
# Set Python version (if using pyenv)
pyenv local 3.12.2  # or any 3.9+
python --version  # Verify
```

### 2. Install Dependencies

You have two options:

**Option A: Using pip (standard)**

```bash
pip install -e .
```

**Option B: Using uv (faster, modern)**

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -e .
```

Both methods read dependencies from `pyproject.toml`.

### 3. Set Up API Keys

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

### 4. Verify Setup

Run the smoke test:

```bash
poetry run python scripts/sanity_check.py
```

**Note:** Always use `poetry run` before Python commands to ensure you're using the Poetry-managed environment.

If it passes, you're ready for Week 2!

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

- Make sure you ran `pip install -e .` (the `-e` flag installs in editable mode)
- Check that you're using the correct Python environment

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

1. Run `python scripts/sanity_check.py` to verify
2. Start Week 2: `python scripts/run_phase1.py --data data/external/venues_data.csv`
