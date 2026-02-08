# Model Notes: "As-Is Defaults" Statement

## Study 1 Commitment

**All models are used with provider defaults.** No temperature, top-p, or decoding parameters are explicitly set unless required by the API.

## Model Configuration

### Parameter Policy

- **Temperature:** `None` (provider default)
- **Max Tokens:** `None` (provider default, except Claude which requires a value)
- **Top-p / Top-k:** Not set (provider default)
- **Other parameters:** Not set (provider default)

### Model-Specific Notes

#### OpenAI (gpt-5-nano)

- Uses system prompt
- No explicit parameters set
- Provider defaults apply

#### Gemini (gemini-2.5-flash)

- No system prompt (Gemini doesn't support it)
- No explicit parameters set
- Provider defaults apply

#### Claude (claude-sonnet-4-5)

- No system prompt (uses user prompt only)
- `max_tokens` set to 8192 (required by API, not a tuning choice)
- `temperature` not set (provider default)

#### Grok (grok-4)

- Uses system prompt
- Uses OpenAI-compatible API
- No explicit parameters set
- Provider defaults apply

## Rationale

This "as-is" approach ensures:

1. **Reproducibility:** Results reflect model behavior without tuning
2. **Fairness:** All models evaluated under their default settings
3. **Simplicity:** No parameter search or optimization

## Version Tracking

Model configurations are frozen in `src/gentags/config.py` with version tracking. Any changes require a new study version.

