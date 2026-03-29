# Phase 1: Multi-Model Extraction

**Claim:** LLMs can extract semantic tags from sparse review text at scale.

**Evidence:**
- 553 venues × 4 models × 3 prompts × 2 runs = 13,272 extractions
- ~265,842 tags extracted across OpenAI, Gemini, Claude, Grok
- 100% completion rate, $12.61 total cost

**Implication:** Gentag extraction is cheap, scalable, and model-agnostic.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Venues | 553 |
| Models | OpenAI `gpt-5-nano`, Gemini `gemini-2.5-flash`, Claude `claude-sonnet-4-5`, Grok `grok-4` |
| Prompts | minimal, anti_hallucination, short_phrase |
| Runs per config | 2 |
| Total extractions | 13,272 |

## Output

Current local canonical copy of Phase 1 outputs: `results/phase1_downloaded/`

Files:
- `*_tags.csv` — one row per tag
- `*_extractions.csv` — one row per extraction
- `meta/*_manifest.json` — reproducibility metadata

Note: the full Phase 1 extraction appears to have been run on a Google Cloud VM and then copied back locally. Downstream Phase 2/3/5 scripts use `results/phase1_downloaded/`, not `results/phase1/`.
