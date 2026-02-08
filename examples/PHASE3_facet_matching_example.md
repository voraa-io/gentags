# Phase 3 facet matching example (real gentags)

This example uses real tags from: `results/phase1_downloaded/week2_run_20251223_191104_tags_openai.csv`
Venue ID: `0C3FBm4g9DPjogLP0Ifl`

## Facet keyword lists (as in `scripts/phase3_analysis.py`)

- food_quality: food, fresh, tasty, delicious, bland, meal, breakfast, lunch, dinner, dish, cook, chef, menu, eat

- coffee_drinks: coffee, espresso, latte, tea, drink, beverage, cappuccino, mocha, brew, roast

- service: staff, service, friendly, rude, slow, fast, waiter, barista, server, attentive, helpful

- ambiance: atmosphere, vibe, cozy, noisy, quiet, decor, music, lighting, ambiance, aesthetic, interior

- price_value: price, expensive, cheap, affordable, value, worth, overpriced, budget, cost, dollar

- crowding: crowded, busy, wait, line, packed, empty, reservation, queue

- seating: seating, outdoor, patio, table, chair, space, indoor, terrace, booth

- dietary: vegan, vegetarian, gluten, allergy, organic, healthy, dairy, keto, paleo

- portions: portion, size, generous, small, large, filling, huge, tiny

- location: location, parking, accessible, downtown, corner, find, neighborhood, walk, drive

## Sample tags and their matched facet

| tag | matched facet |
| --- | --- |
| average receptionist service | service |
| friendly waiter | service |
| attentive waiter | service |
| good court | other |
| good area | other |
| good menu | food_quality |
| nice place | other |
| deliciou food | food_quality |
| price inquiry | price_value |
| hour inquiry | other |
| voicemail | other |
| information request | other |
| high noise level | other |
| late-night noise | other |
| disrespectful staff | service |
| lack of cooperation | other |
| excellent place | other |
| paddle tenni | other |
