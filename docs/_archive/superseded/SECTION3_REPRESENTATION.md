# Section 3: Gentags as Representation Infrastructure

Gentags occupy an intermediate point between symbolic keywords and dense embeddings: **semantic
enough to generalize, lexical enough to attribute.**

**Terminology note:** In this paper, **semantic state** means a set of discrete, evidence-conditioned
semantic propositions (gentags). It is **not** a probabilistic belief state: we do not model
uncertainty, belief updates, or control dynamics.

---

## 3.1 What the Representation Is

Gentags are **short, atomized semantic propositions** extracted from evidence. They are:
- **Inspectable** (readable strings)
- **Persistent** (stored state, not regenerated each query)
- **Composable** (combine tags to express complex constraints)

---

## 3.2 What We Have Already Proven (Phase 2)

- **Semantic stability** despite lexical variation
- **Cross-model agreement** on extracted meaning
- **Evidence-sensitive dispersion** (less evidence → more variability)
- **Retention above random** (tags preserve review meaning)

These establish that gentags are a **consistent representation**.

---

## 3.3 What Phase 3 Must Prove (Structure + Utility)

Phase 3 is the **complete argument**. It must show both:

1. **Structural Proof (State-Gini):** gentags are **factorized** (semantic mass concentrates into
   a few interpretable facets). Baselines (RAKE/TF-IDF/YAKE) should remain diffuse.
2. **Utility Proof (CheckList DIR/INV):** gentags enable **attribution-aware interventions**
   where targeted edits produce predictable downstream changes.

Until Phase 3 is executed, these claims are **planned** and not yet empirical results.
