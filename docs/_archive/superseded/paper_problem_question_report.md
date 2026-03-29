# Paper Problem–Question Report (Gentags)

**Date:** 2026-03-01  
**Purpose:** Lock a paper-ready problem statement + research question that matches what the
code and completed runs actually demonstrate (and does *not* over-claim).

---

## 1) Does the problem statement reflect what you built?

**Problem statement (current framing):**  
Modern language-based systems lack an explicit semantic state abstraction, making downstream
decisions difficult to audit, compare, update, or control under constraints.

### Does Gentags solve *all* of that?

**No.** Gentags is not (yet) a full end-to-end “stateful language system” with belief updates,
temporal memory, and control policies.

### Does Gentags address a core part of that problem?

**Yes.** Gentags concretely addresses three core gaps that sit *upstream* of those larger
systems problems:

1) **Lack of explicit semantic state**  
Gentags externalize meaning into an explicit state object: a stored set of short, human-
inspectable semantic propositions extracted from evidence.

2) **Lack of addressable semantic units**  
Gentags are discrete units you can directly reference, delete, add, replace, and compare
across runs (instead of dealing only with entangled text or opaque vectors).

3) **Inability to empirically evaluate how representation affects constraint-based decisions**  
Gentags enables *decision-causal* evaluation by allowing controlled edits to the
representation (interventions) while holding the judge, persona, and scoring rubric fixed.

### What you did **NOT** build (yet)

These are explicitly out of scope for the current work:

- Full belief updates / probabilistic uncertainty over state
- Temporal memory system (longitudinal updating, decay, contradiction handling)
- Control/policy layer (action selection, planning, closed-loop control)
- Risk-sensitive decision framework (calibration, safety constraints, robust control)

**Conclusion:** Gentags is a **candidate semantic state abstraction**, not the full system.
That’s a valid and publishable scope as a representation + characterization contribution.

---

## 2) So what exactly does Gentags do relative to that problem?

### Proposed representation move

**Given the problem:** meaning is implicit and entangled in raw text, and opaque in embeddings.  
**Gentags proposes:** externalize meaning into **discrete propositional state** (short semantic
constraints) that can be persisted and edited.

This is a representation claim, not a downstream application claim.

### What you empirically test (completed)

You test whether this structural choice improves *decision reliability under hard constraints*
relative to lexical fragment baselines (keywords), and whether it has the stability properties
required of “state”.

Concretely:

- **Stability/consistency of the state representation** across runs/prompts/models (Phase 2).
- **Decision fidelity and constraint compliance** when decisions are made using only the
  representation (Phase 5).
- **Decision-causal sensitivity** via targeted representation edits (Phase 4 DIR), as a
  mechanism-level check (with caveats; see below).

---

## 3) So what is the research question?

The research question should be empirical and narrow (not philosophy).

### Clean primary research question

**Does externalizing semantic meaning as discrete propositional state improve downstream
decision reliability under hard constraints compared to non-normalized lexical
representations?**

### Even tighter (optional phrasing)

**Can discrete propositional semantic state serve as a reliable abstraction layer for
constraint-based decision-making in language-based systems?**

---

## 4) How Gentags answers it (with numbers from completed runs)

This section ties each claim directly to the artifacts you already produced.

### 4.1 Gentags is a plausible “state” abstraction (stability evidence)

Phase 2 establishes that gentags behave like recoverable state: surface forms vary but meaning
is stable, and variability tracks evidence (identifiability).

From `/Users/infa/Documents/voraa/researchGentags/docs/PHASE2_STABILITY.md`:

- **Semantic stability (cosine):** 0.977  
- **Surface variation (Jaccard):** 0.471  
- **Semantic gap (cosine − Jaccard):** 0.504  
- **Retention above random:** +0.164  
- **Evidence → variability correlation:** -0.230

Interpretation (paper-safe): repeated observers produce different words but recover highly
similar semantics; when evidence is sparse, the “state” is less identifiable and dispersion
increases.

### 4.2 Gentags preserves constraint-relevant information better than lexical fragments (decision evidence)

Phase 5 is the strongest direct answer to the research question because it tests decisions
under **hard persona constraints** using only the representation as input, and compares
against keyword baselines.

From `/Users/infa/Documents/voraa/researchGentags/docs/phase5/BASELINE_LEGIBILITY_REPORT.md`:

- **Design:** 50 venues × 4 personas × 6 systems × N=5 = 6,000 judge calls per judge
- **Primary judge:** `gpt-4o-2024-08-06`; **Cross-judge:** `claude-sonnet-4-20250514`
- **Primary metric 1 (decision fidelity):** **FER agreement** (match full-evidence reference)
  - gentag: **79.5%**
  - RAKE: 61.6%, YAKE: 58.5%, TF-IDF: 52.3% (all significantly worse; best p < 0.0001)
- **Primary metric 2 (hard constraint reliability):** **Compliance (P2+P3)**
  - gentag: **97.3%**
  - RAKE: 89.0%, YAKE: 85.3%, TF-IDF: 86.0% (p = 0.0002)
- **Ablation (semantics vs volume):** truncated gentags matched to RAKE tag count still preserve
  large FER-agreement and compliance advantages (so the gain is semantic clarity, not tag volume).

Interpretation (paper-safe): when the decision depends on satisfying hard constraints, gentags
transmit the relevant constraint signal more reliably than lexical fragments produced by
keyword extraction.

### 4.3 Gentags enables decision-causal evaluation (mechanism evidence; not the main headline)

Phase 4 DIR tests show you can perform controlled interventions on the state and observe
directional decision changes (a “representation causal test”).

From `/Users/infa/Documents/voraa/researchGentags/docs/phase4/DIR_SCALED_RUN_REPORT.md`:

- **All-units DIR pass:** gentag 13/16 (81.2%) vs RAKE 10/16 (62.5%), separation +18.8pp
- **Caveats:** Fisher’s exact p = 0.433 (underpowered); placebo movement elevated (gentag 3/16).

Interpretation (paper-safe): the protocol works and demonstrates *addressability + causal
testing* of representations, but the scaled DIR results are best positioned as supporting
evidence / methodology, not the sole basis for the main claim.

---

## 5) What you should NOT claim (current evidence does not support this)

Do **not** claim that Gentags solves:

- “Controllability” as an end-to-end system property
- Belief updates, uncertainty, or calibration
- Long-term memory, time, or contradiction resolution
- Agent drift, policy learning, or risk-sensitive control

Those are valid future-work directions, but they are not established by Phase 2/5 (and Phase 4
is not yet paper-ready as a decisive separation result).

---

## 6) Final paper structure (problem → question → contribution)

### Problem

Language-based systems lack explicit semantic state; meaning stays implicit in text or
entangled in vectors, limiting auditability and constraint-based reliability.

### Research Question

Does discrete propositional semantic state improve decision reliability under hard constraints
compared to non-normalized lexical representations?

### Contribution

1) **Representation:** Gentags as externalized, addressable propositional semantic state.  
2) **Validation:** Empirical evidence that gentags (a) behave like state (stability +
   identifiability) and (b) preserve constraint-relevant information better than keyword
   baselines in constrained decision tasks.

### Suggested one-sentence “Core Contribution” (paper-safe)

Gentags provide an externalized, addressable propositional state representation that is
semantically stable and improves constraint-based decision reliability compared to lexical
keyword representations.
