# Gentags: Discrete Semantic State for Language-Based Systems

**Problem**
Modern systems increasingly rely on LLMs to interpret unstructured text. While LLMs are
effective at producing contextual judgments, they do not provide persistent, inspectable
semantic state. Each query recomputes meaning implicitly, making it difficult to:
- track what a system knows about an entity over time
- identify which aspects of that knowledge change when new evidence arrives
- attribute downstream behavior to specific semantic causes

Raw text is verbose and unstructured. Dense embeddings compress meaning but entangle it in
opaque vectors. Keyword extraction preserves surface terms but lacks semantic structure.
None of these representations function as a state object that can be inspected, compared,
and updated.

The core problem is not retrieval or summarization—it is how to externalize semantic
judgments into a stable, addressable form that systems can reason over.

**Proposal**
We introduce gentags: discrete, model-generated semantic propositions extracted from evidence.
A gentag is:
- Discrete — a bounded linguistic unit (1–4 words)
- Semantic — expresses a meaningful property, not just a token
- Propositional — asserts something about the world (e.g., slow service, crowded atmosphere)
- Externalized — stored outside the model as persistent state
- Evidence-conditioned — grounded in the provided text

Gentags do not require a predefined ontology or schema. They are not labels chosen from a
fixed set, nor probabilistic beliefs. They are explicit semantic statements produced by the
model and treated as state.

**What This Paper Demonstrates**
1. Gentags form a stable semantic representation
Across repeated extractions, different prompts, and different LLMs, gentags show high semantic
consistency despite surface variation. While exact wording may differ, the underlying meaning
remains aligned. This demonstrates that gentags capture semantic content rather than lexical
artifacts.

Why this matters: Without stability, a representation cannot serve as state.

2. Gentags preserve source meaning without mirroring it
Gentags retain a significant portion of the semantic content of the source text, while not
simply copying phrases. Classical keyword methods achieve higher lexical overlap, but gentags
synthesize information across evidence rather than reflecting frequency.

Why this matters: State should summarize meaning, not echo text.

3. Gentags enable localized, attributable semantic state
When gentags are analyzed as a set, their semantic mass concentrates into a small number of
interpretable aspects. Compared to embeddings and keyword baselines, gentags produce factorized
representations where semantic information is not diffusely spread.

As a result:
- differences between two states can be localized
- changes can be attributed to specific semantic propositions
- downstream effects can be traced back to explicit causes

Why this matters: Systems cannot reason about or control what they cannot attribute.

**What Gentags Are Not**
- Not a belief state (no probabilities, no update rules)
- Not an ontology (no predefined categories)
- Not a retrieval index
- Not a recommender system
- Not an agent or control policy

This paper introduces a state representation, not a full decision system.

**Why This Matters Now**
Earlier representation primitives—TF-IDF, topic models, word embeddings—emerged in response to
the computational and modeling limits of their time. They made text usable for downstream
systems, but none provided inspectable semantic state.

LLMs now encode rich semantic judgments internally, but those judgments remain implicit and
transient. Gentags leverage current model capabilities to externalize that latent structure
into a form systems can store, compare, and reason over.

This work defines a missing layer between raw language and downstream decision-making: a
persistent semantic state composed of discrete propositions.

**Expected Impact and Future Work**
Gentags provide a foundation for:
- systems that track evolving semantic knowledge
- attribution-aware reasoning
- controlled updates of semantic state
- future belief, uncertainty, and control frameworks

Future research may extend gentags with confidence measures, contradiction handling, temporal
updates, or domain-specific reasoning. Those are deliberately out of scope here.

**Core Contribution (One Sentence)**
Gentags introduce a practical, evidence-conditioned semantic state representation that makes
meaning persistent, attributable, and inspectable—something existing text representations
cannot provide.
