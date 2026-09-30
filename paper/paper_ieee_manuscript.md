# SemPointer: An Indirection Layer over Dense Retrieval — Measurements and Negative Results

**Author:** Viksit Garg  
**Affiliation:** Department of Artificial Intelligence & Machine Learning, Dr. Akhilesh Das Institute of Professional Studies, New Delhi, India  
**Contact:** `viksitg427@gmail.com`  
**Paper Track:** Theoretical / Conceptual Foundations (IEEE Format)  
**Project Governance Status:** Phase 0/1 Locked; Phase 4 Implementation Gate Active (`[ESTABLISHED]` / `[REASONED]` / `[SPECULATION]` Tagged)

---

## Abstract

Autoregressive Large Language Models re-attend to their full prefix history on every generation step, producing quadratic attention compute across multi-turn interactions. Context compression methods, including token pruning, soft prompt compression, and key-value cache eviction, reduce sequence footprints. Yet they couple information representation with information addressing. Once a prompt is compressed, the model must process the entire proxy uniformly across subsequent turns, without a mechanism to dereference specific items on demand.
We introduce SemPointer, a theoretical framework for random-access semantic memory in transformer-based models. SemPointer separates context retention from reference by defining compact token-level semantic pointers that serve as explicit, reusable addresses to an independent memory substrate. We formalize the lifecycle of pointer generation, substrate retention, query-conditioned selection, and selective resolution. Under explicit cost-model assumptions, we derive sufficient conditions, under the stated cost model, (registry size, access sparsity, substrate) under which pointer-mediated access reduces quadratic self-attention FLOPs, including the crossover threshold $K^*$ for the amortization regime. We define seven axiomatic properties distinguishing semantic pointers from lossy representations, analyze theoretical failure modes including address collision and semantic drift, and formulate six falsifiable hypotheses paired with an operational evaluation agenda. SemPointer is presented strictly as a theoretical framework; no implementation code or empirical evaluation is claimed.

**Index Terms:** Large Language Models, Context Reuse, Semantic Pointers, Random-Access Semantic Memory, Indirection, Theoretical Computer Science.

---

## I. Introduction

Transformer models [ESTABLISHED: Vaswani et al., 2017] process textual context as an unbroken linear sequence. In multi-turn dialogue, long-horizon programming, and iterative document synthesis, historical context accumulates across successive inference calls. Standard autoregressive generation re-attends to the complete prefix history on every decoding step. This attention computation scales quadratically with sequence length, creating a substantial computational and memory burden as conversations extend.

Recent literature addresses this bottleneck through context compression across three main directions: hard token pruning [ESTABLISHED: Jiang et al., 2023, 2024; Pan et al., 2024], learned continuous soft prompts [ESTABLISHED: Mu et al., 2023; Chevalier et al., 2023; Ge et al., 2024; Li et al., 2025], and dynamic key-value (KV) cache eviction [ESTABLISHED: Kim et al., 2024; Liu et al., 2025; Kwon et al., 2023]. While these methods reduce sequence footprints, they share an architectural limitation: they reduce context representations in place, without establishing an addressable indirection layer. Once a prompt is compressed into summary vectors or pruned text, the model processes that proxy uniformly across all downstream turns, regardless of whether specific details within that block relate to the immediate query.

### A. The Indirection Hypothesis

Operating systems and runtime environments separate memory addresses from stored payloads. Processors pass compact references through registers and defer payload retrieval until an instruction accesses the underlying data. We hypothesize that an analogous indirection principle can be realized within the sequence space of Large Language Models:

> **[REASONED] The Indirection Hypothesis:** By generating compact, token-level semantic pointers that reference independently stored contextual substrates, an LLM architecture can support selective, address-based access to historical context, bypassing the need to reprocess unneeded context blocks.

### B. Research Questions

This theoretical study investigates three primary research questions:
*   **RQ1 (Formal Characterization):** What mathematical and structural properties are necessary and sufficient to define a valid semantic pointer in transformer sequence spaces, as distinct from lossy summary vectors or pruned context tokens?
*   **RQ2 (Theoretical Viability):** Under what formal registry scaling, access sparsity, and memory substrate conditions does semantic indirection reduce cumulative self-attention FLOPs compared to uniform prefix re-ingestion?
*   **RQ3 (Failure Bounds):** What are the theoretical limits, collision probabilities, and drift vulnerabilities of persistent semantic pointers across extended horizons?

### C. Contributions

In accordance with project research governance, this paper presents theoretical and conceptual contributions intended to guide subsequent empirical validation:
1.  We distinguish context compression from semantic addressing, establishing seven axiomatic criteria required for true pointer mechanics.
2.  We introduce SemPointer, an architectural abstraction defining a four-phase memory lifecycle (Generation, Substrate Storage, Query-Conditioned Selection, and Dynamic Resolution).
3.  We classify memory substrates into a three-tier taxonomy (Raw Text $\mathcal{M}_{\text{text}}$, KV Cache $\mathcal{M}_{\text{kv}}$, Latent State $\mathcal{M}_{\text{latent}}$) with explicit computational cost formulations.
4.  Under explicit modeling assumptions (A1--A5), Section VI analyzes computational tradeoffs and derives the algebraic crossover threshold $K^*$ where pointer-mediated reuse outperforms linear re-ingestion.
5.  We characterize theoretical vulnerabilities, deriving generalized birthday bounds for address collision and analyzing attention-drift dynamics under relative positional embeddings.
6.  We formulate six testable hypotheses ($H_1$--$H_6$) with formal null hypotheses and specify an operational benchmark suite for empirical testing.

---

## II. Background and Prior-Art Spectrum

### A. Dense Autoregressive Attention and Scaling Walls
In standard autoregressive generation [ESTABLISHED: Vaswani et al., 2017], given an input sequence $X = (x_1, x_2, \dots, x_L)$, each self-attention layer computes:
$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
where $Q, K, V \in \mathbb{R}^{L \times d}$. The computation of $QK^T$ incurs time complexity $\mathcal{O}(L^2 d)$ and requires storing KV state tensors of size $\mathcal{O}(2 n_{\text{layers}} n_{\text{heads}} L d_k)$. When interaction stretches across $T$ turns with accumulating history $L_t = \sum_{i=1}^t |X_i|$, cumulative attention compute scales as $\sum_{t=1}^T \mathcal{O}(L_t^2)$, creating a steep computational and memory bottleneck.

### B. Hard Prompt Compression
Hard prompt compression, developed in works such as LLMLingua [ESTABLISHED: Jiang et al., 2023], LongLLMLingua [ESTABLISHED: Jiang et al., 2024], and LLMLingua-2 [ESTABLISHED: Pan et al., 2024], frames context reduction as token pruning. Lexical perplexity or mutual information scores identify non-essential tokens, which are dropped. This preserves natural language formatting and requires no architectural modifications to the downstream model. The tradeoff is clear: pruning discards tokens irreversibly. Dependencies across distant tokens are severed, causing information loss on dense relational reasoning tasks [ESTABLISHED: Łajewska et al., 2025]. More importantly, hard compression provides no pointer or dereferencing mechanism. A deleted token cannot be recovered on demand.

### C. Continuous and Soft-Token Compression
Gist Tokens [ESTABLISHED: Mu et al., 2023] condense prompt prefixes into virtual tokens via modified attention masks --- but the gists must themselves remain resident in the prefix on every subsequent turn. AutoCompressors [ESTABLISHED: Chevalier et al., 2023] recursively compress chunks into summary vectors, achieving hierarchical reduction at the cost of a single fixed summary per chunk that cannot be selectively re-expanded. ICAE [ESTABLISHED: Ge et al., 2024] trains an encoder--decoder architecture to compress contexts into memory slots, and 500xCompressor [ESTABLISHED: Li et al., 2025] represents an entire context with one special token --- at which point access to any individual constituent fact is structurally unavailable. None of these introduce an indirection layer: if a historical context is irrelevant across intermediate turns, its compressed representation either remains in active memory or permanently discards the underlying text without an address to dereference it on demand.

### D. Dynamic Cache Eviction and Sparse Attention
Compressed Context Memory (CCM) [ESTABLISHED: Kim et al., 2024] and Contextual Semantic Anchors (SAC) [ESTABLISHED: Liu et al., 2025] manage inference-time state by evicting or aggregating KV states. GPU memory pressure decreases. But the eviction is typically irreversible: there is no token-level address by which an evicted state can be later retrieved.

### E. Cognitive Science Disambiguation: Semantic Pointer Architecture
The term *semantic pointer* has an existing lineage in cognitive science. Eliasmith [ESTABLISHED: 2013] and Blouw et al. [ESTABLISHED: 2016] formalized the Semantic Pointer Architecture (SPA) using high-dimensional continuous vectors ($D \approx 500$--$1000$) in Vector Symbolic Architectures that compress perceptual and motor states through binding (circular convolution) in spiking neural networks.

SemPointer addresses a different domain. SPA investigates biological cognition through spiking dynamics. SemPointer concerns discrete or latent token sequences in autoregressive transformers, with the goal of governing context-reuse complexity and selective key-value resolution. The terminological overlap is intentional, but the computational settings do not intersect.

---

## III. Formal Axiomatic Foundations of Semantic Pointers

Compressed representations and semantic pointers serve different functional roles. A compressed context carries information directly within active attention; a semantic pointer acts as an address to external storage, requiring an explicit dereferencing step to retrieve information. This section formalizes the criteria required for pointer-based indirection via Definition 1 and seven axioms.

### Definition 1 (Compressed Representation vs. Semantic Pointer)
*Let $X \in \mathcal{X}$ be an information unit of length $|X|$. A mapping $\mathcal{C}: \mathcal{X} \to \mathcal{Z}$ produces a compressed representation $Z$ if $|Z| < |X|$ and $Z$ directly acts as the information carrier in downstream computation. A mapping $\mathcal{G}: \mathcal{X} \to \mathcal{P}$ produces a semantic pointer $P$ if $P$ acts as an address into an independent memory substrate $\mathcal{M}$, such that the information content of $X$ is accessed only upon executing an explicit resolution operation $\mathcal{R}(P, M, Q)$.*

*   **Axiom 1 (Address Identity and Distinguishability):** Let $X_i, X_j \in \mathcal{X}$ be two semantically distinct context units ($X_i \not\equiv X_j$). Their generated pointers $P_i = \mathcal{G}(X_i)$ and $P_j = \mathcal{G}(X_j)$ must be distinguishable within the address space:
    $$D_{\mathcal{P}}(P_i, P_j) > \delta_{\text{min}} \quad \forall i \neq j$$
    where $D_{\mathcal{P}}$ is a metric on pointer space $\mathcal{P}$ and $\delta_{\text{min}} > 0$.
*   **Axiom 2 (Strict Asymmetric Compactness):** The pointer length $|P_i|$ measured in sequence units (tokens) must satisfy:
    $$|P_i| \ll |X_i|, \quad \text{with } \frac{|P_i|}{|X_i|} \le \kappa, \quad \kappa \in (0, 1)$$
    The pointer representation must not scale with the length of the underlying context unit. In the target operating regime ($k \le 8$, $|X_i| \ge 160$ tokens), $\kappa \le 0.05$; this is a design target, not an assumption. The threshold is definitional: since $k \le 8$, the ratio bound holds by construction for any unit of $\ge 160$ tokens; shorter units are already pointer-scale and gain nothing from indirection.
*   **Axiom 3 (Temporal and Spatial Stability):** A pointer $P_i$ generated at time $t$ must remain valid and addressable across subsequent time horizons $t + \Delta t$, independent of intervening context operations, memory updates, or unrelated token generations.
*   **Axiom 4 (Query-Conditioned Resolution Fidelity):** Given query $Q \in \mathcal{Q}$, resolution operator $\mathcal{R}$ acting on pointer $P_i$ and substrate $M_i = \mathcal{S}(X_i)$ must recover task-relevant mutual information:
    $$I\left(\mathcal{R}(P_i, M_i, Q); X_i \mid Q\right) \ge (1 - \epsilon) H(X_i \mid Q)$$
    for tolerance $\epsilon \in [0, 1)$, where $H(\cdot)$ denotes conditional entropy.
*   **Axiom 5 (Selective Addressability):** The activation of historical information must be sparse. Given a registry of $N$ pointers $\{P_1, \dots, P_N\}$, a query $Q_t$ resolves only the subset of indices $J^* \subseteq \{1, \dots, N\}$ where $|J^*| \ll N$, avoiding uniform activation of the remaining $N - |J^*|$ elements.
*   **Axiom 6 (Multi-Horizon Reusability):** A pointer $P_i$ must support repeated resolution operations across arbitrary non-consecutive queries $Q_{t_1}, Q_{t_2}, \dots, Q_{t_m}$ without requiring re-execution of pointer generation $\mathcal{G}(X_i)$.
*   **Axiom 7 (Algebraic Compositionality):** For compound queries requiring information from multiple context items $\{X_a, X_b\}$, the joint pointer representation must satisfy:
    $$\mathcal{R}(P_a \oplus P_b, \{M_a, M_b\}, Q) \equiv \mathcal{R}(P_a, M_a, Q) \cup \mathcal{R}(P_b, M_b, Q)$$
    where $\oplus$ represents a composition operator in pointer space.

| Paradigm | Mechanism | Stores Original? | Explicit Indirection? | Selective Access? | Reusability | Quadratic Complexity Mitigation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Linear Ingestion** [Vaswani et al., 2017] | Dense Attention | Yes (In Context) | No | No | Inefficient | None ($\mathcal{O}(L^2)$) |
| **Hard Token Pruning** [Jiang et al., 2023] | Token Elimination | No | No | No | No | Moderate (Length Pruned) |
| **Soft Prompt Compression** [Mu et al., 2023] | Learned Latent Tokens | No | No | No | Yes | Moderate (Prefix Condensed) |
| **Dynamic Cache Eviction** [Kim et al., 2024] | KV Discard/Merge | Partial | No | No | No | High (Memory Capped) |
| **Landmark Attention** [Mohtashami & Jaggi, 2023] | Gated Attention Routing | Yes (In Context) | No (Intra-pass) | Yes (Blocks) | No | High (Within single pass) |
| **Retrieval-Augmented Gen.** [Lewis et al., 2020] | External Vector DB | Yes (External) | Partial (Dense Vector) | Yes | Yes | High (Only top-$k$ retrieved) |
| **SemPointer (Proposed)** | **Semantic Indirection** | **Yes (Substrate)** | **Yes (Token Address)** | **Yes (Random)** | **Yes** | **High ($\mathcal{O}((Nk + mB_{\text{res}} + L_Q)^2)$ vs.\ $\mathcal{O}((NB + L_Q)^2)$)** |

---

## IV. The SemPointer Formal Architecture

### A. Mathematical Model
The model operates over five spaces. Context Space $\mathcal{X}$ contains textual sequences $X \in \mathcal{V}^*$. Substrate Space $\mathcal{M}$ holds independent storage representations (raw text, KV tensors, or latent embeddings). Pointer Space $\mathcal{P}$ consists of sequences of $k$ discrete tokens or soft embeddings, with $k$ fixed and small ($k \in [1, 8]$). Query Space $\mathcal{Q}$ contains active prompts, and Output Space $\mathcal{Y}$ contains target generations.

#### 1. Encounter and Substrate Ingestion
When context item $X_i$ is encountered at time $t_i$, the storage operator $\mathcal{S}$ commits $X_i$ to the memory substrate $\mathcal{M}$:
$$M_i = \mathcal{S}(X_i)$$
Concurrently, the pointer generator $\mathcal{G}_{\phi}: \mathcal{X} \to \mathcal{P}$ produces the compact semantic pointer:
$$P_i = \mathcal{G}_{\phi}(X_i) = (p_{i,1}, p_{i,2}, \dots, p_{i,k})$$
The pair $(P_i, M_i)$ is registered in the pointer index $\mathcal{I}$:
$$\mathcal{I}_{t_i} = \mathcal{I}_{t_{i}-1} \cup \{(P_i, M_i)\}$$

#### 2. Query-Conditioned Selection
Upon arrival of a new query $Q_t$, the selection function scores all registered pointers against $Q_t$:
$$s_i = \operatorname{Score}(P_i, Q_t) = \frac{\langle \mathbf{e}(P_i), \mathbf{e}(Q_t) \rangle}{\|\mathbf{e}(P_i)\| \|\mathbf{e}(Q_t)\|}$$
where $\mathbf{e}(\cdot)$ denotes an addressing projection into a shared metric space. The system selects top-$m$ pointers:
$$J^* = \operatorname{top-}m \left(\{s_i\}_{i=1}^N\right)$$

#### 3. Pointer Dereferencing and Resolution
For each selected pointer index $j \in J^*$, the resolution operator $\mathcal{R}_{\theta}$ dereferences $P_j$ against substrate $M_j$, conditioned on $Q_t$:
$$\hat{X}_j = \mathcal{R}_{\theta}(P_j, M_j, Q_t)$$
The resolved information $\hat{X}_{J^*}$ is injected into the primary generation context:
$$\hat{Y}_t \sim f_{\Theta}\left(\cdot \mid [P_1, \dots, P_N], \hat{X}_{J^*}, Q_t\right)$$
Because non-selected pointers remain in their compact state $P_i$ ($i \notin J^*$), the active context consumes only $N \cdot k + \sum_{j \in J^*} |\hat{X}_j| + |Q_t|$ tokens, rather than $\sum_{i=1}^N |X_i| + |Q_t|$.

### B. Architecture and Operational Lifecycle
Figure 1 illustrates the two-panel operational architecture of SemPointer across offline ingestion and online query-conditioned inference:
*   **Phase I (Offline Ingestion & Substrate Storage):** Incoming context block $X_i$ ($B$ tokens) is committed to decoupled storage substrate $\mathcal{M} \in \{\mathcal{M}_{\text{text}}, \mathcal{M}_{\text{kv}}, \mathcal{M}_{\text{latent}}\}$ via $\mathcal{S}$, while generator $\mathcal{G}_\phi$ produces $k$-token semantic pointer $P_i$. The address handle $(P_i, M_i)$ is registered in pointer registry $\mathcal{I}$ alongside cached addressing projection $\mathbf{e}(P_i) \in \mathbb{R}^d$.
*   **Phase II (Online Query Selection, Dereferencing & Inference):** Incoming query $Q_t$ ($L_Q$ tokens) is projected as $\mathbf{e}(Q_t)$ and scored against cached projections in $\mathcal{I}$, selecting sparse active subset $J^* \subseteq \{1, \dots, N\}$ ($m \ll N$). Resolution operator $\mathcal{R}_\theta$ dereferences only active blocks $M_j$ into payloads $\hat{X}_j$, which combine with compact unselected pointers $P_{i \notin J^*}$ and query $Q_t$ to form active prompt $C_t$ for autoregressive decode by LLM backbone $f_\Theta$.

### C. Conceptual Lifecycle (Specification, Not Implementation)

Algorithm 1 specifies the operational lifecycle as a control-flow specification; it is not an implementation, and no executable artifact is claimed.

```
Algorithm 1: Conceptual Lifecycle of SemPointer Addressing and Resolution (specification, not implementation)
Require: Context stream S_stream, Queries {Q_t}, Storage operator S, Pointer generator G_phi, Resolver R_theta, LLM backbone f_Theta
1:  Initialize Pointer Registry I <- empty
2:  for all incoming context blocks X_i in S_stream do
3:      M_i <- S(X_i)                      // Commit to substrate
4:      P_i <- G_phi(X_i)                  // Generate k-token pointer
5:      I <- I union {(P_i, M_i)}          // Register address
6:  end for
7:  for all queries Q_t do
8:      Compute relevance scores s_i = Score(P_i, Q_t) for all (P_i, M_i) in I
9:      J* <- top-m({s_i}_{i=1}^N)         // Identify required pointers
10:     Initialize Active Memory Payload Omega_t <- empty
11:     for all j in J* do
12:         X^_j <- R_theta(P_j, M_j, Q_t) // Dereference and resolve
13:         Omega_t <- Omega_t union {X^_j}
14:     end for
15:     Construct active prompt: C_t <- [{P_i}_{i not in J*}, Omega_t, Q_t]
16:     Emit generation: Y^_t <- f_Theta(C_t)
17: end for
```

---

## V. Memory Substrates and Physical Taxonomies

The SemPointer abstraction makes no commitment to how $M_i$ is physically stored. Three realizations are worth distinguishing, each with a different tradeoff profile:

### A. Raw Text Substrate ($\mathcal{M}_{\text{text}}$)
The simplest variant holds $M_i = X_i$ in flat external storage (RAM or disk). On pointer activation, $X_i$ is fetched and either concatenated directly into context or summarized on the fly with respect to $Q_t$. There is no encoding penalty and no fidelity loss from substrate storage, but resolution latency depends on I/O round-trips and the cost of re-encoding the fetched text.

### B. Key-Value Cache Substrate ($\mathcal{M}_{\text{kv}}$)
Here, $M_i = (\mathbf{K}_i, \mathbf{V}_i) \in \mathbb{R}^{2 \times L_{\text{layers}} \times |X_i| \times d}$, precomputed and held in host memory or paged GPU memory [ESTABLISHED: Kwon et al., 2023]. Resolution pages the KV blocks back into active GPU attention tables without recomputing self-attention projections, so that resolution itself requires few FLOPs. The cost is storage: KV tensors are large, and rotary position embedding (RoPE) displacement can degrade attention quality when blocks are moved far from their original position.

### C. Latent Embedding Substrate ($\mathcal{M}_{\text{latent}}$)
$M_i = \mathbf{Z}_i \in \mathbb{R}^{m \times d}$, where $m \ll |X_i|$ is a latent bottleneck state from an autoencoder [ESTABLISHED: Ge et al., 2024]. Resolution uses cross-attention unpooling conditioned on $P_i$ and $Q_t$. While storage is compact, verbatim text is lost, and the resolver may hallucinate when the bottleneck discards low-salience details. For precision-sensitive tasks, this substrate variant warrants caution.

---

## VI. Theoretical Complexity and Cost Modeling

### A. Modeling Assumptions
To analyze the asymptotic behavior of pointer indirection relative to linear re-ingestion, we establish five explicit modeling assumptions:
*   **A1 (Uniform Block Partitioning):** Historical context is partitioned into $N$ distinct memory units of uniform sequence length $B$, yielding total historical length $L_H = N \cdot B$.
*   **A2 (Sparse Access Frequency):** Across an interactive sequence of $K$ queries $\{Q_1, \dots, Q_K\}$ (each of length $L_Q$), each query accesses on average $m$ historical context units, where access is sparse: $m \ll N$.
*   **A3 (Compact Pointer Invariant):** Each pointer occupies $k$ tokens, where $k \ll B$ in accordance with Axiom 2.
*   **A4 (Upfront Amortization):** The initial encoding of blocks and generation of pointers ($\mathcal{C}_{\text{init}}$) is performed once upon context encounter and amortized over subsequent queries.
*   **A5 (Cached Addressing Embeddings):** The addressing projections $e(P_i)$ are computed once at registration and cached in the index $\mathcal{I}$; scoring a query requires one projection $e(Q_t)$ plus $N$ inner products, giving $\mathcal{C}_{\mathrm{select}}=\mathcal{O}(L_Q d + Nd)$. This is the design point implied by Eq.~(9), and is required by Axiom 3 (validity without re-embedding) and Axiom 6 (no re-execution of $\mathcal{G}_\phi$ at query time).

### B. Resolution Cost by Substrate
The resolution cost $\mathcal{C}_{\mathrm{res}}(m,B_{\mathrm{res}},Q_t)$ depends on the substrate realization of Section V: $\mathcal{M}_{\mathrm{text}}$ requires re-encoding fetched text at query time, $\mathcal{C}_{\mathrm{res}} = m(2dB_{\mathrm{res}}^2 + 4d^2B_{\mathrm{res}})$ --- the same order as per-block ingestion. $\mathcal{M}_{\mathrm{kv}}$ reuses paged KV states without recomputation, so $\mathcal{C}_{\mathrm{res}} \approx 0$ in FLOPs; its binding costs are host--device bandwidth and capacity, not compute. $\mathcal{M}_{\mathrm{latent}}$ incurs cross-attention unpooling over $k$ pointer tokens, $\mathcal{C}_{\mathrm{res}} = \mathcal{O}(m B_{\mathrm{res}} k d)$.

### C. Asymptotic FLOPs Analysis
Consider an LLM with hidden dimension $d$ and multi-head self-attention. (For notational simplicity, $n_{\text{layers}}$ and $n_{\text{heads}}$ are absorbed into the constant $d$ throughout this section). We contrast the computational cost across $K$ queries under linear re-ingestion versus SemPointer.

#### 1. Linear Re-Ingestion Paradigm
In standard autoregressive operation, each query re-attends to the full unpruned history $L_H$:
$$\mathcal{C}_{\text{linear}} = K \cdot \left[ 2 d (L_H + L_Q)^2 + 4 d^2 (L_H + L_Q) \right] \approx 2 K d (N B + L_Q)^2 = \mathcal{O}\left(K d N^2 B^2\right)$$

#### 2. SemPointer Indirection Paradigm
Under the SemPointer lifecycle:
1.  **One-Time Generation and Storage:** Ingesting $N$ blocks and generating $k$-token pointers incurs:
    $$\mathcal{C}_{\text{init}} = N \cdot \left[ 2 d B^2 + 4 d^2 B + \mathcal{C}_{\text{gen}}(B, k) \right]$$
    Both terms are retained; note $4d^2B$ dominates $2dB^2$ whenever $B < 2d$, which holds across typical operating points ($B \le 10^3$ vs.\ $d \ge 4096$). The generator cost $\mathcal{C}_{\mathrm{gen}}$ depends on the architecture of $\mathcal{G}_\phi$:
    *   *Pooling-class readout:* If $\mathcal{G}_\phi$ pools or projects over token representations already computed during the forward pass of block ingestion, $\mathcal{C}_{\mathrm{gen}} = \mathcal{O}(Bd)$ --- a marginal addition to $\mathcal{C}_{\mathrm{init}}$.
    *   *Attention-class generator:* If $\mathcal{G}_\phi$ executes dedicated self-attention over the block (e.g., a learned segmentation head), $\mathcal{C}_{\mathrm{gen}} = c_g(2dB^2 + 4d^2B)$ with $c_g = \Theta(1)$, roughly doubling $\mathcal{C}_{\mathrm{init}}$ for $c_g \approx 1$.
    While $\mathcal{C}_{\mathrm{gen}} \ge \Omega(Bd)$ is the theoretical floor, Proposition 2 evaluates this pooling-class lower bound, whereas Corollary 1 analyzes the attention-class regime.
2.  **Per-Query Pointer Selection:** Scoring query $Q_t$ against $N$ pointers requires:
    $$\mathcal{C}_{\mathrm{select}} = \mathcal{O}(L_Q d + Nd)$$
3.  **Per-Query Resolution and Inference:** The active prompt contains $N$ compact pointers of length $k$, $m$ resolved blocks of length $B_{\text{res}} \le B$, and query $L_Q$. The active sequence length is $L_{\text{active}} = N k + m B_{\text{res}} + L_Q$. Per-query computation is:
    $$\mathcal{C}_{\text{query}} = 2 d (N k + m B_{\text{res}} + L_Q)^2 + \mathcal{C}_{\text{res}}(m, B_{\text{res}}, Q)$$

Cumulative SemPointer complexity across $K$ queries is:
$$\mathcal{C}_{\text{SemPointer}} = \mathcal{C}_{\mathrm{init}} + K[\mathcal{C}_{\mathrm{select}} + \mathcal{C}_{\mathrm{query}}] = N(2dB^2 + 4d^2B + \mathcal{C}_{\mathrm{gen}}) + K\big[2dL_{\mathrm{active}}^2 + 4d^2 L_{\mathrm{active}} + \mathcal{C}_{\mathrm{res}} + \mathcal{C}_{\mathrm{select}}\big]$$

### D. The Indirection Crossover Threshold

> **[REASONED] Proposition 1 (Crossover Threshold):** Under A1--A5, indirection yields lower cumulative FLOPs than linear re-ingestion iff (i) $\Delta\mathcal{C}_{\mathrm{step}} > 0$ --- up to overheads, equivalently $N(B-k) > mB_{\mathrm{res}}$ --- and (ii) $K > K^\ast = \mathcal{C}_{\mathrm{init}}/\Delta\mathcal{C}_{\mathrm{step}}$:
> $$K^* = \frac{\mathcal{C}_{\mathrm{init}}}{\Delta\mathcal{C}_{\mathrm{step}}}$$
> where
> $$\Delta\mathcal{C}_{\mathrm{step}} = \big[N(B-k) - mB_{\mathrm{res}}\big]\cdot \Big[2d\big(N(B{+}k) + mB_{\mathrm{res}} + 2L_Q\big) + 4d^2\Big] - \mathcal{C}_{\mathrm{select}} - \mathcal{C}_{\mathrm{res}}$$
> Condition (i) makes the sparsity requirement of A2 concrete: if a query resolves a large fraction of memory ($mB_{\mathrm{res}} \geq N(B-k)$), indirection is strictly dominated for every $K$.

*Derivation:* The attention term factors as a difference of squares: $(L_H{+}L_Q)^2 - L_{\mathrm{active}}^2 = [N(B{-}k) - mB_{\mathrm{res}}]\,[N(B{+}k) + mB_{\mathrm{res}} + 2L_Q]$; the projection term contributes the linear difference $4d^2[L_H - L_{\mathrm{active}}] = 4d^2[N(B{-}k) - mB_{\mathrm{res}}]$. Per-query savings therefore factorize as (*tokens avoided*) $\times$ (*per-token cost coefficient*), minus addressing and resolution overheads.

> **[REASONED] Proposition 2 (Single-Query Viability):** Under A1--A5 with the $\mathcal{M}_{\mathrm{kv}}$ substrate ($\mathcal{C}_{\mathrm{res}}\approx 0$), $L_Q \ll NB$, and sparse access ($mB_{\mathrm{res}} \ll NB$), $\mathcal{C}_{\mathrm{init}} < \Delta\mathcal{C}_{\mathrm{step}}$, hence $K^\ast < 1$: indirection is FLOP-viable from the first query. Setting $mB_{\mathrm{res}}=0$ and $L_Q=0$, the viability condition reduces to $N > 1 + \mathcal{O}(k/B + dk/B^2)$, satisfied by any registry of practical size under pooling-class generation; Corollary 1 treats attention-class generators. Sparsity of access, not reuse count, is the load-bearing condition. Amortization across $K > K^\ast > 1$ queries governs only the intermediate regime where per-query resolved volume is a non-negligible fraction of the registry, or under resolution-heavy substrates ($\mathcal{M}_{\mathrm{text}}$), where $\mathcal{C}_{\mathrm{res}}$ partially reinstates the quadratic cost at every query.

> **[REASONED] Corollary 1 (Attention-Class Pointer Generators):** Proposition 2 assumes a pooling-class generator with $\mathcal{C}_{\mathrm{gen}}=\mathcal{O}(Bd)$ (e.g., readout pooling over precomputed token embeddings). If $\mathcal{G}_\phi$ is an attention-based readout, $\mathcal{C}_{\mathrm{gen}} = c_g(2dB^2 + 4d^2B)$, $c_g=\Theta(1)$ --- the same asymptotic order as block encoding --- and the viability condition reduces, to leading order, to $N > (1+c_g) + (2d/B)\,c_g$. For $c_g=1$: $N \gtrsim 2 + 2d/B$ (e.g., $N \gtrsim 18$ at $d{=}4096$, $B{=}512$; $N \gtrsim 6$ at $B{=}2048$). Expensive generators thus shift the threshold from $N\gtrsim 2$ to $N\gtrsim 2d/B$ --- still a modest registry --- below which the framework falls back to the amortization regime of Proposition 1.

> **[REASONED] Remark 1 (Scope of Comparison):** $\mathcal{C}_{\mathrm{linear}}$ models the no-cache regime: stateless inference, API access, cache eviction between sessions, or contexts exceeding cache capacity. Against a persistent KV cache with host offload [ESTABLISHED: Kwon et al., 2023], per-query recompute FLOPs approach zero and the binding resource becomes memory capacity and paging bandwidth. In that regime SemPointer's contribution is capacity, not FLOPs: retaining only $Nk$ pointer tokens in fast memory while full context resides in secondary storage. The crossover analysis covers the re-ingestion regime; the capacity argument extends to the cached regime.

---

## VII. Theoretical Vulnerabilities and Failure Modes

A comprehensive theoretical framework must characterize the conditions under which it can fail. Three primary risk categories are examined below:

### A. Address Collision via Generalized Birthday Bounds

> **[REASONED] Proposition 3 (Combinatorial Address Collision Bound):** Let pointer space $\mathcal{P}$ consist of sequences of length $k$ over a discrete vocabulary $\mathcal{V}_p$ of size $V = |\mathcal{V}_p|$. Under the assumption of independent, uniformly distributed pointer generation, the probability that any two of $N$ registered memory items share an identical address is bounded by:
> $$P_{\text{collision}} \approx 1 - \exp\left(-\frac{N(N-1)}{2 V^k}\right)$$

*Proof:* This is the generalized birthday problem over an address space of cardinality $|\mathcal{P}|=V^k$, with the standard exponential approximation under independent uniform assignment.

> **[REASONED] Proposition 4 (Combinatorial Address Space Capacity):** Under the uniform birthday-bound model, to maintain collision probability below threshold $\delta$ across a memory registry of $N$ items, the minimum discrete pointer length $k$ must satisfy:
> $$k \ge \left\lceil \frac{\ln(N^2) - \ln(2 \delta)}{\ln V} \right\rceil$$
> For illustration, with $N = 10^5$, $V = 32{,}000$, and $\delta = 10^{-4}$:
> $$k \ge \left\lceil \frac{\ln(10^{10}) - \ln(2 \times 10^{-4})}{\ln(32{,}000)} \right\rceil = \left\lceil \frac{23.02 + 8.51}{10.37} \right\rceil = 4$$
> This calculation concerns combinatorial address-space capacity only; it does not establish that a learned pointer generator can realize such a distribution in practice. Proposition 4's lower bound ($k \ge 4$ under $N=10^5$, $V=32{,}000$, $\delta=10^{-4}$) and Axiom 2's compactness pressure jointly restrict the viable pointer length to $k \in [4, 8]$.

### B. Semantic Drift Across Extended Horizons
In transformer architectures with relative positional encodings such as RoPE, the distance between a fixed pointer $P_i$ and the generation head grows as interactions accumulate: $\Delta t = t_{\text{curr}} - t_{\text{gen}}$.

To analyze potential drift analytically, we consider a phenomenological decay model for attention over temporal displacement:
$$\mathbb{E}\left[\alpha(Q_t, P_i)\right] \propto \exp\left(-\lambda \cdot |t - t_i|\right)$$
This formulation serves as a phenomenological model to evaluate drift dynamics under positional displacement, rather than a mathematical derivation of RoPE attention geometry. Under this model, as $|t - t_i|$ grows, attention allocated to historical pointers decays. Positional displacement can therefore degrade pointer salience over extended horizons unless position-invariant masking or address refreshing intervenes.

### C. Resolution Divergence and Hallucination
When resolution operator $\mathcal{R}_{\theta}$ operates on a lossy substrate (such as $\mathcal{M}_{\text{latent}}$), the generated text $\hat{X}_i$ may deviate from the true historical content:
$$\mathcal{D}_{\text{KL}}\left(p(X_i) \parallel p(\hat{X}_i \mid P_i, M_i)\right) > 0$$
If this divergence exceeds the tolerance of downstream reasoning, the resolver can introduce hallucinations. For precision-sensitive tasks, this risk makes raw text ($\mathcal{M}_{\text{text}}$) or KV cache ($\mathcal{M}_{\text{kv}}$) substrates preferable.

---

## VIII. Critical Boundary Analysis and Literature Comparison

### A. Disambiguation from Landmark Attention
Landmark Attention [ESTABLISHED: Mohtashami & Jaggi, 2023] uses special tokens to route attention across blocks. Three architectural differences separate it from SemPointer:
1.  Landmark Attention creates a sparse attention pattern within a continuous input sequence, inserting landmark tokens to gate attention during inference. SemPointer is proposed as a persistent memory abstraction: pointers exist in an address registry independently of active context and dereference an external substrate across separate queries and sessions.
2.  A landmark token acts as an internal sequence marker that routes attention during the forward pass. A SemPointer is an addressable token sequence that can be stored, composed, and dereferenced across turns.
3.  Landmark Attention was developed to extend single-sequence context length; multi-turn interactions still require keeping tokens or KV states in active memory. SemPointer is designed to permit eviction of the substrate to secondary storage while retaining only the compact pointer in working memory.

### B. Disambiguation from Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) [ESTABLISHED: Lewis et al., 2020; Cheng et al., 2024] and SemPointer both perform query-conditioned selection over an external store. The main architectural distinction lies in Axiom 7 (Algebraic Compositionality).

Standard RAG pipelines retrieve passages as independent text blocks and append them directly to the prompt context, leaving relational reasoning entirely to downstream attention layers. In contrast, SemPointer maintains an address space of composable references ($P_a \oplus P_b$), enabling compound pointers to be registered and dereferenced jointly (Axiom 7).

To illustrate, consider a two-hop query: *"What did the vendor mentioned in turn 12 quote in the email from turn 40?"* Under standard RAG, both passages are retrieved independently and concatenated into context. Under SemPointer, the pointer for turn 40's email can reference the pointer to turn 12's vendor entry, forming a joint address $(P_{12} \oplus P_{40})$ that is resolved once against the combined substrate $\{M_{12}, M_{40}\}$ conditioned on $Q_t$. This difference is structural, governing how addresses are composed and stored rather than how passages are scored for relevance.

### C. Feature-Level Comparison with Related Approaches

*Table II: Architectural Comparison of Context-Management and Memory Approaches. Entries characterize each system at the level of its published description [5]--[14]; see Section VIII-B for a worked comparison of composable addressing (Axiom 7) against standard RAG concatenation.*

| Framework | Primary Representation | Representation Type | Decoupled Substrate? | Explicit Address? | Selective Access? | Composable Addressing? | Multi-Turn Reusability? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Gist Tokens** [Mu et al., 2023] | Soft Tokens | Continuous | No | No | No | No | Yes |
| **AutoCompressors** [Chevalier et al., 2023] | Summary Vectors | Continuous | No | No | No | No | Yes |
| **ICAE** [Ge et al., 2024] | Memory Slots | Continuous | No | No | No | No | Yes |
| **500xCompressor** [Li et al., 2025] | Special Tokens | Discrete/Latent | No | No | No | No | No |
| **SAC** [Liu et al., 2025] | Anchor Tokens | Token KV | No | No | No | No | No |
| **CCM** [Kim et al., 2024] | Compressed KV | KV Tensors | Yes (KV) | No | No | No | Yes |
| **LLMLingua-2** [Pan et al., 2024] | Pruned Text | Discrete Text | No | No | No | No | No |
| **xRAG** [Cheng et al., 2024] | Document Embedding | Continuous | Yes (Ext DB) | Partial | Yes (Retrieval) | No | Yes |
| **Landmark Attention** [Mohtashami & Jaggi, 2023] | Landmark Tokens | Token Markers | No | No | Yes (Blocks) | No | No |
| **SemPointer (Proposed)** | **Semantic Pointer** | **Token Seq. / Latent** | **Yes (Multi-tier)** | **Yes** | **Yes (Query-Cond.)** | **Yes** | **Yes** |

To anchor these classifications in published mechanisms:
*   **Explicit Address and Decoupled Substrate:** Soft-token baselines---Gist Tokens [5, §2], AutoCompressors [6, §3.1], and ICAE [7, §2]---lack an independent substrate decoupled from attention, acting as prompt prefixes rather than dereferenceable addresses. 500xCompressor [8, §3] collapses context into an inseparable summary token. CCM [9, §3.2] offloads merged KV states (*Yes (KV)*) via implicit cache eviction rather than explicit addresses (*No*). xRAG [14, §2.2] retrieves external documents via continuous dense vectors (*Partial* address, *Yes (Ext DB)* substrate) rather than discrete sequence pointers.
*   **Composable Addressing (Axiom 7):** In AutoCompressors [6, §3.1], summary vectors $\mathbf{s}_i$ are position-dependent chunk condensations lacking an address algebra $\oplus$ to bind non-adjacent references without intermediate vectors (*No*). Gist Tokens [5] and ICAE [7] require all active summaries to reside contiguously in context, delegating relational binding to quadratic attention. Landmark Attention [12, §3] inserts markers for intra-sequence routing, providing no cross-turn address handles (*No*). In contrast, SemPointer maintains a pointer index supporting algebraic composition ($P_a \oplus P_b$), resolving compound references against decoupled substrates.

---

## IX. Falsifiable Hypotheses and Proposed Evaluation Agenda

To guide subsequent empirical investigation, this section formalizes six testable hypotheses alongside an operational benchmarking suite.

### A. Formal Hypotheses

*   **[SPECULATION] Hypothesis 1 (Address Validity and Uniqueness):** A compact semantic pointer $P_i$ of length $k \le 4$ (the stringent end of the range in Section IV-A) generated by $\mathcal{G}_{\phi}$ satisfies address distinctiveness:
    $$H_1: \operatorname{Accuracy}\left(\operatorname{Select}(\mathcal{I}, Q_t) = i^* \mid Q_t \leftrightarrow X_{i^*}\right) \ge 1 - \epsilon_1$$
    with $\epsilon_1 = 0.05$, matching $\epsilon_2$; both tolerances are fixed before evaluation.  
    *Null Hypothesis ($H_{1,0}$):* Pointers degenerate into overlapping representations whose retrieval accuracy is indistinguishable from random selection ($p > 0.05$).
*   **[SPECULATION] Hypothesis 2 (Resolution Fidelity Preservation):** Let $S(\cdot)$ denote downstream task performance. Pointer resolution $\hat{X}_i = \mathcal{R}(P_i, M_i, Q_t)$ preserves task-essential factual information:
    $$H_2: S\left(f(Q_t, \hat{X}_i)\right) \ge (1 - \epsilon_2) S\left(f(Q_t, X_i)\right)$$
    with tolerance $\epsilon_2 \le 0.05$.  
    *Null Hypothesis ($H_{2,0}$):* Resolution introduces factual corruption leading to significant degradation compared to full context: $S(f(Q_t, \hat{X}_i)) < (1 - \epsilon_2) S(f(Q_t, X_i))$.
*   **[SPECULATION] Hypothesis 3 (Random-Access Efficiency Gain):** For query streams requiring selective access ($m \ll N$), SemPointer reduces total inference energy and FLOPs relative to full context:
    $$H_3: \mathcal{C}_{\text{SemPointer}}(K) < \mathcal{C}_{\text{FullContext}}(K) \quad \forall K > K^*$$
    *Null Hypothesis ($H_{3,0}$):* The overhead of pointer generation and resolution cancels all savings, yielding $K^* \to \infty$.
*   **[SPECULATION] Hypothesis 4 (Multi-Turn Amortization Advantage):** The marginal computational cost per query under SemPointer is strictly sublinear with respect to cumulative conversation length $L_{\text{cum}}$:
    $$H_4: \frac{\partial \mathcal{C}_{\text{step}}}{\partial L_{\text{cum}}} \approx \mathcal{O}(k) \ll \mathcal{O}(L_{\text{cum}})$$
    *Null Hypothesis ($H_{4,0}$):* Active memory expansion scales linearly with historical volume.
*   **[SPECULATION] Hypothesis 5 (Compactness-Fidelity Tradeoff Curve):** Resolution fidelity $\Phi$ exhibits monotonic degradation with pointer compression ratio $\kappa = k / |X_i|$:
    $$H_5: \frac{\partial \Phi}{\partial k} > 0, \quad \text{with saturation at } k \ge k_{\text{crit}}$$
    *Null Hypothesis ($H_{5,0}$):* Fidelity does not increase monotonically with $k$, or exhibits no saturation within $k \in [1,8]$.
*   **[SPECULATION] Hypothesis 6 (Interference Sensitivity in High-Density Manifolds):** Selection accuracy degrades gracefully as memory cardinality $N$ scales to $10^5$, obeying the logarithmic capacity bound:
    $$H_6: \Delta \operatorname{Accuracy} \le \beta \log_{10}(N)$$
    *Null Hypothesis ($H_{6,0}$):* Degradation is not graceful --- super-logarithmic in $N$, or collapse below $N = 10^5$.

### B. Proposed Empirical Benchmark Suite
Future empirical validation should examine three diagnostic scenarios:
1.  **Variable-Indirection Needle-in-a-Haystack (VI-NIAH):** Dispersing $N$ factual needles across a $100\text{k}$-token context. Queries require retrieving needle facts via pointers without providing lexical keywords, testing purely semantic addressing.
2.  **Multi-Hop Pointer Traversal:** Queries where needle $A$ contains the semantic pointer to needle $B$, evaluating recursive pointer dereferencing and compositionality.
3.  **Long-Horizon Amortized Chat Benchmark:** Simulating 100-turn conversations where earlier topics are revisited sporadically at turns 10, 45, and 90, directly measuring whether $K > K^*$ FLOP reductions materialize under practical GPU memory constraints.

---

## X. Discussion and Open Challenges

The previous sections present the formal architecture. Three open questions carry practical implications for future implementations:

### A. Differentiable versus Discrete Addressing
The choice of pointer representation involves conflicting tradeoffs. Discrete vocabulary tokens are human-interpretable, cache-compatible, and supported by standard tokenization pipelines. However, they impose a quantization bottleneck: the pointer space is finite, and gradients cannot flow through discrete selection without additional estimators such as Gumbel-softmax or REINFORCE. Continuous soft pointers avoid this restriction and offer flexible expressive capacity, but they introduce opacity and susceptibility to positional drift across transformer layers. Neither option is clearly superior; the choice depends on the implementation of $\mathcal{G}_\phi$ and the resolution operator.

### B. Memory Granularity Dynamics
Fixed-size chunking is an analytical simplification. The model assumes a uniform block size $B$, whereas semantic boundaries in conversation rarely align with fixed token windows. A claim that begins in turn 7 and concludes in turn 8 warrants a single pointer rather than two. A practical generator $\mathcal{G}_\phi$ may therefore require a learned segmentation head to identify proposition or turn boundaries, introducing a training objective that the current formalism leaves open.

### C. Safety, Provenance, and Indirection Attacks
Decoupling pointers from storage creates security considerations distinct from standard retrieval pipelines. An adversary who influences pointer generation by crafting inputs that collide with existing memory addresses could trigger unauthorized memory resolution, exposing historical context inappropriately. Access permissions, pointer provenance tracking, and collision detection become necessary system safeguards. These risks are inherent to reference-based memory access in generative models; they do not arise where context is copied rather than addressed.

---

## XI. Conclusion

This paper's claim is deliberately narrow: if token-level semantic addresses satisfying Axioms 1--7 can be generated and resolved with bounded fidelity, then address-based indirection dominates uniform re-ingestion in cumulative FLOPs within the viability regime of Propositions 1--2, and fails outside it. The framework's weakest link is learnability: nothing here shows that $\mathcal{G}_\phi$ and $\mathcal{R}_\theta$ can be trained to meet Axiom 4's fidelity bound, and until Hypotheses 1--2 are tested --- even on a toy substrate --- the crossover analysis rests on an assumption the framework has not yet earned. What the paper does establish is the scaffolding for that test: the axioms, the substrate-dependent cost model, the collision and drift analyses, and the benchmark suite.

**Data Availability:** Not applicable --- no datasets were generated or analyzed.  
**Funding:** This research received no specific grant.  
**Conflict of Interest:** The author declares none.  
**Ethics:** This work is purely theoretical; no human subjects or personal data were involved.
