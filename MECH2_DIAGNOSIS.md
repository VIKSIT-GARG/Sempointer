# MECH-2 diagnosis — why v1 agentic routing failed (pilot_multihop_v2, n=30 musique)

Source: `experiments/results/pilot_multihop_v2/predictions.jsonl` (30 `agentic_pointer`
rows; `iterative_rag`/`hybrid` shown only for selection-behavior contrast).
Headline: agentic_pointer mean F1 **0.030** vs dense **0.054** at ~3.1x tokens
(mean active_tokens 3285 vs ~1.1–1.7k for iterative_rag rows).

## 1. Routing behavior (the smoking gun)

| stat (n=30) | agentic_pointer v1 |
|---|---|
| routed exactly `[0]` | **23/30 (77%)** |
| single-block selections | 29/30 (97%) |
| multi-block selections | **0/30 (0%)** |
| empty selections | 1/30 |
| score > 0 | 11/30; score = 0 on 19/30 |
| answers with abstention language ("don't know"/"cannot answer"/"do not contain") | 15/30 |

Musique is a 2-hop benchmark: a correct route needs 2+ blocks. v1 never
selected more than one block in 30 questions and collapsed to block 0 in over
three-quarters of them. The `m=4` cap was never binding — the router
under-selects, it is not budget-limited.

Contrast: `iterative_rag` selected 2 blocks on 10/30 and routed to `[0]` once;
`hybrid` spread across ids 17/7/0/22/6/29/19. Only the LLM router shows
position-0 collapse.

## 2. Routing-precision vs reader-accuracy decomposition (proxy-based)

HONEST LIMITATION: `predictions.jsonl` logs `selected_ids`, `answer`, and
`gold` answer strings, but NOT `routing_raw` (router's literal reply) and NOT
gold supporting-block ids. True routing precision (routed set contains a gold
block) is therefore **not directly measurable** from this file. The split below
is a proxy:

- **Router-dominant signature:** 23/30 routes are `[0]` regardless of question;
  0/30 are multi-block on 2-hop questions; 15/30 final answers abstain for lack
  of evidence. A reader given one wrong block cannot answer — abstention is the
  expected downstream symptom of a routing failure, not an independent reader
  failure.
- **Reader probe:** on the 7 questions where v1 routed away from block 0
  (ids 4/7/13/15/16/18), 4 scored > 0 (0.043, 0.054, 0.10, 0.10) vs 7/23 of the
  block-0 routes scoring > 0 (mostly tiny partial overlaps 0.04–0.15). Small
  sample, but the reader extracts partial/full credit when given a non-default
  block — consistent with "reader works when routed correctly".
- **Empty parses:** only 1/30. So the failure is NOT unparseable output (the
  lenient parser's fallback path); it is a router that confidently returns the
  wrong (first) block. Fixing parse robustness alone would change ~nothing.

CONCLUSION: top failure mode = **router position-collapse + single-block
under-selection**. Mechanism hypothesis: 33 blocks × 300-char front-loaded
previews ≈ 3.3k-token routing prompt of near-identical `Passage N:` openings on
a 3B router; "reply-only-numbers" grammar gives no format anchor, and nothing
in the instruction demands multi-block selection. The model satisficies with
block 0.

## 3. What v2 targets (and what it does not)

`experiments/baselines/agentic_pointer_v2.py` uses exactly two of the three
allowed fixes, chosen to hit the diagnosed mode:
(a) keyword-enriched index lines (discriminative per-block signal beyond the
front-loaded preview — attacks block-0 bias) and (b) constrained routing
(numbered shortlist + strict `IDS: a, b` grammar + retry-on-unparse once —
attacks silent single-wrong-block returns and explicitly demands multi-block
selection for multi-hop questions). (c) route-then-rerank was rejected: it
spends a third call + tokens on answer coverage while the deficit is routing
precision.

If v2 still routes `[0]`-dominant single blocks, the diagnosis is wrong and the
concession conditions in `MECH2_HANDOFF.md` apply.
