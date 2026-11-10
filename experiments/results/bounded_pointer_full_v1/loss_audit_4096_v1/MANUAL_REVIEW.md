# Manual review notes: loss audit at 4K

These notes supplement the exhaustive CPU audit in [REPORT.md](REPORT.md). The automated casebook includes all 229 losses; the examples below received additional semantic spot checks. They are illustrative, not an independently double-annotated labeling of every loss.

## What the available evidence can establish

| Evidence check | Cases | Interpretation |
| --- | ---: | --- |
| Fine-grained annotations aligned to the frozen source | 43 | 13 RULER fact/chain cases, 26 LoCoMo cases, 4 QASPER cases |
| Annotated units omitted by bounded, all retained by BM25 | 18 | 9 RULER, 6 LoCoMo, 3 QASPER; support-coverage difference, not automatically 18 causal retrieval failures |
| All annotated units present in bounded | 9 | Does not guarantee the answer is derivable solely from those units or implicate pointers |
| Annotated units incomplete in both payloads | 16 | BM25 can still score higher without recovering every annotated unit |
| LongMemEval support sessions | 11 | 4 wholly absent from bounded payload, 7 partially represented; session-level labels are too coarse to prove necessary evidence missing |
| Annotations unavailable or unaligned | 175 | Includes 173 without usable positive annotations and 2 unaligned fine-grained cases; explicit uncertainty is retained |

The first three fine-grained outcomes partition the 43 aligned cases: 18 + 9 + 16. The 11 coarse cases and 175 unavailable/unaligned cases bring the total to 229.

The 18 “BM25 retains” cases cannot be interpreted as 18 guaranteed wins from enlarging evidence. For example, one omitted annotated paragraph repeats facts that the bounded model already states correctly, and a BM25 answer can still be a refusal despite receiving a slightly higher lexical score.

## Source-grounded examples

Case numbers refer to [casebook.md](casebook.md) and [cases.jsonl](cases.jsonl).

| Case | Task | Saved outcome | What it suggests |
| --- | --- | --- | --- |
| 062 | LoCoMo: recipe ingredients | Bounded expands blocks 4 and 18, omitting annotated turn D8:19 in block 9. It answers “coconut milk” (0.50). BM25 includes block 9 and lists milk, vanilla, sugar and salt (0.80). Pointer-off also says only coconut milk. | Clear practical example of a missing supporting fact; pointer removal alone does not repair it |
| 071 | LoCoMo: tournament | Bounded omits turn D27:1 in block 33 and says Street Fighter. BM25 includes it and answers Valorant. | Relevant-event evidence was omitted |
| 175 | RULER UUID | Supporting key-value fact crosses blocks 6 and 7. Bounded selects blocks 6 and 3; BM25 also selects 7 and returns the UUID. | Retrieval/chunk-boundary failure; preserve adjacent text that completes a fact |
| 176 | RULER UUID | Supporting fact crosses blocks 4 and 5. Bounded selects 4 and 3; BM25 includes 5 and answers correctly. | Second boundary-spanning failure; raising the cap alone does not force the router to retrieve its neighbor |
| 178 | RULER variable tracking | Four of five variables are returned; assignment for AIPIB in block 22 is omitted. BM25 includes block 22 and returns all five. Both bounded and pointer-off score 0.80; BM25 scores 1.00. | Incomplete multi-fact coverage rather than pointer text alone |
| 179 | RULER variable tracking | Bounded retrieves the root assignment but omits four required chain assignments in blocks 1, 2, 3 and 6. BM25 retrieves the chain. | Lexical coverage of the query value is not sufficient evidence coverage |
| 174 | RULER number lookup | Full supporting fact is present, yet bounded and pointer-off answer “367” instead of “3675709.” BM25 answers correctly. | Generation/context-use failure persists without pointers; no output-cap explanation is established |
| 177 | RULER UUID | Full UUID is present in both bounded and pointer-off payloads. Bounded miscopies it; pointer-off and BM25 copy it exactly. Scores 0 / 1 / 1. | A strong saved example of sensitivity to adding the pointer section; repeated runs would be needed to characterize reliability |

All nine variable-tracking losses have incomplete annotated chains in the bounded payload. BM25 contains complete chains in seven of them; the other two remain incomplete in both arms.

## Metric and reference issues: a loss is not necessarily an incorrect answer

| Case | Observation | Implication |
| --- | --- | --- |
| 064 | Reference: “lactose intolerance.” Both bounded and pointer-off correctly explain that Joanna is lactose intolerant, scoring 0.154. BM25 gives a shorter equivalent answer and scores 0.286. | Lexical F1/verbosity difference, not demonstrated missing evidence |
| 065 | Both bounded and BM25 express Nate's respect for Joanna's resilience; their phrasing differs. Scores approximately 0.32 vs 0.57. | Token-overlap loss need not imply a factual error |
| 096 | Bounded explicitly states the reference date, 29 August 2023, within a longer explanation, scoring 0.286. BM25 uses a relative-date explanation and scores 0.375. | A lower F1 can coexist with an explicit correct date |
| 205 | Reference asks for average dialog length. Bounded says “I don't know” (0). BM25 also refuses, but its longer refusal scores 0.10 because of shared reference tokens. | BM25's numerical win is not a successful answer; even its retained paragraph did not cause it to answer correctly |
| 207 | Bounded lists “premises, backing, claim, rebuttal, and refutation” (0.833). BM25 uses singular “premise” (1.0). An annotated paragraph is omitted by bounded, but it already produces the component list. | Annotation omission does not prove a causal retrieval failure; singular/plural mismatch matters |
| 204 | QASPER's saved references include both “only adjacent mentions” and “more than that ... next to adjacent.” | Reference ambiguity and phrasing prevent a clean causal conclusion from the score comparison |

Cases 064, 065, 096 and 207 were semantically spot-checked here. Their interpretation is a qualitative judgment, not a replacement judge score for the dataset.

## Inference/abstention protocol cases

- **Case 083:** LoCoMo asks what Star Wars book Tim might enjoy. The reference names a specific book and authors; the annotated turns establish preferences rather than naming that recommendation. Both systems largely abstain. The reference asks for an inference/recommendation that may require knowledge beyond literal stored text, while the evaluation system prompt says to answer only from memory.
- **Case 085:** The reference lists Star Wars filming locations in Ireland. Bounded and BM25 both say the memory does not provide those locations; small wording differences create the score gap. Retrieving every annotated preference/travel turn does not supply the literal named locations.
- **Case 074:** The cited utterance is spoken by John, while the question asks about Tim's injury. Neighboring context and dates need review; the presence of the annotated utterance does not settle the question.
- Eight LoCoMo losses are category-5 abstention questions. There is no positive gold span whose literal retrieval would establish success.

These are protocol/annotation limitations to document, not grounds to relabel every loss or assume the gold is wrong.

## Repair priorities suggested by this audit

1. **Preserve fact boundaries.** Test sentence-aware block boundaries or a limited adjacent-block expansion when a selected block ends mid-fact. Two exact UUID examples provide a concrete target.
2. **Check complete evidence sets.** Variable chains and multi-event questions can require information beyond the query-matching block. Expanding the candidate route within the existing 4K cap is a more precise intervention than merely increasing the cap.
3. **Measure the pointer intervention separately.** All 229 cases had identical expanded text in bounded and no-index. Removing pointer text improved 94 of these selected losses and matched/exceeded BM25 in 47. These are overlapping with annotation categories, not extra cases to add to 229.
4. **Retain the whole-test counterbalance.** On all 1,000 4K questions, bounded vs no-index is 134 wins / 740 ties / 126 losses. Improvements among the selected 229 losses do not justify concluding that removing pointers improves the whole test.
5. **Treat summary quality separately.** Ninety-one losses are summary tasks, including six SAMSum cases scored by the local QA-F1 adapter. Summary coverage cannot be established by searching for the reference text. A two/four-block policy may underrepresent document events, but this audit does not validate an optimal summary budget.
6. **Audit scoring and abstention before stronger claims.** Do not call the 229 score losses 229 factual failures, and do not treat all 18 annotation omissions as 18 router-caused failures.

None of these proposed repairs has been implemented or benchmarked by this audit. The completed 1,000-question test is exposed; a claimed improvement needs a new protocol and fresh confirmation data.

## Verification

Ten targeted audit tests pass, including prompt/query separation, evidence-versus-label guards, word-boundary matching, token-span coverage, causal-label restraint, and preserving LoCoMo's exact newline alignment. The full offline non-GPU suite recorded 129 passed, 7 failed, 21 setup errors, 7 deselected, and one warning. The 28 failure/error test names and traces are retained in [full_tests.log](full_tests.log); they match the previously documented missing legacy model/tokenizer caches and paper-assets assertion. No GPU inference was run.
