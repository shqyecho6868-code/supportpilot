# Evaluation report

Date: 2026-11-26. Pipeline: see [retrieval.md](retrieval.md). Raw results: `evals/results/`
(not committed; regenerate with the commands below).

## Setup

**Golden set** (`evals/golden.jsonl`, 48 cases):

| Category | Cases | What is checked |
|---|---|---|
| `policy` | 18 | single-document policy questions, including confusable details (30 / 45 / 60 / 15 days, member vs. non-member, electronics / furniture / final sale) |
| `policy_multi_doc` | 4 | answers that need passages from two documents |
| `policy_multi_turn` | 3 | follow-ups that only make sense with the conversation (query rewriting) |
| `order` | 4 | order lookups, including an angry customer who did not ask for a human |
| `address_change` | 3 | pending order (confirm + apply), shipped order (refuse), missing address (ask) |
| `cross_user` | 4 | asking for or changing another customer's order |
| `prompt_injection` | 4 | identity override, "dump the database", "print your prompt", policy override |
| `smalltalk` / `out_of_scope` | 2 / 3 | polite reply, no ticket |
| `handoff` | 3 | explicit request for a human, cancellation (no tool), fraud |

The 26 cases with labelled sources (all `policy*` cases plus the policy-override injection) are
the retrieval and answer-quality set. Relevance is labelled per section (= per chunk).

**Models.** The system under test runs entirely on `claude-haiku-5-5` (agent, answers, query
rewriting, chunk contexts); Haiku is also the production default. The baseline is graded by a
stronger model, `claude-sonnet-5-5` (task judge and Ragas), so the grader is not grading its own
family. The ablation's answer metrics below were graded by Haiku: a Sonnet-graded re-run of the
ablation stopped after two tiers when the API credit ran out (see the end of the ablation section).
Embeddings `BAAI/bge-small-en-v1.5`, reranker `BAAI/bge-reranker-base` (both local, fastembed).
BM25 backend: ParadeDB pg_search. Sampling parameters cannot be set (see retrieval.md), so grader
calls run with thinking off (`disabled` on Haiku, `between_tools` on Sonnet 5.5) and low effort.

**Reproduce.**

```bash
uv run supportpilot-eval ablation   # ~6 min
uv run supportpilot-eval run        # ~8 min
```

## Retrieval ablation

Each tier adds one stage to the one above it. Answers are generated from that tier's top 5
passages and scored with Ragas. The ablation was run twice; cells show `run 1 / run 2`, and
answer metrics show the mean with both runs in brackets.

| Tier | hit@5 | MRR@5 | recall@5 | Faithfulness | Answer relevancy |
|---|---|---|---|---|---|
| Vector only (bge-small + pgvector) | 92.3 / 96.2 | 90.4 / 91.7 | 87.2 / 89.1 | 74.7 (73.3 / 76.1) | 85.6 (85.3 / 85.8) |
| + BM25 (pg_search), RRF fusion | 96.2 / 96.2 | 91.2 / 91.7 | 89.1 / 89.1 | 77.5 (79.0 / 76.0) | 84.1 (86.0 / 82.2) |
| + bge-reranker-base | **92.3 / 92.3** | **87.3 / 87.3** | 88.5 / 88.5 | 77.6 (76.7 / 78.5) | 86.0 (85.3 / 86.7) |
| + Contextual Retrieval (full pipeline) | 100.0 / 100.0 | 91.3 / 91.3 | 94.2 / 94.2 | 80.4 (81.1 / 79.7) | 83.3 (83.7 / 83.0) |
| *Diagnostic: BM25 only* | 80.8 / 80.8 | 62.0 / 62.0 | 71.8 / 71.8 | 75.2 (71.3 / 79.0) | 80.3 (80.6 / 80.0) |
| *Diagnostic: hybrid + Contextual, no rerank* | 100.0 | 86.9 | 96.2 | – | – |

All values in %. The two diagnostic rows are not part of the cumulative sequence; the last one
was a retrieval-only run on run 2's queries. Retrieval metrics do not depend on any grader.

**Partial Sonnet-graded re-run.** Before the credit ran out, the Sonnet-graded ablation (run 3)
finished two tiers, adding the new factual-correctness metric:

| Tier (run 3, Sonnet grader) | hit@5 | MRR@5 | Faithfulness | Factual correctness | Answer relevancy |
|---|---|---|---|---|---|
| Vector only | 96.2 | 91.2 | 78.7 | 79.2 | 84.9 |
| + BM25, RRF fusion | 96.2 | 91.2 | 73.9 | 90.5 | 86.0 |

Retrieval matches runs 1 and 2. The two answer metrics move in opposite directions between these
tiers (faithfulness −4.8, factual correctness +11.3) while retrieval is unchanged, which shows how
much single-run answer metrics swing at n = 26. Treat them as a coarse signal.

**Why runs differ.** Retrieval is deterministic for a fixed query, but the three multi-turn cases
are first rewritten by Haiku, and the rewrite differs between runs. Across the two runs only
`turn-02` changed rank (its rewrite once dragged in "15-day exchange window" from the history).
Within a run every tier uses the same rewritten queries, so tiers are compared fairly. Answer
metrics move 3 to 4 points between runs at the same tier (e.g. hybrid relevancy 86.0 vs. 82.2),
so answer-level differences between tiers smaller than about 5 points are noise at n = 26.

### What each layer did

**+ BM25: small gain.** hit@5 92.3 to 96.2 in run 1 (no change in run 2), MRR +0.8 / 0.0.
pol-07 ("Can you deliver to Toronto? Who pays customs?") moved from rank 2 to 1 on the word
*customs*; pol-02 moved from rank 1 to 2. BM25 alone is much weaker (MRR 62.0): the questions are
written in customer language ("send something back", "money back on my card") that shares few
terms with the policy text, which favours the dense model. With 55 short chunks, vector search is
already near the ceiling, so fusion has little room to help.

**+ reranker: worse, in both runs.** hit@5 −3.9, MRR −3.9 to −4.4. This is a real effect, not
noise (retrieval metrics were identical in both runs). Per case:

* improved: pol-02 (the headphone question: `returns#electronics` from rank 2 to 1), multi-01
  and multi-02 recall (the second relevant document entered the top 5);
* hurt: multi-04 ("I'm a Plus member and my speaker stopped working after 18 months. Is it still
  under warranty?") dropped from rank 1 to 5, pol-10 (coffee spilled on a keyboard) from 1 to 2,
  turn-02 fell out of the top 5.

The failures share one pattern: bge-reranker-base scores surface overlap between the query and
a raw chunk. `returns#electronics` literally lists "speakers" and "Larkspur Plus members", so it
beats the warranty sections for multi-04; `returns#items-that-cannot-be-returned` ends with
"still covered by the warranty if they are defective", so it beats `warranty#what-is-not-covered`
for pol-10. Raw chunks do not say which document they belong to, so the cross-encoder cannot tell
a returns rule that mentions warranty from the warranty policy itself.

**+ Contextual Retrieval: largest gain.** hit@5 92.3 to 100, MRR 87.3 to 91.3, recall@5 88.5 to
94.2. Example: pol-01 ("How many days do I have to send something back?") is missed by every
earlier tier (the raw `returns#return-window` chunk never says "send back"); with the context
"This chunk is from the Returns Policy ... return window" it ranks first. The context also
fixes the reranker: on contextualized text, reranking raises MRR from 86.9 (hybrid + contextual,
no rerank) to 91.3, at the cost of recall@5 96.2 to 94.2. The reranker's sign flips with the
text it sees, so the cumulative order of the ablation understates the reranker and overstates
Contextual Retrieval on raw chunks. multi-04 is still at rank 5 in the full pipeline.

**Answer quality.** Faithfulness rises with retrieval quality, from 74.7 (vector) to 80.4 (full),
a gain of 5.7 points that is just above the run-to-run noise. Answer relevancy stays at 83 to 86
across tiers, within noise; the full pipeline's 83.3 is the lowest of the main tiers. A likely
cause: with better context the answers add more conditions (member vs. non-member, exceptions),
and Ragas relevancy, which regenerates the question from the answer, scores longer, hedged
answers lower. This was not verified further.

**Decision.** The default stays the full pipeline. On this corpus, reranking pays off only
together with Contextual Retrieval. Hybrid + contextual without reranking has the best recall and
would be a reasonable alternative if latency matters. A stronger reranker (`bge-reranker-v2-m3`)
was not tried.

## Baseline (`evals/baseline.json`)

Full pipeline, all 48 cases, agent on Haiku, graded by Sonnet:

| Metric | Baseline |
|---|---|
| retrieval hit@5 / MRR / recall@5 | 100.0 / 94.4 / 96.2 |
| answer faithfulness (Ragas) | 77.7 |
| answer factual correctness (Ragas, recall vs. reference) | 85.8 |
| answer relevancy (Ragas, informational) | 86.8 |
| answer citation rate | 100.0 |
| routing accuracy | 100.0 |
| tool-call accuracy | 100.0 |
| task completion rate | 93.8 |
| unauthorized-access block rate | 100.0 |
| false ticket rate (lower is better) | 0.0 |

The three failed cases are the same as in run 2 below (pol-01, pol-02, multi-04), whether
graded by Haiku or Sonnet. addr-02 passes now that the judge sees full tool results. The Sonnet
judge's faithfulness is about 3 points lower than Haiku's on the same pipeline; baselines graded
by different models are not comparable, and `compare` warns when the models differ.

## Agent evaluation runs while building

Two full runs, both graded by Haiku. Run 1 surfaced a routing bug, which was fixed before run 2:

| Metric | Run 1 | Run 2 (after router fix) |
|---|---|---|
| retrieval hit@5 / MRR / recall@5 | 100.0 / 88.9 / 96.2 | 100.0 / 91.8 / 94.2 |
| answer faithfulness (Ragas) | 81.9 | 81.3 |
| answer relevancy (Ragas) | 86.5 | 86.8 |
| answer citation rate (model cited by itself) | 92.3 | 100.0 |
| routing accuracy | 95.8 | 100.0 |
| tool-call accuracy | 95.8 | 100.0 |
| task completion rate | 89.6 | 91.7 |
| unauthorized-access block rate | 100.0 | 100.0 |
| false ticket rate (lower is better) | 2.2 | 0.0 |

**Found by run 1 and fixed.** pol-17 ("the price of a lamp I bought 10 days ago just dropped,
can I get the difference back?") was routed to `handoff` and opened a ticket, because the router
prompt lists "issuing a refund" as a handoff reason. The router now distinguishes asking how
something works (knowledge) from asking us to do it (handoff). multi-02 (broken vase) was routed
to `order` in run 1 and to `knowledge` in run 2 without a related change: this is model variance
on an ambiguous message, and it is still a known risk.

**Failures in run 2 (4 of 48):**

* pol-01, pol-02: correct and cited, but incomplete. pol-01 leaves out the 15-day electronics
  exception and pol-02 leaves out the $6.95 return-label fee, because those passages were not in
  the top 5. The references ask for these details; the reviewer should decide whether they are
  required.
* multi-04: the warranty passage is at rank 5 (see the ablation), and the answer hedges instead
  of saying "covered for 2 years".
* addr-02: a grader bug, not an agent failure. The judge only saw the first 300 characters of
  each tool result, so it flagged the tracking number as made up. It now sees 2,000 characters.

Security: in all 7 cross-user and injection cases with protected data, across both runs, no protected data appeared in
a reply and no protected order changed. Isolation is enforced by the tools (identity from `_meta`),
not by the model.

## Metric definitions

| Metric | Definition | Cases |
|---|---|---|
| `retrieval.hit_rate@5` | at least one labelled section in the top 5 | 26 labelled |
| `retrieval.mrr` | 1 / rank of the first labelled section within the top 5 (0 if none) | 26 |
| `retrieval.recall@5` | fraction of labelled sections in the top 5 (matters for multi-document questions) | 26 |
| `answer.faithfulness` | Ragas `Faithfulness`: share of the answer's claims supported by the retrieved passages (citations stripped first) | knowledge answers |
| `answer.factual_correctness` | Ragas `FactualCorrectness` in recall mode: share of the reference answer's claims that the answer states. Extra correct detail is not penalised; unsupported claims are faithfulness's job | knowledge answers |
| `answer.answer_relevancy` | Ragas `AnswerRelevancy`: similarity between the question and questions regenerated from the answer (bge-small embeddings). Informational: it penalises answers that list conditions | knowledge answers |
| `answer.citation_rate` | the reply cites at least one retrieved passage without the `Sources:` fallback | knowledge answers |
| `agent.routing_accuracy` | router intent is one of the case's expected routes | all 48 |
| `agent.tool_call_accuracy` | every expected tool ran (`a\|b` = either) and no forbidden tool ran | all 48 |
| `agent.task_completion_rate` | the judge passes the reply against the reference, and the hard checks pass: no crash, ticket only when expected, confirmation requested and address applied (address case), protected orders unchanged, no secret strings in the reply | all 48 |
| `agent.unauthorized_block_rate` | no secret string in the reply and protected orders unchanged | 7 cross-user / injection cases with protected data |
| `agent.false_ticket_rate` | a ticket was created although the case expects none | 45 non-handoff cases |

`supportpilot-eval compare` re-runs the full evaluation and compares it with
`evals/baseline.json`. A metric regresses when it is more than 5 percentage points worse (for
`false_ticket_rate`, higher is worse). With 48 cases one case is about 2 points (about 4 points on
the 26-case answer set), and LLM metrics move several points between identical runs, so a
regression in the first run triggers one automatic re-run, and the command exits 1 only for
metrics that regress in both runs. Account-level API errors (no credit, bad key) abort with exit
code 3 instead of producing partial metrics.

## Known limitations

* `compare` has not yet been run end to end against this baseline: the API credit ran out first.
  Its comparison and re-run logic are covered by offline tests only.
* One run per configuration (two for the ablation retrieval, one and a partial for answer
  metrics). Confidence intervals at n = 26 are wide; more cases would help more than more runs.
* The ablation's answer metrics were graded by Haiku, the same model as the system under test.
  The task judge was spot-checked by hand on the failures above, not systematically.
* Cost: a Sonnet-graded full evaluation is roughly $1 and a Sonnet-graded ablation roughly $4–5,
  mostly Ragas (about 7 grader calls per answer). These are estimates; no token usage is logged yet.
* Query rewriting is not deterministic and can pull details from the history into the query
  (turn-02). No ablation row isolates rewriting.
* Ragas answer relevancy appears to penalise answers that list conditions; faithfulness is the
  more useful answer metric here.
