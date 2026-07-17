# Eleven-stage completion audit

Audit date: 2026-07-17.

This audit treats the repository, immutable run manifests, frozen hashes,
official evaluator output, and test execution as evidence. It does not treat a
plan or an unexecuted configuration as proof by itself.

| Stage | Status | Authoritative evidence |
| --- | --- | --- |
| 1. Freeze and repair the current version | Complete | Main repository history starts at baseline commit `e3930f7`; framework and final protocol are tagged; final runs record clean commit `50095ae`; EAI and iGibson submodule commits are pinned; `uv.lock` and CI exist; the complete 47-test suite passes; tracked-file scanning finds no `/Users/wuzihan` or `/Users/hanson` data path. Immutable/non-overwriting run behavior is covered by tests and manifests. |
| 2. Freeze the data protocol | Complete | `heldout_virtualhome_119.jsonl` contains 119 tasks, has record SHA-256 `4a01bdb4...`, task-ID SHA-256 `036ed8d9...`, and zero overlap with the 202-task development set. The freezer and final-protocol verifier cover the invariants. |
| 3. Complete the RAG method | Complete for the reported method | Lexical, BM25, and field-structured retrievers; four query profiles; and top-1/3/5 prompting are implemented and tested. Shared-slot inflation was removed. The reported fixed P1 is lexical, full-field, top-1 and is compared task-by-task with its no-RAG engineered-prompt control on three models. The thesis does not claim that every supported retriever/top-k/profile cell was empirically exhausted. |
| 4. Run cost-controlled pilots | Complete | Historical 20/50-task pilots, the 20-task three-model development pilot, output-budget canary, and 20-task isolated recovery pilot all have immutable runs, telemetry, and selection records. No held-out task was used for method selection. |
| 5. Add and contrast Memory | Complete | The 357-entry symbolic-teacher memory is frozen with hash `ec5ad98b...`; development and held-out comparisons isolate Memory from Reflection and PDDL. Memory tied Reflection on final paired outcomes and is not claimed as an incremental gain. |
| 6. Add statistics and cost accounting | Complete | Every final child run has `analysis.json` with Wilson intervals, exact McNemar tests, paired bootstrap intervals, difficulty/task-family/failure strata, calls/tokens/latency, repairs, and symbolic search. Final actual telemetry totals 1,230 calls and 2,432,206 tokens. Monetary cost remains explicitly unavailable because One API pricing was not configured. |
| 7. Review benchmarks and literature | Complete | `benchmark_literature_review.md` and `benchmark_evidence.json` position the method against EAI, AxisTilted2, SingaX, CtrlAct, and Re² Agent and enforce the custom-versus-official comparison boundary. |
| 8. Connect official evaluation | Complete for the available action-sequencing scope | The pinned official VirtualHome evaluator runs end to end. Strict conversion of the selected final method covered 83/119 tasks; the compatible subset produced 83/83 executable and 43/83 official task success. This is explicitly not a leaderboard score. Official BEHAVIOR remains unrun without separately licensed iGibson assets, and the project does not claim a complete eight-slot challenge submission. |
| 9. Add a safety experiment | Complete for a controlled mechanism claim | A frozen 30-task set covers explicit hazards, safe near misses, missing steps, invalid operations, and unrecoverable errors. H0/H1/H2-Local report detection, miss, false interception, safe completion, and recovery. The evidence is deliberately not described as open-world LLM safety understanding. |
| 10. Expand model coverage | Complete | One API inventory and realistic canaries were checked. GLM-5-Turbo was promoted as a third family and run on the held-out P0/P1 cells; DeepSeek-V4-Pro was excluded after truncation. Qwen/Llama were unavailable through the account. Fine-tuning was explicitly deferred as optional. |
| 11. Run and audit the frozen final experiment | Complete | Four one-shot matrices completed under `final-protocol-v1`: Planning 714 records, Recovery 952, GLM generalization 238, and Symbolic 119, totalling 2,023. All frozen hashes still match. Results, paired cross-run comparisons, failure analysis, telemetry, official diagnostic, and artifact hashes are recorded in `final_results.md` and the two final evidence JSON files. |

## Completion boundaries

The eleven-stage thesis implementation is complete within its frozen scope.
This statement does not expand the evidence into claims the project did not
test. In particular:

- the final local held-out set is VirtualHome-only;
- no official BEHAVIOR simulator score is available without licensed assets;
- the official result is an 83-task compatibility-selected action-sequencing
  diagnostic, not a challenge submission;
- P2 and H2-PDDL are symbolic references, not LLM contributions;
- Memory did not outperform plain Reflection;
- fine-tuning and a full empirical sweep of every supported RAG setting remain
  optional follow-up experiments rather than hidden prerequisites.

The final quantitative source of truth is
`docs/final_results_evidence.json`; the official-evaluator source of truth is
`docs/final_official_virtualhome_evidence.json`.
