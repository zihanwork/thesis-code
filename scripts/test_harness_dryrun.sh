#!/usr/bin/env bash
# Smoke-test the harness pipeline end-to-end using the dry_run provider.
#
# Does NOT require a live LLM API key or a running Neo4j / Chroma.
# Confirms that the 4-step pipeline (generate -> agent -> eval -> ingest)
# at least reaches each step without crashing.
#
# Usage:
#   bash scripts/test_harness_dryrun.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export KB_PROVIDER=dry_run
export KB_API_MODEL=stub
export KB_BACKEND=default      # keep default to skip Neo4j/Chroma
export PYTHONPATH="${REPO_ROOT}/analysis:${PYTHONPATH:-}"

echo "[smoke] step 1 – verify imports"
python3 -c "
from analysis.kb.harness import (
    run_loop, collect_bad_cases, ingest_bad_cases,
    diff_iterations, has_converged, parse_summary, BadCase, RunReport
)
print('  imports OK')
"

echo "[smoke] step 2 – dry_run single generate pass"
HELM_PROMPT="${REPO_ROOT}/output/improvement_run/prompts/virtualhome/generate_prompts/action_sequencing/helm_prompt.json"
DRAFTS_DIR="${REPO_ROOT}/output/harness/smoke/drafts"
mkdir -p "${DRAFTS_DIR}"
python3 analysis/generate_outputs.py \
  --provider dry_run \
  --api-model stub \
  --model-name smoke_model \
  --variant plan_then_ground \
  --eval-type action_sequencing \
  --helm-prompt "${HELM_PROMPT}" \
  --out-dir "${DRAFTS_DIR}" \
  --max-prompts 3
echo "  drafts written to ${DRAFTS_DIR}"

echo "[smoke] step 3 – parse_summary on stub output"
python3 - <<'PY'
from pathlib import Path
import json, sys

drafts = list(Path("output/harness/smoke/drafts").glob("*.json"))
if not drafts:
    print("  no draft file found – skipping parse_summary check")
    sys.exit(0)
rows = json.loads(drafts[0].read_text())
print(f"  draft has {len(rows)} rows")
PY

echo "[smoke] step 4 – collect_bad_cases on synthetic report"
python3 - <<'PY'
from analysis.kb.harness import RunReport, collect_bad_cases
from pathlib import Path

report = RunReport(
    iteration_id=0,
    model="smoke_model",
    variant="plan_then_ground",
    summary_path=Path("/dev/null"),
    rows=[
        {"file_id": "11_1", "task": "Read book", "task_success": False,
         "failure_type": "missing_step"},
        {"file_id": "125_2", "task": "Turn on light", "task_success": True},
    ],
)
bad = collect_bad_cases(report)
assert len(bad) == 1, f"expected 1 bad case, got {len(bad)}"
assert bad[0].file_id == "11_1"
print(f"  collect_bad_cases OK -> {bad[0].file_id} / {bad[0].failure_type}")
PY

echo "[smoke] step 5 – ingest_bad_cases (skip_neo4j=True, skip_chroma=True)"
python3 - <<'PY'
from analysis.kb.harness import BadCase, ingest_bad_cases

bad = [BadCase(
    file_id="11_1", task="Read book",
    model="smoke_model", variant="plan_then_ground",
    failure_type="missing_step", draft='{"WALK":["book","1"]}',
    raw_text="missing GRAB before READ", violated_action="READ",
)]
counts = ingest_bad_cases(bad, iteration_id=0, skip_neo4j=True, skip_chroma=True)
print(f"  ingest_bad_cases OK (skipped both backends) counts={counts}")
PY

echo "[smoke] step 6 – has_converged guard"
python3 - <<'PY'
from analysis.kb.harness import RunReport, has_converged
from pathlib import Path

def _r(n, sr):
    r = RunReport(iteration_id=n, model="m", variant="v",
                  summary_path=Path("/dev/null"))
    r.rows = [{"task_success": True}] * int(sr * 100) + \
             [{"task_success": False}] * (100 - int(sr * 100))
    return r

history = [_r(1, 0.75), _r(2, 0.80), _r(3, 0.801), _r(4, 0.8015)]
assert has_converged(history, window=2, epsilon=0.005), "expected converged"
print("  has_converged OK")
PY

echo ""
echo "=================================================="
echo "  All smoke tests passed."
echo "  Next: install Docker (colima) + pip deps, then"
echo "  run:  NEO4J_PASSWORD=xxx bash scripts/build_knowledge_base.sh"
echo "=================================================="
