"""Iterative evaluation harness for the knowledge-grounded planner.

Closes the loop between the planning agent and the persistent KG/RAG:

    iteration N:
        run_iteration() -> RunReport (per-task pass/fail summary)
        collect_bad_cases(report) -> List[BadCase]
        ingest_bad_cases(bad_cases, iteration_id=N)
            -> Neo4j (:FailureCase {iteration_id, ...})
            -> Chroma failure_cases collection (upsert)

    iteration N+1:
        agent retrieves these freshly added failures when it sees similar
        tasks, and injects them as few-shot exemplars during repair.

The original benchmark gold sequences are never modified; only LLM
drafts and their violation codes flow into the knowledge base. This
keeps the EAI evaluator's ground truth clean.

Status: runnable orchestration layer. ``run_iteration()`` drives
generation, optional persistent-KB repair, evaluation, and summary
parsing; ``run_loop()`` repeats this process while ingesting failures
and optionally inducing or extracting derived rules. Live runs still
require the EAI evaluator and configured KB/LLM services.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    FAILURE_CASES_COLLECTION,
    MODELS_CACHE,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    REPO_ROOT,
)

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ data types
@dataclass(frozen=True)
class BadCase:
    """One LLM-generated failure to be re-fed into the knowledge base."""

    file_id: str          # VirtualHome script id, e.g. "11_1"
    task: str             # natural-language task description
    model: str            # generator model snapshot, e.g. "deepseek-v4-flash"
    variant: str          # prompt variant, e.g. "plan_then_ground"
    failure_type: str     # taxonomy label or violation code
    draft: str            # LLM-generated action sequence (concatenated JSON)
    raw_text: str         # truncated evaluator detail / violation summary
    violated_action: str = ""

    def uid(self, iteration_id: int) -> str:
        return f"iter{iteration_id}:{self.model}:{self.variant}:{self.file_id}"


@dataclass
class RunReport:
    """Per-task summary produced by one harness iteration."""

    iteration_id: int
    model: str
    variant: str
    summary_path: Path
    rows: List[Dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    @property
    def task_success_rate(self) -> float:
        if not self.rows:
            return 0.0
        passed = sum(1 for r in self.rows if r.get("task_success"))
        return passed / len(self.rows)

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.rows if not r.get("task_success"))


@dataclass
class IterationDiff:
    """How iteration B differs from iteration A."""

    fixed: List[str]            # file_ids that failed in A but passed in B
    regressed: List[str]        # file_ids that passed in A but failed in B
    success_delta: float        # B.task_success_rate - A.task_success_rate


# ------------------------------------------------------------------ run a single iteration
def run_iteration(
    model: str,
    variant: str,
    iteration_id: int,
    task_ids: Optional[Sequence[str]] = None,
    output_root: Optional[Path] = None,
    llm_response_path: Optional[Path] = None,
) -> RunReport:
    """Drive one full generate -> (optional build_kg_planning_agent) -> evaluate pass.

    Pipeline:
    1. If ``llm_response_path`` is None, call ``generate_outputs.py`` to
       produce a draft ``*_outputs.json`` for (model, variant).
    2. Optionally run ``build_kg_planning_agent.py`` to apply persistent-KB
       grounding + conservative local repair (when KB_BACKEND=persistent is
       set and Neo4j / Chroma are available).
    3. Shell out to ``run_action_sequencing_eval.sh`` (via
       LLM_RESPONSE_PATH env var) to run the EAI evaluator.
    4. Locate and parse ``summary.json`` → ``RunReport``.

    Soft-fail: any step that errors is logged as a warning; the report will
    have empty rows, allowing the caller to decide whether to abort the loop.
    """
    output_root = Path(output_root or REPO_ROOT / "output" / "harness" / f"iter{iteration_id}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log.info("[harness] iter=%d model=%s variant=%s -> %s",
             iteration_id, model, variant, output_root)

    # ---- step 1: generate drafts if not supplied
    if llm_response_path is None:
        llm_response_path = _generate_drafts(
            model=model,
            variant=variant,
            output_root=output_root,
            task_ids=task_ids,
        )

    # ---- step 2: optionally run knowledge-grounded agent repair
    import os
    if os.environ.get("KB_BACKEND", "default").lower() == "persistent" \
            and llm_response_path is not None \
            and llm_response_path.is_file():
        llm_response_path = _apply_kg_agent(
            input_path=llm_response_path,
            output_root=output_root,
            iteration_id=iteration_id,
        )

    # ---- step 3: run EAI evaluator
    eval_output_dir = output_root / "eval"
    _run_eai_eval(
        llm_response_path=llm_response_path,
        output_dir=eval_output_dir,
    )

    # ---- step 4: locate summary.json (EAI writes it under
    #  <output_dir>/virtualhome/evaluate_results/action_sequencing/<model>/summary.json)
    summary_path = _find_summary(eval_output_dir, model)
    rows = parse_summary(summary_path) if (summary_path and summary_path.is_file()) else []
    if not rows:
        log.warning("[harness] iter=%d: no summary rows found under %s",
                    iteration_id, eval_output_dir)

    return RunReport(
        iteration_id=iteration_id,
        model=model,
        variant=variant,
        summary_path=summary_path or output_root / "summary_missing.json",
        rows=rows,
        started_at=started,
        finished_at=time.time(),
    )


# ---- internal pipeline helpers
def _generate_drafts(
    model: str,
    variant: str,
    output_root: Path,
    task_ids: Optional[Sequence[str]],
    provider: str = "dry_run",
    api_model: str = "stub",
) -> Optional[Path]:
    """Call generate_outputs.py; return path to produced *_outputs.json or None.

    Signature of generate_outputs.py::

        --provider {openai,anthropic,gemini,openai_compatible,dry_run}
        --api-model API_MODEL        (provider-specific model id)
        --model-name MODEL_NAME      (friendly id used in output filenames)
        --variant VARIANT
        --eval-type action_sequencing
        --helm-prompt HELM_PROMPT    (path to helm_prompt.json)
        --out-dir OUT_DIR            (writes <model-name>_<variant>_outputs.json here)

    Environment variables KB_PROVIDER, KB_API_MODEL, KB_API_KEY_ENV, and
    KB_BASE_URL override the provider defaults so callers don't need to
    change this function.
    """
    import os
    provider = os.environ.get("KB_PROVIDER", provider)
    api_model = os.environ.get("KB_API_MODEL", api_model)
    base_url = os.environ.get("KB_BASE_URL")
    if provider == "openai_compatible" and not base_url:
        log.warning("[harness] KB_BASE_URL is required when KB_PROVIDER=openai_compatible")
        return None

    script = REPO_ROOT / "analysis" / "generate_outputs.py"
    helm_prompt = (
        REPO_ROOT / "output" / "improvement_run" / "prompts"
        / "virtualhome" / "generate_prompts" / "action_sequencing" / "helm_prompt.json"
    )
    if not helm_prompt.is_file():
        log.warning("[harness] helm_prompt.json not found at %s", helm_prompt)
        return None

    drafts_dir = output_root / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    out_file = drafts_dir / f"{model}_{variant}_outputs.json"

    cmd = [
        "python3", str(script),
        "--provider", provider,
        "--api-model", api_model,
        "--model-name", model,
        "--variant", variant,
        "--eval-type", "action_sequencing",
        "--helm-prompt", str(helm_prompt),
        "--out-dir", str(drafts_dir),
    ]
    if "KB_API_KEY_ENV" in os.environ:
        cmd.extend(["--api-key-env", os.environ["KB_API_KEY_ENV"]])
    if base_url:
        cmd.extend(["--base-url", base_url])
    if task_ids:
        cmd.extend(["--max-prompts", str(len(task_ids))])

    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        # generate_outputs writes <model-name>_<variant>_outputs.json
        return out_file if out_file.is_file() else None
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        log.warning("[harness] generate_drafts failed: %s", exc)
        return None


def _apply_kg_agent(
    input_path: Path,
    output_root: Path,
    iteration_id: int,
) -> Path:
    """Run build_kg_planning_agent.py; return path to repaired outputs."""
    script = REPO_ROOT / "analysis" / "build_kg_planning_agent.py"
    repaired = output_root / f"iter{iteration_id}_repaired_outputs.json"
    report = output_root / f"iter{iteration_id}_kg_agent_report.json"
    try:
        import os
        env = {**os.environ, "KB_BACKEND": "persistent"}
        subprocess.run(
            ["python3", str(script),
             "--input", str(input_path),
             "--output", str(repaired),
             "--report", str(report)],
            check=True,
            cwd=str(REPO_ROOT),
            env=env,
        )
        return repaired if repaired.is_file() else input_path
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        log.warning("[harness] kg_agent repair failed: %s; using unrepaired draft", exc)
        return input_path


def _run_eai_eval(
    llm_response_path: Optional[Path],
    output_dir: Path,
) -> None:
    """Shell out to run_action_sequencing_eval.sh with LLM_RESPONSE_PATH set."""
    script = REPO_ROOT / "scripts" / "run_action_sequencing_eval.sh"
    if not script.is_file():
        log.warning("[harness] eval script not found: %s", script)
        return
    import os
    import shutil

    response_root = output_dir / "llm_response"
    if llm_response_path and llm_response_path.is_file():
        raw_root = output_dir / "llm_response_raw"
        raw_dir = raw_root / "virtualhome" / "action_sequencing"
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(llm_response_path, raw_dir / llm_response_path.name)

        norm_root = response_root
        norm_dir = norm_root / "virtualhome" / "action_sequencing"
        norm_dir.mkdir(parents=True, exist_ok=True)
        normalizer = REPO_ROOT / "analysis" / "normalize_action_outputs.py"
        if normalizer.is_file():
            try:
                subprocess.run(
                    ["python3", str(normalizer),
                     "--input-dir", str(raw_dir),
                     "--output-dir", str(norm_dir)],
                    check=True,
                    cwd=str(REPO_ROOT),
                )
                llm_response_arg = norm_root
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                log.warning("[harness] normalise outputs failed: %s; using raw outputs", exc)
                llm_response_arg = raw_root
        else:
            log.warning("[harness] normaliser not found: %s; using raw outputs", normalizer)
            llm_response_arg = raw_root
    else:
        llm_response_arg = llm_response_path

    env = {**os.environ, "LLM_RESPONSE_PATH": str(llm_response_arg or "")}
    try:
        subprocess.run(
            ["bash", str(script), "virtualhome", "none", str(output_dir)],
            check=True,
            cwd=str(REPO_ROOT),
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        log.warning("[harness] eai eval returned non-zero: %s", exc)


def _find_summary(eval_output_dir: Path, model: str) -> Optional[Path]:
    """Locate the summary.json written by EAI under the eval output tree."""
    # Canonical path: <eval_output_dir>/virtualhome/evaluate_results/action_sequencing/<model>/summary.json
    canonical = (
        eval_output_dir / "virtualhome" / "evaluate_results"
        / "action_sequencing" / model / "summary.json"
    )
    if canonical.is_file():
        return canonical
    # Fallback: glob
    for p in eval_output_dir.rglob("summary.json"):
        return p
    return None


def parse_summary(summary_path: Path) -> List[Dict[str, Any]]:
    """Normalise EAI summary.json into a flat per-task row list."""
    if not summary_path.is_file():
        return []
    blob = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    if isinstance(blob, list):
        rows = [r for r in blob if isinstance(r, dict)]
    elif isinstance(blob, dict):
        for key in ("rows", "tasks", "results", "items"):
            v = blob.get(key)
            if isinstance(v, list):
                rows = [r for r in v if isinstance(r, dict)]
                break
    if rows:
        return rows

    error_info_path = summary_path.parent / "error_info.json"
    if not error_info_path.is_file():
        return []
    try:
        error_info = json.loads(error_info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.warning("[harness] failed to parse error_info.json at %s: %s", error_info_path, exc)
        return []
    if not isinstance(error_info, dict):
        return []

    for file_id, info in sorted(error_info.items()):
        if not isinstance(info, dict):
            continue
        error_type = info.get("error_type")
        executable = bool(info.get("executable"))
        actions = info.get("actions", [])
        if isinstance(actions, str):
            draft = actions
        else:
            draft = json.dumps(actions, ensure_ascii=False)
        raw = json.dumps(info, ensure_ascii=False)
        rows.append({
            "file_id": str(file_id),
            "identifier": str(file_id),
            "task_success": executable and error_type is None,
            "executable": executable,
            "failure_type": str(error_type or ""),
            "error_action": str(info.get("error_action") or ""),
            "violated_action": str(info.get("error_action") or ""),
            "actions": actions,
            "draft": draft,
            "raw_failure_text": raw,
            "detail": raw,
        })
    return rows


# ------------------------------------------------------------------ extract bad cases
def collect_bad_cases(
    report: RunReport,
    classify: Optional[Any] = None,
) -> List[BadCase]:
    """Filter the run report down to the failures we want to re-ingest.

    ``classify`` is an optional callable ``(row) -> failure_type``; when
    not supplied we fall back to the row's ``failure_type`` field or the
    first violation code emitted by the in-process verifier.
    """
    bad: List[BadCase] = []
    for row in report.rows:
        if row.get("task_success"):
            continue
        file_id = str(row.get("file_id") or row.get("identifier") or "")
        if not file_id:
            continue
        failure_type = (
            classify(row) if classify
            else str(row.get("failure_type") or row.get("violation_code") or "unknown")
        )
        bad.append(BadCase(
            file_id=file_id,
            task=str(row.get("task") or row.get("task_name") or ""),
            model=report.model,
            variant=report.variant,
            failure_type=failure_type,
            draft=str(row.get("draft") or row.get("action_sequence") or ""),
            raw_text=str(row.get("raw_failure_text") or row.get("detail") or "")[:1000],
            violated_action=str(row.get("violated_action") or row.get("action") or ""),
        ))
    log.info("[harness] iter=%d collected %d bad cases", report.iteration_id, len(bad))
    return bad


# ------------------------------------------------------------------ ingest into KB
def ingest_bad_cases(
    bad_cases: Sequence[BadCase],
    iteration_id: int,
    *,
    skip_neo4j: bool = False,
    skip_chroma: bool = False,
) -> Dict[str, int]:
    """Upsert bad cases into Neo4j and the Chroma failure_cases collection.

    Idempotent: keyed on ``BadCase.uid(iteration_id)`` for both backends.
    """
    counts = {"neo4j": 0, "chroma": 0}
    if not bad_cases:
        return counts
    if not skip_neo4j:
        counts["neo4j"] = _ingest_neo4j(bad_cases, iteration_id)
    if not skip_chroma:
        counts["chroma"] = _ingest_chroma(bad_cases, iteration_id)
    log.info("[harness] iter=%d ingested neo4j=%d chroma=%d",
             iteration_id, counts["neo4j"], counts["chroma"])
    return counts


def _ingest_neo4j(bad_cases: Sequence[BadCase], iteration_id: int) -> int:
    from neo4j import GraphDatabase

    rows = [
        {
            "uid": bc.uid(iteration_id),
            "iteration_id": iteration_id,
            "file_id": bc.file_id,
            "model": bc.model,
            "variant": bc.variant,
            "failure_type": bc.failure_type,
            "task": bc.task,
            "raw": bc.raw_text,
            "draft": bc.draft[:2000],
            "action": bc.violated_action,
        }
        for bc in bad_cases
    ]
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            session.run(
                """
                UNWIND $rows AS r
                MERGE (f:FailureCase {uid: r.uid})
                SET f.iteration_id = r.iteration_id,
                    f.file_id      = r.file_id,
                    f.model        = r.model,
                    f.variant      = r.variant,
                    f.failure_type = r.failure_type,
                    f.task         = r.task,
                    f.raw          = r.raw,
                    f.draft        = r.draft,
                    f.source       = 'harness'
                WITH f, r
                OPTIONAL MATCH (s:Scene {file_id: r.file_id})
                FOREACH (_ IN CASE WHEN s IS NULL THEN [] ELSE [1] END |
                  MERGE (f)-[:OCCURRED_IN]->(s)
                )
                WITH f, r WHERE r.action <> ''
                MERGE (a:Action {name: r.action})
                MERGE (f)-[:VIOLATES]->(a)
                """,
                rows=rows,
            )
    finally:
        driver.close()
    return len(rows)


def _ingest_chroma(bad_cases: Sequence[BadCase], iteration_id: int) -> int:
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    coll = client.get_or_create_collection(
        name=FAILURE_CASES_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    embedder = SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(MODELS_CACHE))

    ids = [bc.uid(iteration_id) for bc in bad_cases]
    metas = [
        {
            "iteration_id": iteration_id,
            "file_id": bc.file_id,
            "model": bc.model,
            "variant": bc.variant,
            "failure_type": bc.failure_type,
            "task": bc.task,
            "violated_action": bc.violated_action,
            "source": "harness",
        }
        for bc in bad_cases
    ]
    docs = [
        f"task={bc.task} failure_type={bc.failure_type} "
        f"action={bc.violated_action} :: {bc.raw_text}"[:2000]
        for bc in bad_cases
    ]
    embeddings = embedder.encode(docs, normalize_embeddings=True).tolist()
    coll.upsert(ids=ids, metadatas=metas, documents=docs, embeddings=embeddings)
    return len(ids)


# ------------------------------------------------------------------ cross-iteration analysis
def diff_iterations(a: RunReport, b: RunReport) -> IterationDiff:
    """Compute pass/fail flips between two reports keyed by file_id."""
    a_status = {str(r.get("file_id")): bool(r.get("task_success")) for r in a.rows}
    b_status = {str(r.get("file_id")): bool(r.get("task_success")) for r in b.rows}
    common = set(a_status) & set(b_status)
    fixed = sorted(fid for fid in common if not a_status[fid] and b_status[fid])
    regressed = sorted(fid for fid in common if a_status[fid] and not b_status[fid])
    return IterationDiff(
        fixed=fixed,
        regressed=regressed,
        success_delta=b.task_success_rate - a.task_success_rate,
    )


def has_converged(
    history: Sequence[RunReport],
    *,
    window: int = 2,
    epsilon: float = 0.005,
) -> bool:
    """True when the last ``window`` iterations all gained < epsilon."""
    if len(history) < window + 1:
        return False
    recent = history[-(window + 1):]
    deltas = [
        recent[i + 1].task_success_rate - recent[i].task_success_rate
        for i in range(window)
    ]
    return all(d < epsilon for d in deltas)


# ------------------------------------------------------------------ full loop
def run_loop(
    model: str,
    variant: str,
    *,
    max_iterations: int = 5,
    task_ids: Optional[Sequence[str]] = None,
    llm_client: Any = None,
    neo4j_driver: Any = None,
) -> List[RunReport]:
    """Drive the full evaluate -> ingest -> rule-induction -> evaluate cycle.

    Each iteration:
    1. run_iteration (generate + eval)
    2. collect_bad_cases → ingest to Neo4j + Chroma
    3. If llm_client provided → induce new DerivedRule nodes from bad cases
    4. If neo4j_driver provided → re-extract simulation log rules from new output
    5. Check convergence
    """
    history: List[RunReport] = []

    for n in range(1, max_iterations + 1):
        report = run_iteration(model, variant, iteration_id=n, task_ids=task_ids)
        history.append(report)
        bad = collect_bad_cases(report)
        ingest_bad_cases(bad, iteration_id=n)
        log.info(
            "[harness] iter=%d success=%.4f failed=%d",
            n, report.task_success_rate, report.num_failed,
        )

        # Direction 2: LLM-based rule induction from bad cases
        if llm_client is not None and bad and neo4j_driver is not None:
            try:
                from .rule_induction import induce_rules_from_bad_cases, persist_derived_rules
            except ImportError:
                from rule_induction import induce_rules_from_bad_cases, persist_derived_rules
            bad_dicts = [
                {
                    "task_id": bc.file_id,
                    "failure_detail": f"{bc.failure_type}: {bc.raw_text}",
                    "failed_action": bc.violated_action,
                    "uid": bc.uid(n),
                }
                for bc in bad
            ]
            new_rules = induce_rules_from_bad_cases(bad_dicts, llm_client, iteration_id=n)
            if new_rules:
                persist_derived_rules(new_rules, neo4j_driver)
                log.info("[harness] iter=%d inducted %d new rules", n, len(new_rules))

        # Direction 3: re-extract simulation log rules (picks up new error_info.json)
        if neo4j_driver is not None:
            try:
                from .simulation_rule_extraction import extract_rules_from_logs, persist_sim_rules
            except ImportError:
                from simulation_rule_extraction import extract_rules_from_logs, persist_sim_rules
            output_dir = str(REPO_ROOT / "output" / "harness" / f"iter{n}")
            sim_rules = extract_rules_from_logs(output_dir, min_count=1)
            if sim_rules:
                persist_sim_rules(sim_rules, neo4j_driver)
                log.info("[harness] iter=%d added %d sim rules", n, len(sim_rules))

        if has_converged(history):
            log.info("[harness] converged at iter=%d", n)
            break
    return history


__all__ = [
    "BadCase",
    "RunReport",
    "IterationDiff",
    "run_iteration",
    "parse_summary",
    "collect_bad_cases",
    "ingest_bad_cases",
    "diff_iterations",
    "has_converged",
    "run_loop",
]
