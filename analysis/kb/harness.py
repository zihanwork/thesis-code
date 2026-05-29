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

Status: skeleton only. ``run_iteration()`` currently invokes the
existing ``scripts/run_action_sequencing_eval.sh`` pipeline and parses
its output; replace with a direct Python call once Agent v2 is in
place. ``ingest_bad_cases()`` is fully wired to Neo4j + Chroma and is
safe to run in isolation.
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
) -> RunReport:
    """Drive one full generate -> evaluate pass.

    The current implementation shells out to the existing pipeline. Once
    Agent v2 is in place this should be replaced with a direct Python
    call so we can pass the persistent retriever / verifier explicitly.
    """
    output_root = Path(output_root or REPO_ROOT / "output" / "harness" / f"iter{iteration_id}")
    output_root.mkdir(parents=True, exist_ok=True)

    log.info("[harness] iter=%d model=%s variant=%s -> %s",
             iteration_id, model, variant, output_root)
    cmd = [
        "bash", str(REPO_ROOT / "scripts" / "run_action_sequencing_eval.sh"),
        "--model", model,
        "--variant", variant,
        "--out", str(output_root),
    ]
    if task_ids:
        cmd.extend(["--task-ids", ",".join(task_ids)])

    started = time.time()
    # NOTE: current shell script does not yet accept these flags; this is
    # the target interface for the v2 wiring. Until then, callers pass an
    # already-produced summary.json path via ``parse_summary``.
    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        log.warning("[harness] external runner failed (%s); "
                    "expected during skeleton phase", exc)

    summary_path = output_root / "summary.json"
    rows = parse_summary(summary_path) if summary_path.is_file() else []
    return RunReport(
        iteration_id=iteration_id,
        model=model,
        variant=variant,
        summary_path=summary_path,
        rows=rows,
        started_at=started,
        finished_at=time.time(),
    )


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
) -> List[RunReport]:
    """Drive the full evaluate -> ingest -> evaluate cycle until convergence."""
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
