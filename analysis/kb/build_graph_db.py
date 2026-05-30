"""Populate Neo4j with the VirtualHome KG (scenes + rules + failures).

Three layers:

1. **Scene layer**: one ``Scene`` per JSON file, ``Object`` nodes for
   each VirtualHome entity, and typed ``RELATION`` edges between them.
2. **Rule layer**: schema-level ``Action``/``Property``/``Precondition``/
   ``Effect`` nodes derived from ``analysis.precondition_kg.RULES``.
3. **Failure layer**: ``FailureCase`` nodes loaded from
   ``output/diagnostics`` and linked to the scene + violated action.

Run once (after applying schema.cypher)::

    python -m analysis.kb.build_graph_db
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import (
    DIAGNOSTICS_DIR,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    SCENE_GRAPH_ROOT,
)

log = logging.getLogger("kb.build_graph_db")

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.cypher"


# ------------------------------------------------------------------ helpers
def _file_id(path: Path) -> str:
    name = path.stem
    return name[len("file"):] if name.startswith("file") else name


def _scene_id_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("TrimmedTestScene") and part.endswith("_graph"):
            return part[len("TrimmedTestScene"):-len("_graph")]
    return "?"


def _scene_files(root: Path) -> Iterable[Path]:
    for scene_dir in sorted(root.glob("TrimmedTestScene*_graph")):
        results = scene_dir / "results_intentions_march-13-18"
        if not results.is_dir():
            continue
        for f in sorted(results.glob("file*.json")):
            yield f


# ------------------------------------------------------------------ schema
def apply_schema(session) -> None:
    if not SCHEMA_PATH.is_file():
        log.warning("schema.cypher not found at %s", SCHEMA_PATH)
        return
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    for stmt in text.split(";"):
        s = stmt.strip()
        if not s or s.startswith("//"):
            continue
        session.run(s)
    log.info("schema applied")


# ------------------------------------------------------------------ scene layer
def load_scene(session, path: Path) -> None:
    blob = json.loads(path.read_text(encoding="utf-8"))
    init_graph = blob.get("init_graph") or {}
    nodes = init_graph.get("nodes") or []
    edges = init_graph.get("edges") or []
    file_id = _file_id(path)
    scene_id = _scene_id_from_path(path)

    session.run(
        "MERGE (s:Scene {file_id: $file_id}) SET s.scene_id = $scene_id",
        file_id=file_id,
        scene_id=scene_id,
    )

    object_rows = [
        {
            "file_id": file_id,
            "node_id": int(n["id"]),
            "class_name": str(n.get("class_name", "")),
            "category": str(n.get("category", "")),
            "properties": list(n.get("properties") or []),
            "states": list(n.get("states") or []),
        }
        for n in nodes
        if n.get("id") is not None
    ]
    if object_rows:
        session.run(
            """
            UNWIND $rows AS row
            MERGE (o:Object {file_id: row.file_id, node_id: row.node_id})
            SET o.class_name = row.class_name,
                o.category   = row.category,
                o.properties = row.properties,
                o.states     = row.states
            WITH o, row
            MATCH (s:Scene {file_id: row.file_id})
            MERGE (s)-[:CONTAINS]->(o)
            """,
            rows=object_rows,
        )

    edge_rows = [
        {
            "file_id": file_id,
            "from_id": int(e["from_id"]),
            "to_id": int(e["to_id"]),
            "rel": str(e.get("relation_type", "RELATED")),
        }
        for e in edges
        if e.get("from_id") is not None and e.get("to_id") is not None
    ]
    if edge_rows:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (a:Object {file_id: row.file_id, node_id: row.from_id})
            MATCH (b:Object {file_id: row.file_id, node_id: row.to_id})
            MERGE (a)-[r:RELATION {type: row.rel}]->(b)
            """,
            rows=edge_rows,
        )


def load_all_scenes(session, root: Path, limit: Optional[int] = None) -> int:
    count = 0
    for path in _scene_files(root):
        load_scene(session, path)
        count += 1
        if count % 50 == 0:
            log.info("scenes loaded: %d", count)
        if limit and count >= limit:
            break
    log.info("scenes total: %d", count)
    return count


# ------------------------------------------------------------------ rule layer
def load_rules(session) -> None:
    # Import here to avoid pulling analysis deps when only running scene-load.
    try:
        from analysis.precondition_kg import RULES
    except ModuleNotFoundError:
        from precondition_kg import RULES  # type: ignore[no-redef]

    rule_rows: List[Dict] = []
    prop_edges: List[Dict] = []
    pre_edges: List[Dict] = []
    eff_edges: List[Dict] = []

    # Map effects by hand to mirror _apply_effects in precondition_kg.
    effect_map: Dict[str, List[str]] = {
        "WALK": ["VISIT_ARG1"],
        "RUN": ["VISIT_ARG1"],
        "TURNTO": ["VISIT_ARG1"],
        "FIND": ["VISIT_ARG1"],
        "GRAB": ["HOLD_ARG1", "VISIT_ARG1"],
        "PUTBACK": ["RELEASE_ARG1", "VISIT_ARG2"],
        "PUTIN": ["RELEASE_ARG1", "VISIT_ARG2"],
        "OPEN": ["OPEN_ARG1", "VISIT_ARG1"],
        "CLOSE": ["CLOSE_ARG1"],
        "SWITCHON": ["SWITCH_ON_ARG1", "VISIT_ARG1"],
        "SWITCHOFF": ["SWITCH_OFF_ARG1"],
        "PLUGIN": ["PLUG_IN_ARG1", "VISIT_ARG1"],
        "PLUGOUT": ["PLUG_OUT_ARG1"],
        "SIT": ["SIT_ON_ARG1", "VISIT_ARG1"],
        "STANDUP": ["STAND_UP"],
    }

    for name, rule in RULES.items():
        rule_rows.append({"name": name, "arity": rule.arity})
        for prop in rule.required_props_arg1:
            prop_edges.append({"action": name, "prop": prop, "slot": 1})
        for prop in rule.required_props_arg2:
            prop_edges.append({"action": name, "prop": prop, "slot": 2})
        if rule.requires_walk_to_arg1:
            pre_edges.append({"action": name, "kind": "WALK_TO_ARG1"})
        if rule.requires_walk_to_arg2:
            pre_edges.append({"action": name, "kind": "WALK_TO_ARG2"})
        if rule.requires_hold_arg1:
            pre_edges.append({"action": name, "kind": "HOLD_ARG1"})
        if rule.requires_open_arg2:
            pre_edges.append({"action": name, "kind": "OPEN_ARG2"})
        for eff in effect_map.get(name, []):
            eff_edges.append({"action": name, "kind": eff})

    session.run(
        "UNWIND $rows AS r MERGE (a:Action {name: r.name}) SET a.arity = r.arity",
        rows=rule_rows,
    )
    if prop_edges:
        session.run(
            """
            UNWIND $rows AS r
            MERGE (p:Property {name: r.prop})
            WITH r, p
            MATCH (a:Action {name: r.action})
            MERGE (a)-[rel:REQUIRES_PROP]->(p)
            SET rel.slot = r.slot
            """,
            rows=prop_edges,
        )
    if pre_edges:
        session.run(
            """
            UNWIND $rows AS r
            MERGE (p:Precondition {kind: r.kind})
            WITH r, p
            MATCH (a:Action {name: r.action})
            MERGE (a)-[:REQUIRES_STATE]->(p)
            """,
            rows=pre_edges,
        )
    if eff_edges:
        session.run(
            """
            UNWIND $rows AS r
            MERGE (e:Effect {kind: r.kind})
            WITH r, e
            MATCH (a:Action {name: r.action})
            MERGE (a)-[:PRODUCES]->(e)
            """,
            rows=eff_edges,
        )
    log.info("rules loaded: %d", len(rule_rows))


# ------------------------------------------------------------------ failure layer
def load_failures(session, diag_dir: Path) -> int:
    sources = [
        diag_dir / "goal_correct_but_action_fail_top_cases.json",
        diag_dir / "kg_planning_agent_report.json",
    ]
    count = 0
    for path in sources:
        if not path.is_file():
            continue
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("skip %s: %s", path, exc)
            continue
        records: List[dict] = []
        if isinstance(blob, list):
            records = [r for r in blob if isinstance(r, dict)]
        elif isinstance(blob, dict):
            for key in ("rows", "cases", "items", "records"):
                v = blob.get(key)
                if isinstance(v, list):
                    records = [r for r in v if isinstance(r, dict)]
                    break

        rows: List[Dict] = []
        for i, rec in enumerate(records):
            file_id = str(rec.get("file_id") or rec.get("identifier") or "")
            model = str(rec.get("model") or "")
            failure_type = str(rec.get("failure_type") or rec.get("code") or "unknown")
            task = str(rec.get("task") or rec.get("task_name") or "")
            action = str(rec.get("action") or rec.get("violated_action") or "")
            raw = str(rec.get("raw_failure_text") or rec.get("detail") or "")[:1000]
            uid = f"{path.stem}:{i}:{file_id}:{model}"
            rows.append({
                "uid": uid,
                "file_id": file_id,
                "model": model,
                "failure_type": failure_type,
                "task": task,
                "action": action,
                "raw": raw,
                "source": path.name,
            })
        if not rows:
            continue
        session.run(
            """
            UNWIND $rows AS r
            MERGE (f:FailureCase {uid: r.uid})
            SET f.file_id = r.file_id,
                f.model = r.model,
                f.failure_type = r.failure_type,
                f.task = r.task,
                f.raw = r.raw,
                f.source = r.source
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
        count += len(rows)
    log.info("failure cases loaded: %d", count)
    return count


# ------------------------------------------------------------------ task templates
def load_task_templates(session) -> int:
    """Load TaskTemplate nodes + STEP_OF edges into Neo4j (idempotent)."""
    try:
        from analysis.kb.task_templates import TASK_TEMPLATES
    except ModuleNotFoundError:
        from kb.task_templates import TASK_TEMPLATES  # type: ignore[no-redef]

    template_rows = []
    step_rows = []
    for tmpl in TASK_TEMPLATES.values():
        template_rows.append({
            "name": tmpl.name,
            "category": tmpl.category,
            "description": tmpl.description,
            "key_objects": tmpl.key_objects,
        })
        for i, step in enumerate(tmpl.steps):
            step_rows.append({
                "template": tmpl.name,
                "step_index": i,
                "action": step.action,
                "arg1_class": step.arg1_class or "",
                "arg2_class": step.arg2_class or "",
                "note": step.note,
            })

    session.run(
        """
        UNWIND $rows AS r
        MERGE (t:TaskTemplate {name: r.name})
        SET t.category = r.category,
            t.description = r.description,
            t.key_objects = r.key_objects
        """,
        rows=template_rows,
    )
    if step_rows:
        session.run(
            """
            UNWIND $rows AS r
            MATCH (t:TaskTemplate {name: r.template})
            MERGE (a:Action {name: r.action})
            MERGE (t)-[rel:STEP_OF]->(a)
            SET rel.step_index = r.step_index,
                rel.arg1_class = r.arg1_class,
                rel.arg2_class = r.arg2_class,
                rel.note = r.note
            """,
            rows=step_rows,
        )
    log.info("task templates loaded: %d (%d steps)", len(template_rows), len(step_rows))
    return len(template_rows)
def load_sim_rules(
    driver,
    output_dir: Optional[str] = None,
    min_count: int = 3,
) -> int:
    """Extract rules from error_info.json files and write them to Neo4j."""
    try:
        from .simulation_rule_extraction import extract_rules_from_logs, persist_sim_rules
    except ImportError:
        from simulation_rule_extraction import extract_rules_from_logs, persist_sim_rules

    out_dir = output_dir or str(REPO_ROOT / "output")
    rules = extract_rules_from_logs(out_dir, min_count=min_count)
    if not rules:
        log.warning("load_sim_rules: no rules extracted (min_count=%d)", min_count)
        return 0
    n = persist_sim_rules(rules, driver, database=NEO4J_DATABASE)
    return n


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", default=str(SCENE_GRAPH_ROOT))
    parser.add_argument("--diagnostics", default=str(DIAGNOSTICS_DIR))
    parser.add_argument("--output-dir", default=None,
                        help="root output dir for sim log extraction (default: output/)")
    parser.add_argument("--limit-scenes", type=int, default=None,
                        help="cap scene count (debug)")
    parser.add_argument("--skip-scenes", action="store_true")
    parser.add_argument("--skip-rules", action="store_true")
    parser.add_argument("--skip-failures", action="store_true")
    parser.add_argument("--skip-templates", action="store_true")
    parser.add_argument("--skip-sim-rules", action="store_true")
    parser.add_argument("--skip-schema", action="store_true")
    parser.add_argument("--sim-min-count", type=int, default=3,
                        help="min occurrences for simulation log rule")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            if not args.skip_schema:
                apply_schema(session)
            if not args.skip_scenes:
                load_all_scenes(session, Path(args.scene_root), args.limit_scenes)
            if not args.skip_rules:
                load_rules(session)
            if not args.skip_failures:
                load_failures(session, Path(args.diagnostics))
            if not args.skip_templates:
                load_task_templates(session)
        if not args.skip_sim_rules:
            load_sim_rules(driver, output_dir=args.output_dir, min_count=args.sim_min_count)
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
