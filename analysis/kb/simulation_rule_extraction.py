"""Direction 3: extract constraint rules from VirtualHome evaluation logs.

Scans all ``error_info.json`` files produced by the eai-eval pipeline, aggregates
failure patterns by (error_type, action_verb, object_class), and converts
high-frequency patterns into DerivedRule nodes in Neo4j.

Usage (standalone)::

    python -m analysis.kb.simulation_rule_extraction \
        --output-dir output/ \
        --min-count 3

Or call from Python::

    from analysis.kb.simulation_rule_extraction import extract_rules_from_logs, persist_sim_rules
    rules = extract_rules_from_logs("output/", min_count=3)
    persist_sim_rules(rules, neo4j_driver)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SimLogRule:
    """A constraint rule extracted from simulation error_info.json files."""
    uid: str
    action: str            # VirtualHome action verb, e.g. "LOOKAT"
    object_class: str      # Object class involved, e.g. "television"
    error_type: str        # Original error type: missing_step / affordance_error / etc.
    condition: str         # Human-readable constraint
    cypher_hint: str       # Formal hint for Neo4j query
    count: int             # Number of evidence cases
    evidence_ids: List[str]  # file_ids that triggered this rule
    source: str = "simulation_log"
    confidence: float = 1.0

    def to_neo4j_row(self) -> Dict:
        return {
            "uid": self.uid,
            "action": self.action,
            "condition": self.condition,
            "cypher_hint": self.cypher_hint,
            "evidence_uids": self.evidence_ids[:20],
            "iteration_id": 0,
            "source": self.source,
            "confidence": min(1.0, self.count / 10.0),  # saturates at 10+ cases
        }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(r"\[([A-Z_]+)\]\s+<([^>]+)>")


def _parse_action(action_str: str) -> Optional[Tuple[str, str]]:
    """Return (verb, object_class) from '[VERB] <obj_class> (id)' strings."""
    if isinstance(action_str, dict):
        # dict form: {"VERB": ["obj", "id"]} or {"VERB": ["obj1", "id1", "obj2", "id2"]}
        for verb, args in action_str.items():
            obj_class = args[0] if args else "unknown"
            return verb.upper(), obj_class.lower()
    m = _ACTION_RE.search(str(action_str))
    if m:
        return m.group(1).upper(), m.group(2).lower()
    return None


def _load_error_info(path: Path) -> Dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        log.debug("skip %s: %s", path, exc)
        return {}


# ---------------------------------------------------------------------------
# Rule generation from patterns
# ---------------------------------------------------------------------------

# Maps (error_type, action_verb) → condition template
_CONDITION_TEMPLATES = {
    ("missing_step", "LOOKAT"):
        "Before [LOOKAT] <{obj}>, the target object must be in a reachable state "
        "(e.g. TV must be ON, book must be on an accessible surface)",
    ("missing_step", "LIE"):
        "Before [LIE] on a surface, character must WALK to it first",
    ("missing_step", "GRAB"):
        "Before [GRAB] <{obj}>, the container holding it may need to be OPENed first",
    ("missing_step", "RINSE"):
        "Handwashing requires both [WASH] and [RINSE] steps in sequence",
    ("missing_step", "WASH"):
        "Handwashing requires both [WASH] and [RINSE] steps in sequence",
    ("missing_step", "PUTIN"):
        "Before [PUTIN] <{obj}>, the item must be GRABbed first",
    ("missing_step", "DRINK"):
        "Before [DRINK] from <{obj}>, the item must be GRABbed and may need filling first",
    ("missing_step", "SWITCHON"):
        "Before [SWITCHON] <{obj}>, the object may need to be OPENed or positioned correctly",
    ("missing_step", "PUTON"):
        "Before [PUTON] <{obj}>, the item must be GRABbed first",
    ("affordance_error", None):
        "Object <{obj}> does not support action [{action}]: verify SWITCHABLE/GRABBABLE property",
    ("hallucination error", None):
        "Action [{action}] is not a valid VirtualHome action; use only the 22 supported verbs",
    ("wrong_temporal_order", None):
        "Action [{action}] on <{obj}> was executed out of order; check prerequisite steps",
    ("parsing error", None):
        "Action sequence contained a format/parsing error; verify action syntax",
    ("parameter error", None):
        "Action [{action}] received wrong parameter count or type",
}

_CYPHER_TEMPLATES = {
    ("missing_step", "LOOKAT"):
        "MATCH (o:Object {{class_name: '{obj}'}}) WHERE NOT (o)-[:HAS_STATE]->(:State {{name: 'ON'}}) RETURN o",
    ("missing_step", "GRAB"):
        "MATCH (o:Object {{class_name: '{obj}'}})-[:INSIDE]->(c:Object) WHERE c.class_name IN ['cabinet','fridge','freezer'] RETURN c",
    ("affordance_error", None):
        "MATCH (o:Object {{class_name: '{obj}'}}) WHERE NOT o.properties CONTAINS '{action}ABLE' RETURN o",
}


def _find_hallucinated_action(
    actions: List,
    valid_verbs: set,
) -> Optional[str]:
    """Return the first action in the list whose verb is not in valid_verbs."""
    for act in actions:
        if isinstance(act, dict):
            for verb in act:
                if verb.upper() not in valid_verbs:
                    obj_args = act[verb]
                    obj = obj_args[0] if obj_args else "unknown"
                    return f"[{verb.upper()}] <{obj}> (0)"
        else:
            m = _ACTION_RE.search(str(act))
            if m and m.group(1).upper() not in valid_verbs:
                return str(act)
    return None


def _make_condition(error_type: str, action: str, obj_class: str) -> str:
    tpl = (
        _CONDITION_TEMPLATES.get((error_type, action))
        or _CONDITION_TEMPLATES.get((error_type, None))
        or "Action [{action}] on <{obj}> failed with {error_type}"
    )
    return tpl.format(action=action, obj=obj_class, error_type=error_type)


def _make_cypher(error_type: str, action: str, obj_class: str) -> str:
    tpl = (
        _CYPHER_TEMPLATES.get((error_type, action))
        or _CYPHER_TEMPLATES.get((error_type, None))
        or ""
    )
    return tpl.format(action=action, obj=obj_class)


def _make_uid(error_type: str, action: str, obj_class: str) -> str:
    key = f"simlog|{error_type}|{action}|{obj_class}"
    return "simrule:" + hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_rules_from_logs(
    output_dir: str,
    min_count: int = 3,
) -> List[SimLogRule]:
    """Walk *output_dir* recursively for ``error_info.json`` files and extract rules.

    Parameters
    ----------
    output_dir:
        Root directory that contains all experiment result subdirectories.
    min_count:
        Minimum number of occurrences for a pattern to become a rule.

    Returns
    -------
    List of SimLogRule instances (not yet persisted).
    """
    root = Path(output_dir)
    error_files = list(root.rglob("error_info.json"))
    log.info("simulation_rule_extraction: found %d error_info.json files", len(error_files))

    # pattern_key → (count, evidence_list)
    # pattern_key = (error_type, action_verb, obj_class)
    counts: Dict[Tuple, int] = defaultdict(int)
    evidence: Dict[Tuple, List[str]] = defaultdict(list)

    # Valid VirtualHome action verbs (22 canonical actions)
    _VALID_VERBS = {
        "WALK", "RUN", "GRAB", "PUTIN", "PUTBACK", "DROP", "THROW",
        "OPEN", "CLOSE", "LOOKAT", "SITDOWN", "STANDUP", "LIE", "POUR",
        "TYPE", "WATCH", "MOVE", "WASH", "RINSE", "SCRUB",
        "PUTON", "PUTOFF", "SWITCHON", "SWITCHOFF", "DRINK", "EAT", "CUT",
        "SLEEP", "WAKEUP", "PUSH", "PULL", "READ",
    }

    for ef in error_files:
        data = _load_error_info(ef)
        for file_id, rec in data.items():
            error_type = rec.get("error_type")
            if not error_type:
                continue
            error_action_raw = rec.get("error_action")
            actions = rec.get("actions", [])

            if not error_action_raw:
                if error_type == "hallucination error":
                    # Find the first non-standard action in the sequence
                    error_action_raw = _find_hallucinated_action(actions, _VALID_VERBS)
                elif error_type == "parsing error":
                    # Parsing errors are format issues; record as generic (no specific action)
                    key = (error_type, "PARSE_FORMAT", "any")
                    counts[key] += 1
                    evidence[key].append(str(file_id))
                    continue
                else:
                    error_action_raw = actions[0] if actions else None

            if not error_action_raw:
                continue

            parsed = _parse_action(error_action_raw)
            if not parsed:
                continue
            verb, obj_class = parsed

            # Skip if verb is valid but appears under hallucination (was mis-attributed)
            if error_type == "hallucination error" and verb in _VALID_VERBS:
                continue

            key = (error_type, verb, obj_class)
            counts[key] += 1
            evidence[key].append(str(file_id))

    rules: List[SimLogRule] = []
    for (error_type, verb, obj_class), cnt in counts.items():
        if cnt < min_count:
            continue
        rules.append(SimLogRule(
            uid=_make_uid(error_type, verb, obj_class),
            action=verb,
            object_class=obj_class,
            error_type=error_type,
            condition=_make_condition(error_type, verb, obj_class),
            cypher_hint=_make_cypher(error_type, verb, obj_class),
            count=cnt,
            evidence_ids=evidence[(error_type, verb, obj_class)][:20],
        ))

    rules.sort(key=lambda r: r.count, reverse=True)
    log.info(
        "simulation_rule_extraction: extracted %d rules (min_count=%d) from %d patterns",
        len(rules), min_count, len(counts),
    )
    return rules


# ---------------------------------------------------------------------------
# Neo4j persistence
# ---------------------------------------------------------------------------

def persist_sim_rules(
    rules: List[SimLogRule],
    neo4j_driver,
    database: str = "neo4j",
) -> int:
    """Write SimLogRule nodes to Neo4j as DerivedRule nodes (idempotent via uid MERGE)."""
    if not rules:
        return 0
    rows = [r.to_neo4j_row() for r in rules]
    with neo4j_driver.session(database=database) as session:
        session.run(
            """
            UNWIND $rows AS r
            MERGE (dr:DerivedRule {uid: r.uid})
            SET dr.action        = r.action,
                dr.condition     = r.condition,
                dr.cypher_hint   = r.cypher_hint,
                dr.evidence_uids = r.evidence_uids,
                dr.iteration_id  = r.iteration_id,
                dr.source        = r.source,
                dr.confidence    = r.confidence
            WITH dr, r
            MERGE (a:Action {name: r.action})
            MERGE (dr)-[:CONSTRAINS]->(a)
            """,
            rows=rows,
        )
    log.info("simulation_rule_extraction: persisted %d DerivedRule nodes", len(rows))
    return len(rows)


def get_sim_rules_for_action(
    action: str,
    neo4j_driver,
    database: str = "neo4j",
) -> List[Dict]:
    """Retrieve simulation-extracted rules for a given action."""
    with neo4j_driver.session(database=database) as session:
        res = session.run(
            """
            MATCH (dr:DerivedRule {source: 'simulation_log'})-[:CONSTRAINS]->(a:Action {name: $action})
            RETURN dr.uid AS uid, dr.condition AS condition,
                   dr.cypher_hint AS cypher_hint, dr.confidence AS confidence
            ORDER BY dr.confidence DESC
            """,
            action=action.upper(),
        )
        return [dict(r) for r in res]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Extract rules from VirtualHome simulation logs")
    parser.add_argument("--output-dir", default="output/", help="Root output directory")
    parser.add_argument("--min-count", type=int, default=3, help="Min occurrences for a rule")
    parser.add_argument("--persist", action="store_true", help="Write rules to Neo4j")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="neo4j_password")
    args = parser.parse_args()

    rules = extract_rules_from_logs(args.output_dir, min_count=args.min_count)

    print(f"\nExtracted {len(rules)} rules:\n")
    for r in rules:
        print(f"  [{r.action}] <{r.object_class}> ({r.error_type}) count={r.count}")
        print(f"    condition: {r.condition}")
        print()

    if args.persist:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            args.neo4j_uri,
            auth=(args.neo4j_user, args.neo4j_password),
        )
        n = persist_sim_rules(rules, driver)
        driver.close()
        print(f"Persisted {n} rules to Neo4j.")
