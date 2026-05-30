"""Neo4j-backed precondition KG (drop-in for ``PreconditionKG``).

Loads the rule base from Neo4j on first use (so the rule semantics live
in the graph as the source of truth) and reuses the existing symbolic
verifier from ``analysis.precondition_kg`` to keep verification
behaviour identical.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from analysis.precondition_kg import ActionRule, PreconditionKG, Violation
except ModuleNotFoundError:
    from precondition_kg import ActionRule, PreconditionKG, Violation  # type: ignore[no-redef]

from .config import (
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)

log = logging.getLogger(__name__)


def _load_rules_from_neo4j() -> Dict[str, ActionRule]:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    rules: Dict[str, ActionRule] = {}
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            actions = session.run("MATCH (a:Action) RETURN a.name AS name, a.arity AS arity")
            for rec in actions:
                name = str(rec["name"])
                arity = int(rec["arity"]) if rec["arity"] is not None else 1
                props_arg1, props_arg2 = _props_for(session, name)
                pres = _preconditions_for(session, name)
                rules[name] = ActionRule(
                    name=name,
                    arity=arity,
                    required_props_arg1=props_arg1,
                    required_props_arg2=props_arg2,
                    requires_walk_to_arg1="WALK_TO_ARG1" in pres,
                    requires_walk_to_arg2="WALK_TO_ARG2" in pres,
                    requires_hold_arg1="HOLD_ARG1" in pres,
                    requires_open_arg2="OPEN_ARG2" in pres,
                )
    finally:
        driver.close()
    return rules


def _props_for(session, action_name: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    res = session.run(
        """
        MATCH (a:Action {name: $name})-[r:REQUIRES_PROP]->(p:Property)
        RETURN p.name AS prop, r.slot AS slot
        """,
        name=action_name,
    )
    arg1: List[str] = []
    arg2: List[str] = []
    for rec in res:
        prop = str(rec["prop"])
        slot = int(rec["slot"]) if rec["slot"] is not None else 1
        if slot == 2:
            arg2.append(prop)
        else:
            arg1.append(prop)
    return tuple(arg1), tuple(arg2)


def _preconditions_for(session, action_name: str) -> set:
    res = session.run(
        """
        MATCH (a:Action {name: $name})-[:REQUIRES_STATE]->(p:Precondition)
        RETURN p.kind AS kind
        """,
        name=action_name,
    )
    return {str(r["kind"]) for r in res}


class PersistentPreconditionKG(PreconditionKG):
    """``PreconditionKG`` whose rules are read from Neo4j on construction."""

    _SHARED: Optional["PersistentPreconditionKG"] = None  # type: ignore[assignment]

    def __init__(self, rules: Optional[Dict[str, ActionRule]] = None) -> None:
        if rules is None:
            try:
                rules = _load_rules_from_neo4j()
            except Exception as exc:
                log.warning("persistent KG: rule load from Neo4j failed (%s); "
                            "falling back to in-code rules", exc)
                rules = None
        if not rules:
            try:
                from analysis.precondition_kg import RULES as _DEFAULT_RULES
            except ModuleNotFoundError:
                from precondition_kg import RULES as _DEFAULT_RULES  # type: ignore[no-redef]

            rules = dict(_DEFAULT_RULES)
        super().__init__(rules)

    @classmethod
    def shared(cls) -> "PersistentPreconditionKG":  # type: ignore[override]
        if cls._SHARED is None:
            cls._SHARED = cls()
        return cls._SHARED

    # ----- diagnostic helper: similar past failures via Neo4j
    def query_failures_for_action(
        self,
        action: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            with driver.session(database=NEO4J_DATABASE) as session:
                res = session.run(
                    """
                    MATCH (f:FailureCase)-[:VIOLATES]->(a:Action {name: $name})
                    RETURN f.uid AS uid, f.file_id AS file_id, f.model AS model,
                           f.failure_type AS failure_type, f.task AS task,
                           f.raw AS raw
                    LIMIT $limit
                    """,
                    name=action,
                    limit=limit,
                )
                return [dict(r) for r in res]
        finally:
            driver.close()


__all__ = ["PersistentPreconditionKG"]
