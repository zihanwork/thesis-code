"""Direction 2: bad-case → LLM rule induction → write derived rules to Neo4j.

For each batch of bad cases, calls an LLM to:
  1. Identify the common failure pattern
  2. Propose a new constraint rule (action + condition)
  3. Persist the derived rule as a DerivedRule node in Neo4j

Usage::

    from analysis.kb.rule_induction import induce_rules_from_bad_cases
    new_rules = induce_rules_from_bad_cases(bad_cases, llm_client, iteration_id=1)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ data types

@dataclass
class DerivedRule:
    """A constraint rule inducted from failure cases by an LLM."""
    uid: str
    action: str                   # The action this rule applies to (e.g. "GRAB")
    condition: str                # Natural language condition (e.g. "object must be GRABBABLE")
    cypher_hint: str              # Optional Cypher-like formal expression
    evidence_uids: List[str]      # bad case uids that triggered this rule
    iteration_id: int
    source: str = "llm_induction"
    confidence: float = 1.0

    def to_neo4j_row(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "action": self.action,
            "condition": self.condition,
            "cypher_hint": self.cypher_hint,
            "evidence_uids": self.evidence_uids,
            "iteration_id": self.iteration_id,
            "source": self.source,
            "confidence": self.confidence,
        }


# ------------------------------------------------------------------ LLM prompt

_SYSTEM_PROMPT = """You are a knowledge graph expert for embodied planning in VirtualHome.
Given a list of failed action sequences and their error reasons, your job is to identify
common constraint patterns and express them as concise rules.

Each rule must follow this JSON schema:
{
  "action": "<VirtualHome action name, e.g. GRAB>",
  "condition": "<natural language precondition that was violated>",
  "cypher_hint": "<optional formal constraint, e.g. arg1 MUST HAVE property GRABBABLE>"
}

Return a JSON array of rules. Return only the JSON array, no other text.
Focus on patterns that appear in multiple failures. Skip one-off errors."""

_USER_TEMPLATE = """Failed action sequences from iteration {iteration_id}:

{cases_text}

Identify up to 5 new constraint rules that would prevent these failures.
Return JSON array only."""


def _format_cases(bad_cases: List[Any]) -> str:
    lines = []
    for i, bc in enumerate(bad_cases[:20]):  # cap at 20 to stay within context
        task = getattr(bc, "task_id", "") or bc.get("task_id", "") if isinstance(bc, dict) else ""
        failure = getattr(bc, "failure_detail", "") or bc.get("failure_detail", "") if isinstance(bc, dict) else ""
        action = getattr(bc, "failed_action", "") or bc.get("failed_action", "") if isinstance(bc, dict) else ""
        lines.append(f"{i+1}. task={task} action={action} reason={failure}")
    return "\n".join(lines)


def _call_llm(client: Any, cases_text: str, iteration_id: int) -> List[Dict]:
    """Call the LLM to induce rules. client must have a chat.completions.create method."""
    prompt = _USER_TEMPLATE.format(iteration_id=iteration_id, cases_text=cases_text)
    try:
        resp = client.chat.completions.create(
            model=getattr(client, "_model", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as exc:
        log.warning("rule_induction: LLM call failed: %s", exc)
        return []


def _make_uid(action: str, condition: str, iteration_id: int) -> str:
    key = f"{action}|{condition}|{iteration_id}"
    return "rule:" + hashlib.md5(key.encode()).hexdigest()[:12]


# ------------------------------------------------------------------ main API

def induce_rules_from_bad_cases(
    bad_cases: List[Any],
    llm_client: Any,
    iteration_id: int = 0,
    sleep: float = 1.0,
) -> List[DerivedRule]:
    """Use the LLM to induce new constraint rules from bad cases.

    Parameters
    ----------
    bad_cases:
        List of BadCase dataclass instances or dicts with task_id/failure_detail/failed_action.
    llm_client:
        OpenAI-compatible client with chat.completions.create().
    iteration_id:
        Current harness iteration number (stored on the DerivedRule node).
    sleep:
        Seconds to wait after LLM call (rate limit buffer).

    Returns
    -------
    List of DerivedRule instances (not yet persisted to Neo4j).
    """
    if not bad_cases:
        return []

    cases_text = _format_cases(bad_cases)
    raw_rules = _call_llm(llm_client, cases_text, iteration_id)
    time.sleep(sleep)

    evidence_uids = []
    for bc in bad_cases:
        uid = getattr(bc, "uid", None) or (bc.get("uid") if isinstance(bc, dict) else None)
        if uid:
            evidence_uids.append(str(uid))

    derived: List[DerivedRule] = []
    for r in raw_rules:
        action = str(r.get("action", "")).upper().strip()
        condition = str(r.get("condition", "")).strip()
        if not action or not condition:
            continue
        derived.append(DerivedRule(
            uid=_make_uid(action, condition, iteration_id),
            action=action,
            condition=condition,
            cypher_hint=str(r.get("cypher_hint", "")),
            evidence_uids=evidence_uids[:10],
            iteration_id=iteration_id,
        ))

    log.info("rule_induction: inducted %d rules from %d bad cases (iter %d)",
             len(derived), len(bad_cases), iteration_id)
    return derived


def persist_derived_rules(
    rules: List[DerivedRule],
    neo4j_driver: Any,
    database: str = "neo4j",
) -> int:
    """Write DerivedRule nodes to Neo4j (idempotent via uid MERGE)."""
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
    log.info("rule_induction: persisted %d DerivedRule nodes", len(rows))
    return len(rows)


def get_derived_rules_for_action(
    action: str,
    neo4j_driver: Any,
    database: str = "neo4j",
) -> List[Dict]:
    """Retrieve all derived rules for a given action from Neo4j."""
    with neo4j_driver.session(database=database) as session:
        res = session.run(
            """
            MATCH (dr:DerivedRule)-[:CONSTRAINS]->(a:Action {name: $action})
            RETURN dr.uid AS uid, dr.condition AS condition,
                   dr.cypher_hint AS cypher_hint, dr.iteration_id AS iteration_id,
                   dr.confidence AS confidence
            ORDER BY dr.iteration_id DESC
            """,
            action=action.upper(),
        )
        return [dict(r) for r in res]
