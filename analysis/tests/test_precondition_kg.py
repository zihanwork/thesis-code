"""Unit tests for analysis.precondition_kg."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pytest

from analysis.precondition_kg import (
    PreconditionKG,
    Violation,
    ViolationCode,
    parse_action_sequence,
)


@dataclass
class _FakeNode:
    id: int
    class_name: str
    properties: tuple = ()


def _scene(*nodes: _FakeNode) -> Dict[int, _FakeNode]:
    return {n.id: n for n in nodes}


@pytest.fixture
def kg() -> PreconditionKG:
    return PreconditionKG()


def test_parse_action_sequence_basic():
    steps, parse_viols = parse_action_sequence(
        '{"WALK":["floor_lamp","1000"]}{"SWITCHON":["floor_lamp","1000"]}'
    )
    assert parse_viols == []
    assert steps == [
        ("WALK", ["floor_lamp", "1000"]),
        ("SWITCHON", ["floor_lamp", "1000"]),
    ]


def test_parse_action_sequence_empty_emits_parse_error():
    steps, parse_viols = parse_action_sequence("not-an-action")
    assert steps == []
    assert len(parse_viols) == 1
    assert parse_viols[0].code == ViolationCode.PARSE_ERROR


def test_missing_walk_before_switchon(kg: PreconditionKG):
    scene = _scene(_FakeNode(1000, "floor_lamp", ("HAS_SWITCH",)))
    viols = kg.verify('{"SWITCHON":["floor_lamp","1000"]}', scene)
    codes = {v.code for v in viols}
    assert ViolationCode.MISSING_WALK in codes


def test_walk_then_switchon_is_clean(kg: PreconditionKG):
    scene = _scene(_FakeNode(1000, "floor_lamp", ("HAS_SWITCH",)))
    viols = kg.verify(
        '{"WALK":["floor_lamp","1000"]}{"SWITCHON":["floor_lamp","1000"]}',
        scene,
    )
    assert viols == []


def test_putin_without_open_triggers_missing_open(kg: PreconditionKG):
    scene = _scene(
        _FakeNode(112, "cup", ("GRABBABLE",)),
        _FakeNode(104, "dishwasher", ("CAN_OPEN",)),
    )
    draft = (
        '{"WALK":["cup","112"]}'
        '{"GRAB":["cup","112"]}'
        '{"WALK":["dishwasher","104"]}'
        '{"PUTIN":["cup","112","dishwasher","104"]}'
    )
    viols = kg.verify(draft, scene)
    codes = {v.code for v in viols}
    assert ViolationCode.MISSING_OPEN in codes


def test_unknown_id_reported(kg: PreconditionKG):
    scene = _scene(_FakeNode(1000, "floor_lamp", ("HAS_SWITCH",)))
    viols = kg.verify(
        '{"WALK":["floor_lamp","9999"]}{"SWITCHON":["floor_lamp","9999"]}',
        scene,
    )
    codes = {v.code for v in viols}
    assert ViolationCode.UNKNOWN_ID in codes


def test_not_grabbable_reported(kg: PreconditionKG):
    scene = _scene(_FakeNode(50, "wall", ()))  # no GRABBABLE property
    viols = kg.verify(
        '{"WALK":["wall","50"]}{"GRAB":["wall","50"]}',
        scene,
    )
    codes = {v.code for v in viols}
    assert ViolationCode.NOT_GRABBABLE in codes


def test_arity_mismatch_reported(kg: PreconditionKG):
    scene = _scene(_FakeNode(1, "floor_lamp", ("HAS_SWITCH",)))
    viols = kg.verify('{"SWITCHON":["floor_lamp"]}', scene)
    codes = {v.code for v in viols}
    assert ViolationCode.ARITY_MISMATCH in codes


def test_summarise_reports_counts(kg: PreconditionKG):
    scene = _scene(_FakeNode(1, "floor_lamp", ("HAS_SWITCH",)))
    viols = kg.verify('{"SWITCHON":["floor_lamp","1"]}', scene)
    report = kg.summarise(viols)
    assert "MISSING_WALK" in report
    assert report.startswith("[KG Verifier]")


def test_summarise_clean_case(kg: PreconditionKG):
    report = kg.summarise([])
    assert "No precondition violations" in report
