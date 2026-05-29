"""Unit tests for analysis.selective_recovery."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from analysis.selective_recovery import recover_sequence


@dataclass
class _FakeNode:
    id: int
    class_name: str
    properties: tuple = ()


def _scene(*nodes: _FakeNode) -> Dict[int, _FakeNode]:
    return {node.id: node for node in nodes}


def test_inserts_grab_before_read_when_object_is_grabbable():
    scene = _scene(_FakeNode(1000, "novel", ("GRABBABLE",)))
    draft = '{"WALK":["novel","1000"]}{"READ":["novel","1000"]}'

    recovered, report = recover_sequence(draft, scene)

    assert report.changed
    assert report.inserted_actions == 1
    assert recovered == (
        '{"WALK":["novel","1000"]}'
        '{"GRAB":["novel","1000"]}'
        '{"READ":["novel","1000"]}'
    )


def test_inserts_walk_and_grab_before_drink():
    scene = _scene(_FakeNode(1001, "drinking_glass", ("GRABBABLE",)))
    draft = '{"DRINK":["drinking_glass","1001"]}'

    recovered, report = recover_sequence(draft, scene)

    assert report.changed
    assert report.inserted_actions == 2
    assert recovered == (
        '{"WALK":["drinking_glass","1001"]}'
        '{"GRAB":["drinking_glass","1001"]}'
        '{"DRINK":["drinking_glass","1001"]}'
    )


def test_inserts_open_before_putin():
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

    recovered, report = recover_sequence(draft, scene)

    assert report.changed
    assert report.inserted_actions == 1
    assert recovered == (
        '{"WALK":["cup","112"]}'
        '{"GRAB":["cup","112"]}'
        '{"WALK":["dishwasher","104"]}'
        '{"OPEN":["dishwasher","104"]}'
        '{"PUTIN":["cup","112","dishwasher","104"]}'
    )


def test_leaves_unknown_id_unchanged():
    scene = _scene(_FakeNode(1000, "novel", ("GRABBABLE",)))
    draft = '{"READ":["novel","9999"]}'

    recovered, report = recover_sequence(draft, scene)

    assert not report.changed
    assert report.skipped_reason == "unsafe_unknown_id"
    assert recovered == draft


def test_leaves_clean_sequence_unchanged():
    scene = _scene(_FakeNode(1000, "novel", ("GRABBABLE",)))
    draft = (
        '{"WALK":["novel","1000"]}'
        '{"GRAB":["novel","1000"]}'
        '{"READ":["novel","1000"]}'
    )

    recovered, report = recover_sequence(draft, scene)

    assert not report.changed
    assert report.skipped_reason == "no_repair_needed"
    assert recovered == draft
