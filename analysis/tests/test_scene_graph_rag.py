"""Unit tests for analysis.scene_graph_rag."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.scene_graph_rag import (
    SceneGraphRetriever,
    _extract_script_id,
    _serialise,
)


@pytest.fixture(scope="module")
def retriever() -> SceneGraphRetriever:
    return SceneGraphRetriever()


def test_extract_script_id_basic():
    assert _extract_script_id("11_1") == "11_1"
    assert _extract_script_id("27_2") == "27_2"
    assert _extract_script_id("") is None
    assert _extract_script_id("garbage") is None


def test_retrieve_returns_block_for_known_task(retriever: SceneGraphRetriever):
    block = retriever.retrieve(
        "11_1",
        task_prompt="Task: Turn on light. The relevant objects in the scene are: floor_lamp(1000).",
    )
    assert block is not None, "expected subgraph for file11_1.json which ships with EAI"
    assert block.startswith("[Scene Subgraph]")
    assert block.rstrip().endswith("[/Scene Subgraph]")
    assert "Objects:" in block


def test_retrieve_returns_none_for_missing_identifier(retriever: SceneGraphRetriever):
    assert retriever.retrieve(None) is None
    assert retriever.retrieve("99999999_9") is None  # no such scene file


def test_retrieve_respects_object_budget(retriever: SceneGraphRetriever):
    block = retriever.retrieve(
        "11_1",
        task_prompt="Task: Turn on light",
        k_neighbours=3,
        max_objects=5,
    )
    assert block is not None
    # Count "- " object lines between "Objects:" and either "Relations:" or closing tag.
    lines = block.splitlines()
    i = lines.index("Objects:") + 1
    count = 0
    while i < len(lines) and lines[i].startswith("- "):
        count += 1
        i += 1
    assert count <= 5, f"expected at most 5 objects, got {count}"


def test_load_scene_objects_returns_nodes(retriever: SceneGraphRetriever):
    objs = retriever.load_scene_objects("11_1")
    assert objs, "expected non-empty scene_objects for 11_1"
    sample_id = next(iter(objs))
    node = objs[sample_id]
    assert hasattr(node, "class_name")
    assert hasattr(node, "properties")


def test_serialise_empty_returns_empty_string():
    assert _serialise([], []) == ""
