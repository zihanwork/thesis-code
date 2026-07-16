from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import re

from embodied_gap.core.action_schema import Action
from embodied_gap.core.task_schema import Task


FIELD_PROFILES: dict[str, tuple[str, ...]] = {
    "instruction": ("instruction",),
    "instruction_goal": ("instruction", "goal"),
    "instruction_state_goal": ("instruction", "state", "goal"),
    "instruction_state_goal_schema": ("instruction", "state", "goal", "schema"),
}

STRUCTURED_WEIGHTS = {
    "instruction": 0.40,
    "goal": 0.30,
    "state": 0.20,
    "schema": 0.10,
}


@dataclass(frozen=True)
class RetrievedExample:
    task: Task
    score: float
    components: dict[str, float] = field(default_factory=dict)


class ExampleRetriever:
    """Deterministic lexical, BM25, or field-structured task retrieval."""

    def __init__(
        self,
        examples: list[Task],
        *,
        method: str = "lexical",
        field_profile: str = "instruction_state_goal_schema",
    ) -> None:
        if method not in {"lexical", "bm25", "structured"}:
            raise ValueError(f"Unsupported retrieval method: {method}")
        if field_profile not in FIELD_PROFILES:
            raise ValueError(f"Unsupported retrieval field profile: {field_profile}")
        self.examples = [task for task in examples if task.gold_plan]
        self.method = method
        self.field_profile = field_profile
        self.fields = FIELD_PROFILES[field_profile]
        self._bm25_documents = [self._combined_tokens(task) for task in self.examples]
        self._bm25_document_frequency = self._document_frequency(self._bm25_documents)
        self._bm25_average_length = (
            sum(len(document) for document in self._bm25_documents)
            / len(self._bm25_documents)
            if self._bm25_documents
            else 0.0
        )

    def retrieve(self, query: Task, k: int = 3) -> list[RetrievedExample]:
        if k <= 0:
            return []
        scored: list[RetrievedExample] = []
        for index, example in enumerate(self.examples):
            if example.id == query.id:
                continue
            if self.method == "structured":
                score, components = self._structured_score(query, example)
            elif self.method == "bm25":
                score = self._bm25_score(query, index)
                components = {"bm25_raw": score}
            else:
                score = _jaccard(
                    set(self._combined_tokens(query)),
                    set(self._combined_tokens(example)),
                )
                components = {"lexical_jaccard": score}
            scored.append(RetrievedExample(example, score, components))

        if self.method == "bm25":
            scored = [
                RetrievedExample(
                    item.task,
                    round(item.score / (item.score + 1.0), 6) if item.score > 0 else 0.0,
                    {
                        **item.components,
                        "bm25_normalized": (
                            item.score / (item.score + 1.0) if item.score > 0 else 0.0
                        ),
                    },
                )
                for item in scored
            ]
        scored.sort(key=lambda item: (-item.score, item.task.id))
        return scored[:k]

    def _structured_score(
        self,
        query: Task,
        example: Task,
    ) -> tuple[float, dict[str, float]]:
        components = {
            field_name: _jaccard(
                set(_tokens(_field_text(query, field_name))),
                set(_tokens(_field_text(example, field_name))),
            )
            for field_name in self.fields
        }
        total_weight = sum(STRUCTURED_WEIGHTS[field_name] for field_name in self.fields)
        score = sum(
            components[field_name] * STRUCTURED_WEIGHTS[field_name]
            for field_name in self.fields
        ) / total_weight
        return round(score, 6), components

    def _combined_tokens(self, task: Task) -> list[str]:
        return _tokens(" ".join(_field_text(task, field_name) for field_name in self.fields))

    def _document_frequency(self, documents: list[list[str]]) -> Counter[str]:
        frequency: Counter[str] = Counter()
        for document in documents:
            frequency.update(set(document))
        return frequency

    def _bm25_score(self, query: Task, document_index: int) -> float:
        document = self._bm25_documents[document_index]
        if not document or not self._bm25_average_length:
            return 0.0
        query_terms = set(self._combined_tokens(query))
        term_frequency = Counter(document)
        document_count = len(self._bm25_documents)
        k1 = 1.5
        b = 0.75
        score = 0.0
        for term in query_terms:
            frequency = self._bm25_document_frequency.get(term, 0)
            inverse_document_frequency = math.log(
                1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            count = term_frequency.get(term, 0)
            denominator = count + k1 * (
                1.0 - b + b * len(document) / self._bm25_average_length
            )
            if denominator:
                score += inverse_document_frequency * count * (k1 + 1.0) / denominator
        return score


def _field_text(task: Task, field_name: str) -> str:
    if field_name == "instruction":
        return task.instruction
    if field_name == "goal":
        return " ".join(task.goal_facts)
    if field_name == "state":
        return " ".join(task.initial_facts)
    if field_name == "schema":
        metadata_actions = task.metadata.get("action_names", [])
        if not isinstance(metadata_actions, list):
            metadata_actions = []
        return " ".join([*task.allowed_actions, *(str(value) for value in metadata_actions)])
    raise ValueError(f"Unsupported retrieval field: {field_name}")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def adapt_plan(example: Task, query: Task) -> tuple[Action, ...]:
    adapted: list[Action] = []
    for action in example.gold_plan:
        new_action = action
        for slot_name, old_value in example.slots.items():
            new_value = query.slots.get(slot_name)
            if new_value:
                new_action = new_action.replace(old_value, new_value)
        adapted.append(new_action)
    return tuple(adapted)
