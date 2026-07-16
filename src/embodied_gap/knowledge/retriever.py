from __future__ import annotations

from dataclasses import dataclass

from embodied_gap.core.action_schema import Action, tokenize
from embodied_gap.core.task_schema import Task


@dataclass(frozen=True)
class RetrievedExample:
    task: Task
    score: float


class ExampleRetriever:
    """Simple deterministic retriever over task text, tags, and slots."""

    def __init__(self, examples: list[Task]) -> None:
        self.examples = [task for task in examples if task.gold_plan]

    def retrieve(self, query: Task, k: int = 3) -> list[RetrievedExample]:
        query_tokens = tokenize(query.instruction + " " + " ".join(query.tags))
        scored: list[RetrievedExample] = []
        for example in self.examples:
            if example.id == query.id:
                continue
            example_tokens = tokenize(example.instruction + " " + " ".join(example.tags))
            union = query_tokens | example_tokens
            score = len(query_tokens & example_tokens) / len(union) if union else 0.0
            score += 0.25 * len(set(query.tags) & set(example.tags))
            score += 0.10 * len(set(query.slots) & set(example.slots))
            scored.append(RetrievedExample(task=example, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]


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
