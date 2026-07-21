from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count, product
from pathlib import Path
from typing import Iterable

from embodied_gap.core.action_schema import Action
from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.state_schema import WorldState
from embodied_gap.core.task_schema import Task
from embodied_gap.execution.pddl_executor import (
    PDDLBackedExecutor,
    PDDLActionSchema,
    PDDLContext,
    apply_effect,
    collect_effects,
    collect_missing,
    eval_condition,
    load_domain,
    object_types,
    resolve_domain_path,
    type_compatible,
)
from embodied_gap.knowledge.failure_memory import classify_failure_patterns


@dataclass(frozen=True)
class PDDLSearchResult:
    actions: tuple[Action, ...]
    explored_states: int
    solved: bool
    reason: str
    candidate_count: int = 0
    memory_patterns: tuple[str, ...] = ()


class PDDLGroundedSearch:
    """Goal-directed grounded search over EAI PDDL domains."""

    def __init__(
        self,
        max_depth: int = 18,
        max_expansions: int = 500,
        max_candidates: int = 1200,
    ) -> None:
        self.max_depth = max_depth
        self.max_expansions = max_expansions
        self.max_candidates = max_candidates
        self._cache: dict[str, PDDLSearchResult] = {}

    def can_search(self, task: Task) -> bool:
        return bool(resolve_domain_path(task))

    def search(self, task: Task) -> PDDLSearchResult:
        cached = self._cache.get(task.id)
        if cached is not None:
            return cached

        def remember(result: PDDLSearchResult) -> PDDLSearchResult:
            self._cache[task.id] = result
            return result

        domain_path = resolve_domain_path(task)
        if domain_path is None:
            return remember(PDDLSearchResult((), 0, False, "missing_pddl_domain"))

        domain = load_domain(domain_path)
        objects = object_types(task)
        macro = macro_plan(task, domain_path)
        if macro:
            patterns = tuple(pattern.name for pattern in classify_failure_patterns(task))
            return remember(PDDLSearchResult(macro, 0, True, "macro_goal_regression", 0, patterns))

        candidates, candidate_limit_reached = bounded_grounded_actions(
            task,
            domain_path,
            max_candidates=self.max_candidates,
        )
        if candidate_limit_reached:
            return remember(
                PDDLSearchResult(
                    (),
                    0,
                    False,
                    "too_many_grounded_candidates",
                    len(candidates),
                )
            )
        if not candidates:
            return remember(PDDLSearchResult((), 0, False, "no_grounded_candidates"))

        initial = frozenset(task.initial_facts)
        frontier: list[tuple[int, int, int, frozenset[str], tuple[Action, ...]]] = []
        counter = count()
        initial_score = self._priority(task, initial, initial, candidates, domain.actions, objects)
        heappush(frontier, (initial_score, 0, next(counter), initial, ()))
        best_depth: dict[frozenset[str], int] = {initial: 0}
        explored = 0

        while frontier and explored < self.max_expansions:
            _, depth, _, state_key, prefix = heappop(frontier)
            explored += 1
            state = set(state_key)
            if task.goal.is_satisfied(WorldState.from_facts(state)):
                return remember(PDDLSearchResult(prefix, explored, True, "goal_satisfied", len(candidates)))
            if depth >= self.max_depth:
                continue

            for action in candidates:
                schema = domain.actions.get(action_name(action))
                if schema is None:
                    continue
                next_state = apply_grounded_action(schema, objects, state, action)
                if next_state is None or next_state == state_key:
                    continue
                next_depth = depth + 1
                if best_depth.get(next_state, self.max_depth + 1) <= next_depth:
                    continue
                best_depth[next_state] = next_depth
                priority = self._priority(task, next_state, state_key, candidates, domain.actions, objects)
                heappush(frontier, (priority + next_depth, next_depth, next(counter), next_state, prefix + (action,)))

        return remember(PDDLSearchResult((), explored, False, "search_exhausted", len(candidates)))

    def _priority(
        self,
        task: Task,
        state: frozenset[str],
        previous_state: frozenset[str],
        candidates: tuple[Action, ...],
        schemas: dict[str, PDDLActionSchema],
        objects: dict[str, str],
    ) -> int:
        del candidates, schemas, objects
        world = WorldState.from_facts(state)
        previous_world = WorldState.from_facts(previous_state)
        unsatisfied = sum(
            1 for fact in task.goal_facts if not task.goal._fact_satisfied(fact, world)
        )
        goal_gain = sum(
            1
            for fact in task.goal_facts
            if task.goal._fact_satisfied(fact, world)
            and not task.goal._fact_satisfied(fact, previous_world)
        )
        return unsatisfied * 20 - goal_gain * 5


def macro_plan(task: Task, domain_path: str | Path) -> tuple[Action, ...]:
    dataset = task.slots.get("dataset")
    if dataset == "virtualhome":
        plan = virtualhome_macro_plan(task, domain_path)
    else:
        plan = ()
    if not plan:
        return ()
    trace = PDDLBackedExecutor().execute(task, PlanCandidate("pddl_macro", plan))
    if trace.executable and task.goal.is_satisfied(trace.final_state):
        return plan
    return ()


def virtualhome_macro_plan(task: Task, domain_path: str | Path) -> tuple[Action, ...]:
    builder = MacroPlanBuilder(task, domain_path)
    character = builder.first_object_of_type("character") or "character"

    inside_character_goals: list[str] = []
    facing_goals: list[str] = []
    container_goals: list[str] = []
    surface_goals: list[str] = []
    closed_goals: list[str] = []
    plugged_goals: list[str] = []
    on_goals: list[str] = []
    regular_goals: list[str] = []
    right_hand_goals: list[str] = []
    left_hand_goals: list[str] = []
    for goal in task.goal_facts:
        name, args = split_action(positive_goal(goal))
        if goal.startswith("not("):
            continue
        if name == "holds_rh" and len(args) == 2:
            right_hand_goals.append(goal)
        elif name == "holds_lh" and len(args) == 2:
            left_hand_goals.append(goal)
        elif name == "inside" and len(args) == 2 and args[0] == character:
            inside_character_goals.append(goal)
        elif name == "facing" and len(args) == 2 and args[0] == character:
            facing_goals.append(goal)
        elif name == "obj_inside" and len(args) == 2:
            container_goals.append(goal)
        elif name == "obj_ontop" and len(args) == 2:
            surface_goals.append(goal)
        elif name == "closed" and len(args) == 1:
            closed_goals.append(goal)
        elif name == "plugged_in" and len(args) == 1:
            plugged_goals.append(goal)
        elif name == "on" and len(args) == 1:
            on_goals.append(goal)
        else:
            regular_goals.append(goal)

    for goal in inside_character_goals:
        name, args = split_action(goal)
        builder.try_action("walk_into", character, args[1])

    for goal in facing_goals:
        _, args = split_action(goal)
        builder.try_action("turn_to", character, args[1])

    for goal in container_goals:
        _, args = split_action(goal)
        obj, container = args
        builder.ensure_vh_holding(obj, character)
        builder.ensure_vh_open(container, character)
        builder.try_action("walk_towards", character, container)
        builder.try_action("put_inside", character, obj, container)

    for goal in surface_goals:
        _, args = split_action(goal)
        obj, surface = args
        builder.ensure_vh_holding(obj, character)
        builder.try_action("walk_towards", character, surface)
        builder.try_action("put_on", character, obj, surface)

    for goal in right_hand_goals + left_hand_goals:
        _, args = split_action(goal)
        obj = args[1]
        builder.ensure_vh_holding(obj, character)

    for goal in closed_goals:
        _, args = split_action(goal)
        obj = args[0]
        builder.try_action("walk_towards", character, obj)
        builder.try_action("close", character, obj)

    for goal in plugged_goals + on_goals:
        _, args = split_action(goal)
        obj = args[0]
        builder.try_action("walk_towards", character, obj)
        builder.try_action("plug_in", character, obj)

    for goal in on_goals:
        _, args = split_action(goal)
        obj = args[0]
        builder.try_action("walk_towards", character, obj)
        builder.try_action("switch_on", character, obj)

    for goal in regular_goals:
        name, args = split_action(goal)
        if name == "open" and len(args) == 1:
            builder.ensure_vh_open(args[0], character)
    return tuple(builder.plan)


class MacroPlanBuilder:
    def __init__(self, task: Task, domain_path: str | Path) -> None:
        self.task = task
        self.domain = load_domain(domain_path)
        self.objects = object_types(task)
        self.state: set[str] = set(task.initial_facts)
        self.plan: list[Action] = []

    def first_object_of_type(self, type_name: str) -> str | None:
        for object_name, object_type in sorted(self.objects.items()):
            if object_type == type_name:
                return object_name
        return None

    def try_action(self, name: str, *args: str) -> bool:
        schema = self.domain.actions.get(name)
        if schema is None:
            return False
        action = format_grounded_action(name, tuple(args))
        next_state = apply_grounded_action(schema, self.objects, self.state, action)
        if next_state is None:
            return False
        if next_state != frozenset(self.state):
            self.plan.append(action)
            self.state = set(next_state)
        return True

    def has_fact(self, fact: str) -> bool:
        return fact in self.state

    def objects_of_type(self, type_name: str) -> list[str]:
        return sorted(
            object_name
            for object_name, object_type in self.objects.items()
            if type_compatible(object_type, type_name)
        )

    def held_virtualhome_objects(self, character: str) -> list[str]:
        held: list[str] = []
        for fact in sorted(self.state):
            name, args = split_action(fact)
            if name in {"holds_lh", "holds_rh"} and len(args) == 2 and args[0] == character:
                held.append(args[1])
        return held

    def containers_holding_virtualhome_object(self, obj: str) -> list[str]:
        return [
            args[1]
            for fact in sorted(self.state)
            for name, args in [split_action(fact)]
            if name == "obj_inside" and len(args) == 2 and args[0] == obj
        ]

    def ensure_vh_open(self, obj: str, character: str) -> None:
        if self.has_fact(f"open({obj})"):
            return
        self.try_action("walk_towards", character, obj)
        self.try_action("open", character, obj)

    def ensure_vh_accessible(self, obj: str, character: str) -> None:
        for container in self.containers_holding_virtualhome_object(obj):
            if self.has_fact(f"closed({container})"):
                self.ensure_vh_open(container, character)
        self.try_action("walk_towards", character, obj)

    def ensure_vh_holding(self, obj: str, character: str) -> None:
        if obj in self.held_virtualhome_objects(character):
            return
        self.ensure_vh_accessible(obj, character)
        self.try_action("grab", character, obj)


def generate_grounded_actions(task: Task, domain_path: str | Path) -> Iterable[Action]:
    domain = load_domain(domain_path)
    objects = object_types(task)
    allowed_names = set(task.allowed_actions) if task.allowed_actions else set(domain.actions)
    for name, schema in sorted(domain.actions.items()):
        if name not in allowed_names:
            continue
        object_domains = [
            sorted(
                object_name
                for object_name, object_type in objects.items()
                if type_compatible(object_type, parameter.type_name)
            )
            for parameter in schema.parameters
        ]
        if any(not domain_values for domain_values in object_domains):
            continue
        for args in product(*object_domains):
            yield format_grounded_action(name, args)


def bounded_grounded_actions(
    task: Task,
    domain_path: str | Path,
    *,
    max_candidates: int,
) -> tuple[tuple[Action, ...], bool]:
    candidates: list[Action] = []
    for action in generate_grounded_actions(task, domain_path):
        candidates.append(action)
        if len(candidates) > max_candidates:
            return tuple(candidates), True
    return tuple(candidates), False


def apply_grounded_action(
    schema: PDDLActionSchema,
    objects: dict[str, str],
    state: set[str] | frozenset[str],
    action: Action,
) -> frozenset[str] | None:
    bindings = action_bindings(schema, action)
    if bindings is None:
        return None
    working_state = set(state)
    context = PDDLContext(objects=objects, state=working_state, bindings=bindings)
    if not eval_condition(schema.precondition, context):
        return None
    apply_effect(schema.effect, context)
    return frozenset(working_state)


def wanted_preconditions_for_goals(
    task: Task,
    state: frozenset[str],
    candidates: tuple[Action, ...],
    schemas: dict[str, PDDLActionSchema],
    objects: dict[str, str],
) -> set[str]:
    wanted: set[str] = set()
    world = WorldState.from_facts(state)
    unsatisfied_goals = [fact for fact in task.goal_facts if not task.goal._fact_satisfied(fact, world)]
    if not unsatisfied_goals:
        return wanted
    for action in candidates:
        schema = schemas.get(action_name(action))
        if schema is None:
            continue
        bindings = action_bindings(schema, action)
        if bindings is None:
            continue
        context = PDDLContext(objects=objects, state=set(state), bindings=bindings)
        adds, _ = collect_effects(schema.effect, context)
        if not (set(unsatisfied_goals) & adds):
            continue
        wanted.update(collect_missing(schema.precondition, context))
    return wanted


def action_bindings(schema: PDDLActionSchema, action: Action) -> dict[str, str] | None:
    name, args = split_action(action)
    if name != schema.name or len(args) != len(schema.parameters):
        return None
    return dict(zip((parameter.name for parameter in schema.parameters), args, strict=True))


def split_action(action: Action) -> tuple[str, tuple[str, ...]]:
    if "(" not in action or not action.endswith(")"):
        return action, ()
    name, raw_args = action.split("(", 1)
    raw_args = raw_args[:-1].strip()
    args = tuple(part.strip() for part in raw_args.split(",")) if raw_args else ()
    return name, args


def action_name(action: Action) -> str:
    return split_action(action)[0]


def positive_goal(goal: str) -> str:
    if goal.startswith("not(") and goal.endswith(")"):
        return goal[4:-1]
    return goal


def format_grounded_action(name: str, args: tuple[str, ...]) -> Action:
    return f"{name}({', '.join(args)})" if args else f"{name}()"
