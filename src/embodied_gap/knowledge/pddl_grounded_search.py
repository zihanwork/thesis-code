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
    if dataset == "behavior":
        plan = behavior_macro_plan(task, domain_path)
    elif dataset == "virtualhome":
        plan = virtualhome_macro_plan(task, domain_path)
    else:
        plan = ()
    if not plan:
        return ()
    trace = PDDLBackedExecutor().execute(task, PlanCandidate("pddl_macro", plan))
    if trace.executable and task.goal.is_satisfied(trace.final_state):
        return plan
    return ()


def behavior_macro_plan(task: Task, domain_path: str | Path) -> tuple[Action, ...]:
    builder = MacroPlanBuilder(task, domain_path)
    agent = builder.first_object_of_type("agent")
    if not agent:
        return ()

    inside_goals: list[tuple[str, str]] = []
    transform_goals: list[str] = []
    other_goals: list[str] = []
    for goal in task.goal_facts:
        name, args = split_action(positive_goal(goal))
        if goal.startswith("not("):
            if name in {"dusty", "stained"} and len(args) == 1:
                builder.clean_object(args[0], stain_type=name, agent=agent)
            elif name == "inside" and len(args) == 2:
                builder.remove_from_container(args[0], agent)
            continue
        if name == "inside" and len(args) == 2:
            inside_goals.append((args[0], args[1]))
        elif name == "soaked" and len(args) == 1:
            builder.soak_object(args[0], agent)
        elif name in {"sliced", "cooked", "frozen"} and len(args) == 1:
            transform_goals.append(goal)
        else:
            other_goals.append(goal)

    for container in sorted({container for _, container in inside_goals}):
        builder.try_action("navigate_to", container, agent)
        builder.try_action("open", container, agent)

    for goal in transform_goals:
        name, args = split_action(goal)
        if name == "sliced":
            builder.slice_object(args[0], agent)
        elif name == "cooked":
            builder.cook_object(args[0], agent)
        elif name == "frozen":
            builder.freeze_object(args[0], agent)

    for obj, container in inside_goals:
        builder.ensure_open(container, agent)
        builder.ensure_holding(obj, agent)
        builder.try_action("navigate_to", container, agent)
        builder.try_action("place_inside", obj, container, agent)

    for goal in other_goals:
        name, args = split_action(goal)
        if name == "ontop" and len(args) == 2:
            obj, surface = args
            builder.ensure_holding(obj, agent)
            builder.try_action("navigate_to", surface, agent)
            builder.try_action("place_ontop", obj, surface, agent)
        elif name == "nextto" and len(args) == 2:
            obj, target = args
            builder.ensure_holding(obj, agent)
            builder.try_action("navigate_to", target, agent)
            builder.try_action("place_nextto", obj, target, agent)
        elif name == "onfloor" and len(args) == 2:
            obj, floor = args
            builder.ensure_holding(obj, agent)
            builder.try_action("navigate_to", floor, agent)
            builder.try_action("place_onfloor", obj, floor, agent)
        elif name == "open" and len(args) == 1:
            builder.try_action("navigate_to", args[0], agent)
            builder.try_action("open", args[0], agent)
        elif name == "toggled_on" and len(args) == 1:
            builder.try_action("navigate_to", args[0], agent)
            builder.try_action("toggle_on", args[0], agent)
    return tuple(builder.plan)


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

    def held_behavior_objects(self) -> list[str]:
        return [
            split_action(fact)[1][0]
            for fact in sorted(self.state)
            if fact.startswith("holding(") and len(split_action(fact)[1]) == 1
        ]

    def release_held_behavior_objects(self, agent: str) -> None:
        for obj in self.held_behavior_objects():
            self.try_action("release", obj, agent)

    def containers_holding_behavior_object(self, obj: str) -> list[str]:
        return [
            args[1]
            for fact in sorted(self.state)
            for name, args in [split_action(fact)]
            if name == "inside" and len(args) == 2 and args[0] == obj
        ]

    def ensure_open(self, obj: str, agent: str) -> None:
        if self.has_fact(f"open({obj})"):
            return
        self.release_held_behavior_objects(agent)
        self.try_action("navigate_to", obj, agent)
        self.try_action("open", obj, agent)

    def ensure_behavior_accessible(self, obj: str, agent: str) -> None:
        for container in self.containers_holding_behavior_object(obj):
            if not self.has_fact(f"open({container})"):
                self.ensure_open(container, agent)
        self.try_action("navigate_to", obj, agent)

    def ensure_holding(self, obj: str, agent: str) -> None:
        if self.has_fact(f"holding({obj})"):
            return
        self.release_held_behavior_objects(agent)
        self.ensure_behavior_accessible(obj, agent)
        self.try_action("grasp", obj, agent)

    def first_feasible_action_with_tool(self, prefix: str) -> tuple[str, str] | None:
        for name, schema in sorted(self.domain.actions.items()):
            if not name.startswith(prefix) or not schema.parameters:
                continue
            tool_type = schema.parameters[0].type_name
            tools = self.objects_of_type(tool_type)
            if tools:
                return name, tools[0]
        return None

    def first_feasible_clean_action(self, prefix: str, target: str) -> tuple[str, str] | None:
        """Pick a clean_* action whose tool object exists and is not the target itself."""
        for name, schema in sorted(self.domain.actions.items()):
            if not name.startswith(prefix) or not schema.parameters:
                continue
            tool_type = schema.parameters[0].type_name
            tools = [tool for tool in self.objects_of_type(tool_type) if tool != target]
            if tools:
                return name, tools[0]
        return None

    def removable_from_container(self, obj: str) -> bool:
        return any(
            split_action(fact)[0] == "inside" and split_action(fact)[1][:1] == (obj,)
            for fact in self.state
        )

    def remove_from_container(self, obj: str, agent: str) -> None:
        """Make inside(obj, *) false by grasping the object out of its container."""
        if not self.removable_from_container(obj):
            return
        self.ensure_holding(obj, agent)
        floors = self.objects_of_type("floor_n_01")
        if floors:
            self.try_action("navigate_to", floors[0], agent)
            self.try_action("place_onfloor", obj, floors[0], agent)
        else:
            self.try_action("release", obj, agent)

    def first_sink(self) -> str | None:
        sinks = self.objects_of_type("sink_n_01")
        return sinks[0] if sinks else None

    def first_pan(self) -> str | None:
        pans = self.objects_of_type("pan_n_01")
        return pans[0] if pans else None

    def first_fridge(self) -> str | None:
        fridges = self.objects_of_type("electric_refrigerator_n_01")
        return fridges[0] if fridges else None

    def first_board(self) -> str | None:
        boards = self.objects_of_type("countertop_n_01")
        return boards[0] if boards else None

    def first_knife_action(self) -> tuple[str, str, str | None] | None:
        carving_knives = self.objects_of_type("carving_knife_n_01")
        board = self.first_board()
        if "slice-carvingknife" in self.domain.actions and carving_knives and board:
            return "slice-carvingknife", carving_knives[0], board
        knives = self.objects_of_type("knife_n_01")
        if "slice" in self.domain.actions and knives:
            return "slice", knives[0], None
        return None

    def ensure_sink_on(self, sink: str, agent: str) -> None:
        if self.has_fact(f"toggled_on({sink})"):
            return
        self.release_held_behavior_objects(agent)
        self.try_action("navigate_to", sink, agent)
        self.try_action("toggle_on", sink, agent)

    def soak_object(self, obj: str, agent: str) -> None:
        sink = self.first_sink()
        if not sink:
            return
        self.ensure_sink_on(sink, agent)
        self.ensure_holding(obj, agent)
        self.try_action("navigate_to", sink, agent)
        self.try_action("soak", obj, sink, agent)

    def clean_object(self, obj: str, *, stain_type: str, agent: str) -> None:
        prefix = "clean_dusty_" if stain_type == "dusty" else "clean_stained_"
        tool = self.first_feasible_clean_action(prefix, obj)
        if tool is None:
            return
        action_name, tool_obj = tool
        if stain_type == "stained" and not self.has_fact(f"soaked({tool_obj})"):
            self.soak_object(tool_obj, agent)
        self.ensure_holding(tool_obj, agent)
        self.try_action("navigate_to", obj, agent)
        self.try_action(action_name, tool_obj, obj, agent)

    def slice_object(self, obj: str, agent: str) -> None:
        knife = self.first_knife_action()
        if knife is None:
            return
        action_name, knife_obj, board = knife
        self.ensure_holding(knife_obj, agent)
        self.ensure_behavior_accessible(obj, agent)
        if action_name == "slice-carvingknife" and board:
            self.try_action(action_name, obj, knife_obj, board, agent)
        else:
            self.try_action(action_name, obj, knife_obj, agent)

    def cook_object(self, obj: str, agent: str) -> None:
        pan = self.first_pan()
        if not pan:
            return
        self.ensure_holding(obj, agent)
        self.try_action("navigate_to", pan, agent)
        self.try_action("place_ontop", obj, pan, agent)
        self.try_action("cook", obj, pan)

    def freeze_object(self, obj: str, agent: str) -> None:
        fridge = self.first_fridge()
        if not fridge:
            return
        self.ensure_open(fridge, agent)
        self.ensure_holding(obj, agent)
        self.try_action("navigate_to", fridge, agent)
        self.try_action("place_inside", obj, fridge, agent)
        self.try_action("freeze", obj, fridge)

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
