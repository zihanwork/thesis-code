from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Iterable

from embodied_gap.core.action_schema import parse_call
from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.state_schema import WorldState
from embodied_gap.core.task_schema import Task
from embodied_gap.core.violation_schema import Violation, ViolationType
from embodied_gap.datasets.eai_adapter import PDDLExpr, format_fact, parse_pddl
from embodied_gap.datasets.resource_paths import resolve_domain_path

from .symbolic_executor import ExecutionTrace, StepTrace


@dataclass(frozen=True)
class PDDLParameter:
    name: str
    type_name: str


@dataclass(frozen=True)
class PDDLActionSchema:
    name: str
    parameters: tuple[PDDLParameter, ...]
    precondition: PDDLExpr
    effect: PDDLExpr


@dataclass(frozen=True)
class PDDLDomain:
    name: str
    actions: dict[str, PDDLActionSchema]


@dataclass(frozen=True)
class PDDLExecutionFailure:
    violation_type: ViolationType
    message: str
    missing_preconditions: tuple[str, ...] = ()
    details: dict[str, Any] | None = None


class PDDLBackedExecutor:
    """Small PDDL interpreter for clean EAI action-sequencing tasks.

    It supports the subset used by EAI VirtualHome and BEHAVIOR domains:
    conjunction, disjunction, negation, existential/universal quantifiers, and
    conditional effects.
    """

    def can_execute(self, task: Task) -> bool:
        return bool(resolve_domain_path(task))

    def execute(
        self,
        task: Task,
        plan: PlanCandidate,
        *,
        stop_on_safety: bool = False,
    ) -> ExecutionTrace:
        domain_path = resolve_domain_path(task)
        if domain_path is None:
            raise ValueError(f"No PDDL domain path available for task {task.id}")

        if plan.rejected:
            return ExecutionTrace(
                task_id=task.id,
                planner_name=plan.planner_name,
                actions=plan.actions,
                status="rejected",
                final_state=WorldState.from_facts(task.initial_facts),
                metadata={"reason": "planner_rejected", "engine": "pddl_backed"},
            )

        domain = load_domain(domain_path)
        objects = object_types(task)
        state = set(task.initial_facts)
        steps: list[StepTrace] = []
        risk = False

        for index, action in enumerate(plan.actions):
            before = WorldState.from_facts(state)
            if action in task.safety_rules.forbidden_actions:
                risk = True
                if stop_on_safety:
                    violation = Violation(
                        type=ViolationType.SAFETY,
                        message=f"Forbidden action would be executed: {action}",
                        step_index=index,
                        action=action,
                    )
                    steps.append(failed_step(index, action, before, violation))
                    return failed_trace(task, plan, before, steps, violation, risk, domain_path)

            failure = self._apply_action(domain, objects, state, action)
            if failure:
                violation = Violation(
                    type=failure.violation_type,
                    message=failure.message,
                    step_index=index,
                    action=action,
                    missing_preconditions=failure.missing_preconditions,
                    details=failure.details or {},
                )
                steps.append(failed_step(index, action, before, violation))
                return failed_trace(task, plan, before, steps, violation, risk, domain_path)

            after = WorldState.from_facts(state)
            if any(goal in after.facts for goal in task.safety_rules.forbidden_goal_facts):
                risk = True
            steps.append(
                StepTrace(
                    index=index,
                    action=action,
                    before=before.to_list(),
                    after=after.to_list(),
                )
            )

        final_state = WorldState.from_facts(state)
        return ExecutionTrace(
            task_id=task.id,
            planner_name=plan.planner_name,
            actions=plan.actions,
            status="success",
            final_state=final_state,
            steps=tuple(steps),
            risk=risk,
            metadata={
                "engine": "pddl_backed",
                "domain_path": str(domain_path),
                "goal_satisfied": task.goal.is_satisfied(final_state),
            },
        )

    def _apply_action(
        self,
        domain: PDDLDomain,
        objects: dict[str, str],
        state: set[str],
        action: str,
    ) -> PDDLExecutionFailure | None:
        call = parse_call(action)
        schema = domain.actions.get(call.name)
        if schema is None:
            return PDDLExecutionFailure(
                ViolationType.HALLUCINATION,
                f"Action is not declared in PDDL domain: {call.name}",
                details={"action_name": call.name},
            )
        if len(call.args) != len(schema.parameters):
            return PDDLExecutionFailure(
                ViolationType.ACTION_ARG_NUM,
                f"Action {call.name} expects {len(schema.parameters)} arguments, got {len(call.args)}.",
                details={"expected": len(schema.parameters), "actual": len(call.args)},
            )

        bindings = dict(zip((param.name for param in schema.parameters), call.args, strict=True))
        for param, arg in zip(schema.parameters, call.args, strict=True):
            object_type = objects.get(arg)
            if object_type is None:
                return PDDLExecutionFailure(
                    ViolationType.HALLUCINATION,
                    f"Action argument is not an object in the PDDL problem: {arg}",
                    details={"argument": arg},
                )
            if not type_compatible(object_type, param.type_name):
                return PDDLExecutionFailure(
                    ViolationType.AFFORDANCE,
                    f"Object {arg} has type {object_type}, incompatible with {param.type_name}.",
                    missing_preconditions=(f"type({arg}, {param.type_name})",),
                    details={"argument": arg, "object_type": object_type, "expected_type": param.type_name},
                )

        context = PDDLContext(objects=objects, state=state, bindings=bindings)
        if not eval_condition(schema.precondition, context):
            return PDDLExecutionFailure(
                ViolationType.MISSING_STEP,
                f"Unsatisfied PDDL preconditions for action: {action}",
                missing_preconditions=tuple(collect_missing(schema.precondition, context)),
            )
        apply_effect(schema.effect, context)
        return None


@dataclass
class PDDLContext:
    objects: dict[str, str]
    state: set[str]
    bindings: dict[str, str]

    def with_bindings(self, extra: dict[str, str]) -> "PDDLContext":
        merged = dict(self.bindings)
        merged.update(extra)
        return PDDLContext(objects=self.objects, state=self.state, bindings=merged)


@lru_cache(maxsize=8)
def load_domain(path: str | Path) -> PDDLDomain:
    domain_path = Path(path)
    return parse_domain(domain_path.read_text(encoding="utf-8", errors="replace"))


def parse_domain(text: str) -> PDDLDomain:
    expr = parse_pddl(text)
    if not isinstance(expr, list) or not expr or expr[0] != "define":
        raise ValueError("PDDL domain must start with (define ...).")
    domain_name = ""
    actions: dict[str, PDDLActionSchema] = {}
    for section in expr[1:]:
        if not isinstance(section, list) or not section:
            continue
        if section[0] == "domain" and len(section) > 1:
            domain_name = str(section[1])
        elif section[0] == ":action" and len(section) > 1:
            action = parse_action_schema(section)
            actions[action.name] = action
    return PDDLDomain(name=domain_name, actions=actions)


def parse_action_schema(section: list[PDDLExpr]) -> PDDLActionSchema:
    name = str(section[1])
    fields: dict[str, PDDLExpr] = {}
    index = 2
    while index < len(section):
        key = str(section[index])
        if key.startswith(":") and index + 1 < len(section):
            fields[key] = section[index + 1]
            index += 2
        else:
            index += 1
    params_expr = fields.get(":parameters", [])
    parameters = parse_typed_parameters(params_expr if isinstance(params_expr, list) else [])
    return PDDLActionSchema(
        name=name,
        parameters=parameters,
        precondition=fields.get(":precondition", []),
        effect=fields.get(":effect", []),
    )


def parse_typed_parameters(expr: list[PDDLExpr]) -> tuple[PDDLParameter, ...]:
    flat = [str(item) for item in expr if isinstance(item, str)]
    parameters: list[PDDLParameter] = []
    pending: list[str] = []
    index = 0
    while index < len(flat):
        token = flat[index]
        if token == "-":
            if index + 1 >= len(flat):
                raise ValueError("PDDL parameter type marker '-' without type.")
            type_name = flat[index + 1]
            parameters.extend(PDDLParameter(name=name, type_name=type_name) for name in pending)
            pending = []
            index += 2
        else:
            pending.append(token)
            index += 1
    parameters.extend(PDDLParameter(name=name, type_name="object") for name in pending)
    return tuple(parameters)


def eval_condition(expr: PDDLExpr, context: PDDLContext) -> bool:
    if expr == [] or expr == "()":
        return True
    if isinstance(expr, str):
        return bool(expr)
    if not expr:
        return True
    op = str(expr[0])
    if op == "and":
        return all(eval_condition(item, context) for item in expr[1:])
    if op == "or":
        return any(eval_condition(item, context) for item in expr[1:])
    if op == "not" and len(expr) == 2:
        return not eval_condition(expr[1], context)
    if op == "exists" and len(expr) >= 3:
        variables = parse_typed_parameters(expr[1] if isinstance(expr[1], list) else [])
        return any(
            eval_condition(expr[2], context.with_bindings(bindings))
            for bindings in iter_quantifier_bindings(variables, context.objects)
        )
    if op == "forall" and len(expr) >= 3:
        variables = parse_typed_parameters(expr[1] if isinstance(expr[1], list) else [])
        return all(
            eval_condition(expr[2], context.with_bindings(bindings))
            for bindings in iter_quantifier_bindings(variables, context.objects)
        )
    fact = ground_fact(expr, context.bindings)
    if fact.startswith("same_obj("):
        args = parse_call(fact).args
        return len(args) == 2 and args[0] == args[1]
    return fact in context.state


def collect_missing(expr: PDDLExpr, context: PDDLContext) -> list[str]:
    if expr == [] or expr == "()":
        return []
    if isinstance(expr, str):
        return []
    if not expr:
        return []
    op = str(expr[0])
    if op == "and":
        missing: list[str] = []
        for item in expr[1:]:
            missing.extend(collect_missing(item, context))
        return missing
    if op == "or":
        if eval_condition(expr, context):
            return []
        branches = [collect_missing(item, context) for item in expr[1:]]
        return min(branches, key=len) if branches else []
    if op == "not" and len(expr) == 2:
        if eval_condition(expr, context):
            return []
        return [f"not({ground_fact(expr[1], context.bindings)})"]
    if op in {"exists", "forall"}:
        return [] if eval_condition(expr, context) else [stringify_expr(expr, context.bindings)]
    return [] if eval_condition(expr, context) else [ground_fact(expr, context.bindings)]


def apply_effect(expr: PDDLExpr, context: PDDLContext) -> None:
    adds, deletes = collect_effects(expr, context)
    for fact in deletes:
        context.state.discard(fact)
        context.state.add(f"not({fact})")
    for fact in adds:
        context.state.add(fact)
        context.state.discard(f"not({fact})")


def collect_effects(expr: PDDLExpr, context: PDDLContext) -> tuple[set[str], set[str]]:
    adds: set[str] = set()
    deletes: set[str] = set()
    _collect_effects(expr, context, adds, deletes)
    return adds, deletes


def _collect_effects(
    expr: PDDLExpr,
    context: PDDLContext,
    adds: set[str],
    deletes: set[str],
) -> None:
    if expr == [] or expr == "()":
        return
    if isinstance(expr, str):
        return
    if not expr:
        return
    op = str(expr[0])
    if op == "and":
        for item in expr[1:]:
            _collect_effects(item, context, adds, deletes)
        return
    if op == "not" and len(expr) == 2:
        fact = ground_fact(expr[1], context.bindings)
        deletes.add(fact)
        return
    if op == "when" and len(expr) == 3:
        if eval_condition(expr[1], context):
            _collect_effects(expr[2], context, adds, deletes)
        return
    if op == "forall" and len(expr) >= 3:
        variables = parse_typed_parameters(expr[1] if isinstance(expr[1], list) else [])
        for bindings in iter_quantifier_bindings(variables, context.objects):
            _collect_effects(expr[2], context.with_bindings(bindings), adds, deletes)
        return
    fact = ground_fact(expr, context.bindings)
    adds.add(fact)


def iter_quantifier_bindings(
    variables: tuple[PDDLParameter, ...],
    objects: dict[str, str],
) -> Iterable[dict[str, str]]:
    domains = [
        [
            object_name
            for object_name, object_type in objects.items()
            if type_compatible(object_type, variable.type_name)
        ]
        for variable in variables
    ]
    for values in product(*domains):
        yield dict(zip((variable.name for variable in variables), values, strict=True))


def ground_fact(expr: PDDLExpr, bindings: dict[str, str]) -> str:
    grounded = substitute_expr(expr, bindings)
    return format_fact(grounded)


def substitute_expr(expr: PDDLExpr, bindings: dict[str, str]) -> PDDLExpr:
    if isinstance(expr, str):
        return bindings.get(expr, expr)
    return [substitute_expr(item, bindings) for item in expr]


def stringify_expr(expr: PDDLExpr, bindings: dict[str, str]) -> str:
    grounded = substitute_expr(expr, bindings)
    if isinstance(grounded, str):
        return grounded
    return format_fact(grounded)


def object_types(task: Task) -> dict[str, str]:
    objects = task.metadata.get("objects", {})
    if not isinstance(objects, dict):
        return {}
    return {str(key): str(value) for key, value in objects.items()}


def type_compatible(object_type: str, expected_type: str) -> bool:
    if expected_type == "object":
        return True
    return object_type == expected_type


def failed_step(
    index: int,
    action: str,
    before: WorldState,
    violation: Violation,
) -> StepTrace:
    return StepTrace(
        index=index,
        action=action,
        before=before.to_list(),
        after=before.to_list(),
        violation=violation,
    )


def failed_trace(
    task: Task,
    plan: PlanCandidate,
    state: WorldState,
    steps: list[StepTrace],
    violation: Violation,
    risk: bool,
    domain_path: Path,
) -> ExecutionTrace:
    return ExecutionTrace(
        task_id=task.id,
        planner_name=plan.planner_name,
        actions=plan.actions,
        status="failed",
        final_state=state,
        steps=tuple(steps),
        violation=violation,
        risk=risk,
        metadata={"engine": "pddl_backed", "domain_path": str(domain_path)},
    )
