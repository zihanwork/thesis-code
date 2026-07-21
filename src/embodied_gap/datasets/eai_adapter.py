from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from embodied_gap.core.goal_schema import GoalSpec
from embodied_gap.core.task_schema import Task, dump_jsonl
from embodied_gap.datasets.resource_paths import portable_source_root

PDDLExpr = str | list["PDDLExpr"]


class EAIAdapterError(ValueError):
    """Raised when clean EAI source resources cannot be read consistently."""


@dataclass(frozen=True)
class PDDLProblem:
    problem_name: str
    domain_name: str
    objects: dict[str, str]
    init_facts: tuple[str, ...]
    goal_facts: tuple[str, ...]
    raw_text: str


@dataclass(frozen=True)
class EAIExportSummary:
    dataset: str
    task_count: int
    task_family_count: int
    gold_plan_count: int
    empty_goal_count: int
    train_count: int
    eval_count: int
    output_path: str
    raw_output_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "task_count": self.task_count,
            "task_family_count": self.task_family_count,
            "gold_plan_count": self.gold_plan_count,
            "empty_goal_count": self.empty_goal_count,
            "train_count": self.train_count,
            "eval_count": self.eval_count,
            "output_path": self.output_path,
            "raw_output_path": self.raw_output_path,
        }


class EmbodiedAgentInterfaceAdapter:
    """Convert clean EAI benchmark resources into project data files.

    This adapter intentionally reads only raw resources distributed with the
    EAI source tree, such as PDDL problems and gold PDDL plans. It does not
    read historical model outputs, diagnostics, or prior experiment folders.
    """

    SUPPORTED_DATASETS = ("virtualhome",)

    def __init__(self, source_root: str | Path) -> None:
        self.source_root = Path(source_root).expanduser().resolve()
        self.source_root_reference = portable_source_root(self.source_root)

    def load(self, dataset: str, train_ratio: float = 0.2) -> list[Task]:
        dataset_root = self._dataset_root(dataset)
        return self._load_dataset(dataset_root, dataset, train_ratio)

    def export(
        self,
        out_dir: str | Path,
        datasets: tuple[str, ...] = SUPPORTED_DATASETS,
        train_ratio: float = 0.2,
        write_raw: bool = True,
    ) -> dict[str, Any]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        all_rows: list[dict[str, Any]] = []
        summaries: dict[str, EAIExportSummary] = {}
        for dataset in datasets:
            tasks = self.load(dataset, train_ratio=train_ratio)
            dataset_rows = [task.to_dict() for task in tasks]
            dataset_file = out_path / f"{dataset}_tasks.jsonl"
            dump_jsonl(dataset_file, dataset_rows)
            all_rows.extend(dataset_rows)

            raw_file: Path | None = None
            if write_raw:
                raw_file = out_path / f"{dataset}_raw_pddl.jsonl"
                dump_jsonl(raw_file, [self._raw_record(task) for task in tasks])

            summaries[dataset] = EAIExportSummary(
                dataset=dataset,
                task_count=len(tasks),
                task_family_count=len({task.slots.get("task_family", "") for task in tasks}),
                gold_plan_count=sum(bool(task.gold_plan) for task in tasks),
                empty_goal_count=sum(not task.goal_facts for task in tasks),
                train_count=sum(task.split == "train" for task in tasks),
                eval_count=sum(task.split != "train" for task in tasks),
                output_path=str(dataset_file),
                raw_output_path=str(raw_file) if raw_file else None,
            )

        combined_file = out_path / "all_tasks.jsonl"
        dump_jsonl(combined_file, all_rows)
        manifest = {
            "source_root": self.source_root_reference,
            "source_root_env_override": "EAI_SOURCE_ROOT",
            "source_policy": "clean_eai_raw_resources_only",
            "excluded_paths": [
                "output",
                "output_norm_all",
                "output_single_norm",
                "diagnostics",
                "evaluate_results",
            ],
            "train_split": {
                "strategy": "deterministic_sha1",
                "train_ratio": train_ratio,
                "note": "EAI raw resources do not define a train/eval split for this thesis matrix.",
            },
            "combined_output_path": str(combined_file),
            "datasets": {key: value.to_dict() for key, value in summaries.items()},
        }
        manifest_file = out_path / "manifest.json"
        manifest_file.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return manifest

    def _load_dataset(self, dataset_root: Path, dataset: str, train_ratio: float) -> list[Task]:
        id2task = self._read_json(dataset_root / "id2task.json", default={})
        id2action = self._read_json(dataset_root / "id2action.json", default={})
        id2predicate = self._read_json(dataset_root / "id2predicate.json", default={})
        action_definitions = self._read_json(dataset_root / "gold_action.json", default={})
        gold_plans = self._read_json(dataset_root / "gold_pddl_plan.json", default={})
        success_tasks = set(self._read_json(dataset_root / "success_task.json", default=[]))
        failed_tasks = set(self._read_json(dataset_root / "failed_task.json", default=[]))
        problem_paths = sorted((dataset_root / "problem_pddl").rglob("*.pddl"))
        if not problem_paths:
            raise EAIAdapterError(f"No PDDL problems found under {dataset_root / 'problem_pddl'}")

        tasks: list[Task] = []
        for problem_path in problem_paths:
            problem_id = problem_path.stem
            parsed = parse_pddl_problem(problem_path.read_text(encoding="utf-8", errors="replace"))
            task_family = self._task_family(dataset, problem_path, parsed)
            gold_plan = tuple(format_plan_step(step) for step in gold_plans.get(problem_id, []))
            action_names = tuple(id2action.get(problem_id, []))
            predicate_names = tuple(id2predicate.get(problem_id, []))
            allowed_actions = tuple(sorted(action_definitions.keys()))
            instruction = self._instruction(problem_id, task_family, parsed, id2task)
            split = assign_split(dataset, problem_id, train_ratio)
            source_rel = problem_path.relative_to(dataset_root)
            domain_path = dataset_root / "virtualhome.pddl"
            problem_root_rel = problem_path.relative_to(self.source_root)
            domain_root_rel = domain_path.relative_to(self.source_root)
            tags = (
                "eai",
                dataset,
                "pddl",
                f"family:{task_family}",
                "has_gold_plan" if gold_plan else "missing_gold_plan",
                "empty_goal" if not parsed.goal_facts else "state_goal",
            )
            metadata = {
                "benchmark": "Embodied Agent Interface",
                "dataset": dataset,
                "problem_name": parsed.problem_name,
                "domain_name": parsed.domain_name,
                "source_root": self.source_root_reference,
                "source_relative_path": str(source_rel),
                "problem_relative_path": problem_root_rel.as_posix(),
                "domain_relative_path": domain_root_rel.as_posix(),
                "domain_pddl_path": domain_root_rel.as_posix(),
                "object_count": len(parsed.objects),
                "objects": parsed.objects,
                "init_fact_count": len(parsed.init_facts),
                "goal_fact_count": len(parsed.goal_facts),
                "action_names": list(action_names),
                "predicate_names": list(predicate_names),
                "success_task": problem_id in success_tasks,
                "failed_task": problem_id in failed_tasks,
                "adapter_policy": "raw_pddl_and_gold_plan_only",
                "executor_status": "pddl_semantics_not_flattened",
            }
            tasks.append(
                Task(
                    id=f"eai_{dataset}__{clean_id(task_family)}__{clean_id(problem_id)}",
                    split=split,
                    instruction=instruction,
                    tags=tags,
                    slots={
                        "dataset": dataset,
                        "task_family": task_family,
                        "file_id": problem_id,
                        "domain": parsed.domain_name,
                    },
                    initial_facts=parsed.init_facts,
                    goal=GoalSpec(parsed.goal_facts),
                    allowed_actions=allowed_actions,
                    gold_plan=gold_plan,
                    action_model={},
                    source=f"eai_raw_{dataset}",
                    metadata=metadata,
                )
            )
        return tasks

    def _dataset_root(self, dataset: str) -> Path:
        if dataset not in self.SUPPORTED_DATASETS:
            raise EAIAdapterError(
                f"Unsupported EAI dataset '{dataset}'. Expected one of {self.SUPPORTED_DATASETS}."
            )
        if any(part in {"output", "output_norm_all", "output_single_norm", "diagnostics"} for part in self.source_root.parts):
            raise EAIAdapterError("Refusing to import from historical output/diagnostics directories.")

        candidates = [
            self.source_root / "src" / "virtualhome_eval" / "resources" / dataset,
            self.source_root / "virtualhome_eval" / "resources" / dataset,
            self.source_root / dataset,
            self.source_root,
        ]
        for candidate in candidates:
            if (candidate / "problem_pddl").exists() and (candidate / "id2task.json").exists():
                return candidate
        raise EAIAdapterError(
            f"Could not locate EAI {dataset} resources below {self.source_root}. "
            "Expected src/virtualhome_eval/resources/<dataset>/problem_pddl."
        )

    def _raw_record(self, task: Task) -> dict[str, Any]:
        dataset_root = self._dataset_root(task.slots["dataset"])
        raw_path = dataset_root / task.metadata["source_relative_path"]
        return {
            "id": task.id,
            "dataset": task.slots["dataset"],
            "task_family": task.slots["task_family"],
            "file_id": task.slots["file_id"],
            "instruction": task.instruction,
            "gold_plan": list(task.gold_plan),
            "pddl": raw_path.read_text(encoding="utf-8", errors="replace"),
        }

    def _task_family(self, dataset: str, problem_path: Path, parsed: PDDLProblem) -> str:
        if dataset == "virtualhome":
            return problem_path.parent.name
        return parsed.problem_name

    def _instruction(
        self,
        problem_id: str,
        task_family: str,
        parsed: PDDLProblem,
        id2task: dict[str, Any],
    ) -> str:
        label = str(id2task.get(problem_id) or task_family or parsed.problem_name)
        if label == problem_id:
            label = parsed.problem_name
        return humanize(label)

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))


def parse_pddl_problem(text: str) -> PDDLProblem:
    expr = parse_pddl(text)
    if not isinstance(expr, list) or not expr or expr[0] != "define":
        raise EAIAdapterError("PDDL problem must start with (define ...).")

    problem_name = ""
    domain_name = ""
    objects: dict[str, str] = {}
    init_facts: tuple[str, ...] = ()
    goal_facts: tuple[str, ...] = ()

    for section in expr[1:]:
        if not isinstance(section, list) or not section:
            continue
        head = section[0]
        if head == "problem" and len(section) > 1:
            problem_name = str(section[1])
        elif head == ":domain" and len(section) > 1:
            domain_name = str(section[1])
        elif head == ":objects":
            objects = parse_typed_objects(section[1:])
        elif head == ":init":
            init_facts = tuple(format_fact(item) for item in section[1:])
        elif head == ":goal":
            goal_facts = tuple(format_fact(item) for item in unwrap_goal(section[1:]))

    return PDDLProblem(
        problem_name=problem_name,
        domain_name=domain_name,
        objects=objects,
        init_facts=init_facts,
        goal_facts=goal_facts,
        raw_text=text,
    )


def parse_pddl(text: str) -> PDDLExpr:
    tokens = tokenize_pddl(text)
    if not tokens:
        raise EAIAdapterError("Empty PDDL text.")
    expr, index = parse_expr(tokens, 0)
    if index != len(tokens):
        raise EAIAdapterError("Unexpected trailing PDDL tokens.")
    return expr


def tokenize_pddl(text: str) -> list[str]:
    no_comments = re.sub(r";[^\n]*", "", text)
    return re.findall(r"\(|\)|[^\s()]+", no_comments)


def parse_expr(tokens: list[str], index: int) -> tuple[PDDLExpr, int]:
    if index >= len(tokens):
        raise EAIAdapterError("Unexpected end of PDDL.")
    token = tokens[index]
    if token == "(":
        index += 1
        values: list[PDDLExpr] = []
        while index < len(tokens) and tokens[index] != ")":
            value, index = parse_expr(tokens, index)
            values.append(value)
        if index >= len(tokens):
            raise EAIAdapterError("Unclosed PDDL expression.")
        return values, index + 1
    if token == ")":
        raise EAIAdapterError("Unexpected ')' in PDDL.")
    return token, index + 1


def parse_typed_objects(tokens: list[PDDLExpr]) -> dict[str, str]:
    objects: dict[str, str] = {}
    pending: list[str] = []
    index = 0
    flat = [str(token) for token in tokens if isinstance(token, str)]
    while index < len(flat):
        token = flat[index]
        if token == "-":
            if index + 1 >= len(flat):
                raise EAIAdapterError("Object type marker '-' without type.")
            object_type = flat[index + 1]
            for name in pending:
                objects[name] = object_type
            pending = []
            index += 2
        else:
            pending.append(token)
            index += 1
    for name in pending:
        objects[name] = "object"
    return objects


def unwrap_goal(goal_items: list[PDDLExpr]) -> list[PDDLExpr]:
    if not goal_items:
        return []
    if len(goal_items) == 1 and isinstance(goal_items[0], list):
        root = goal_items[0]
        if root and root[0] == "and":
            return list(root[1:])
        return [root]
    return goal_items


def format_fact(expr: PDDLExpr) -> str:
    if isinstance(expr, str):
        return expr
    if not expr:
        return "()"
    name = str(expr[0])
    if name == "not" and len(expr) == 2:
        return f"not({format_fact(expr[1])})"
    args = [format_fact(item) if isinstance(item, list) else str(item) for item in expr[1:]]
    return f"{name}({', '.join(args)})"


def format_plan_step(step: str) -> str:
    parts = str(step).strip().split()
    if not parts:
        return "noop()"
    action_name = parts[0]
    args = parts[1:]
    if not args:
        return f"{action_name}()"
    return f"{action_name}({', '.join(args)})"


def assign_split(dataset: str, problem_id: str, train_ratio: float) -> str:
    if train_ratio <= 0:
        return "eval"
    if train_ratio >= 1:
        return "train"
    digest = hashlib.sha1(f"{dataset}:{problem_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10000
    return "train" if bucket < int(train_ratio * 10000) else "eval"


def humanize(label: str) -> str:
    words = re.sub(r"[_\-]+", " ", label).strip()
    return re.sub(r"\s+", " ", words)


def clean_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    return cleaned or "unknown"
