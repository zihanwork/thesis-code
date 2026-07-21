from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from embodied_gap.core.task_schema import Task

EAI_SOURCE_ROOT_ENV = "EAI_SOURCE_ROOT"
DEFAULT_EAI_RELATIVE_ROOT = Path("external/embodied-agent-interface")


def project_root() -> Path:
    """Return the checkout root without depending on the current directory."""

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "embodied_gap").is_dir():
            return parent
    return Path.cwd()


def portable_source_root(source_root: str | Path) -> str:
    """Serialize an EAI checkout as a portable project-relative hint."""

    root = Path(source_root).expanduser().resolve()
    try:
        return root.relative_to(project_root()).as_posix()
    except ValueError:
        # External checkouts should be supplied through EAI_SOURCE_ROOT when the
        # processed data is moved to another machine. Keep only a non-absolute
        # checkout hint in generated records.
        return root.name


def candidate_source_roots(metadata: dict[str, Any] | None = None) -> tuple[Path, ...]:
    """Return EAI checkout candidates in override-to-default order."""

    metadata = metadata or {}
    candidates: list[Path] = []

    env_root = os.environ.get(EAI_SOURCE_ROOT_ENV)
    if env_root:
        candidates.append(Path(env_root).expanduser())

    for key in ("source_root", "source_root_relative"):
        raw = metadata.get(key)
        if not isinstance(raw, str) or not raw or raw.startswith("${"):
            continue
        path = Path(raw).expanduser()
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend((project_root() / path, Path.cwd() / path))

    candidates.append(project_root() / DEFAULT_EAI_RELATIVE_ROOT)
    for anchor in (Path.cwd(), *Path.cwd().parents):
        candidates.append(anchor / DEFAULT_EAI_RELATIVE_ROOT)

    return tuple(_dedupe_paths(candidates))


def resolve_domain_path(task: Task) -> Path | None:
    """Resolve an EAI domain file even when stored absolute paths are stale."""

    metadata = task.metadata
    dataset = str(task.slots.get("dataset") or metadata.get("dataset") or "").lower()
    roots = candidate_source_roots(metadata)
    candidates: list[Path] = []

    raw_path = metadata.get("domain_pddl_path")
    if isinstance(raw_path, str) and raw_path:
        candidates.extend(_expand_path_hint(raw_path, roots))

    relative_path = metadata.get("domain_relative_path")
    if isinstance(relative_path, str) and relative_path:
        candidates.extend(_expand_path_hint(relative_path, roots))
        for root in roots:
            for dataset_root in dataset_root_candidates(root, dataset):
                candidates.append(dataset_root / Path(relative_path).name)

    for root in roots:
        candidates.extend(_standard_domain_paths(root, dataset))

    return _first_file(candidates)


def resolve_problem_path(task: Task) -> Path | None:
    """Resolve the source problem PDDL for audit and reproduction workflows."""

    metadata = task.metadata
    dataset = str(task.slots.get("dataset") or metadata.get("dataset") or "").lower()
    roots = candidate_source_roots(metadata)
    candidates: list[Path] = []

    for key in ("problem_pddl_path", "problem_relative_path"):
        raw_path = metadata.get(key)
        if isinstance(raw_path, str) and raw_path:
            candidates.extend(_expand_path_hint(raw_path, roots))

    source_relative = metadata.get("source_relative_path")
    if isinstance(source_relative, str) and source_relative:
        for root in roots:
            for dataset_root in dataset_root_candidates(root, dataset):
                candidates.append(dataset_root / source_relative)

    file_id = task.slots.get("file_id")
    task_family = task.slots.get("task_family")
    if file_id:
        filename = f"{file_id}.pddl"
        for root in roots:
            if dataset == "virtualhome" and task_family:
                candidates.append(
                    root
                    / "src"
                    / "virtualhome_eval"
                    / "resources"
                    / "virtualhome"
                    / "problem_pddl"
                    / task_family
                    / filename
                )

    return _first_file(candidates)


def dataset_root_candidates(source_root: Path, dataset: str) -> tuple[Path, ...]:
    candidates = [
        source_root / "src" / "virtualhome_eval" / "resources" / dataset,
        source_root / "virtualhome_eval" / "resources" / dataset,
        source_root / dataset,
        source_root,
    ]
    return tuple(_dedupe_paths(candidates))


def _standard_domain_paths(source_root: Path, dataset: str) -> tuple[Path, ...]:
    if dataset == "virtualhome":
        return (
            source_root
            / "src"
            / "virtualhome_eval"
            / "resources"
            / "virtualhome"
            / "virtualhome.pddl",
            source_root
            / "src"
            / "virtualhome_eval"
            / "resources"
            / "virtualhome"
            / "virtualhome_pd.pddl",
        )
    return ()


def _expand_path_hint(raw_path: str, roots: Iterable[Path]) -> list[Path]:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return [path]
    candidates = [project_root() / path, Path.cwd() / path]
    candidates.extend(root / path for root in roots)
    return candidates


def _first_file(candidates: Iterable[Path]) -> Path | None:
    for candidate in _dedupe_paths(candidates):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = str(path.expanduser())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(path.expanduser())
    return unique
