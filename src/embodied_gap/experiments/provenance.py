from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
from typing import Any

from embodied_gap.datasets.resource_paths import project_root
from embodied_gap.llm.prompts import PLANNING_PROMPT_VERSION, REPAIR_PROMPT_VERSION


MANIFEST_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_fingerprint(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    source = Path(path)
    payload: dict[str, Any] = {
        "path": display_path(source),
        "exists": source.is_file(),
    }
    if not source.is_file():
        return payload
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    payload.update({"sha256": digest.hexdigest(), "bytes": size})
    return payload


def taskset_fingerprint(path: str | Path) -> dict[str, Any]:
    payload = file_fingerprint(path) or {"path": display_path(Path(path)), "exists": False}
    source = Path(path)
    if not source.is_file():
        return payload

    task_ids: list[str] = []
    eval_task_ids: list[str] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_id = str(row["id"])
            task_ids.append(task_id)
            if row.get("split", "eval") != "train":
                eval_task_ids.append(task_id)
    payload.update(
        {
            "task_count": len(task_ids),
            "evaluation_task_count": len(eval_task_ids),
            "task_ids": task_ids,
            "evaluation_task_ids": eval_task_ids,
            "task_ids_sha256": canonical_sha256(task_ids),
            "evaluation_task_ids_sha256": canonical_sha256(eval_task_ids),
        }
    )
    return payload


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(target)


@dataclass
class RunContext:
    run_id: str
    name: str
    output_dir: Path
    manifest: dict[str, Any]
    index_path: Path | None = None

    @classmethod
    def create(
        cls,
        base_output_dir: str | Path,
        *,
        name: str,
        config: dict[str, Any],
        tasks_path: str | Path,
        retrieval_examples_path: str | Path | None = None,
        models: list[dict[str, Any]] | None = None,
    ) -> "RunContext":
        base = Path(base_output_dir)
        base.mkdir(parents=True, exist_ok=True)
        config_hash = canonical_sha256(config)
        for _ in range(10):
            run_id = make_run_id(config_hash)
            output_dir = base / run_id
            try:
                output_dir.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            context = cls._build(
                run_id=run_id,
                name=name,
                output_dir=output_dir,
                config=config,
                tasks_path=tasks_path,
                retrieval_examples_path=retrieval_examples_path,
                models=models,
                index_path=base / "run_index.jsonl",
            )
            context.write()
            return context
        raise FileExistsError(f"Could not allocate a unique run directory below {base}")

    @classmethod
    def create_at(
        cls,
        output_dir: str | Path,
        *,
        run_id: str,
        name: str,
        config: dict[str, Any],
        tasks_path: str | Path,
        retrieval_examples_path: str | Path | None = None,
        models: list[dict[str, Any]] | None = None,
        parent_run_id: str | None = None,
    ) -> "RunContext":
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=False)
        context = cls._build(
            run_id=run_id,
            name=name,
            output_dir=target,
            config=config,
            tasks_path=tasks_path,
            retrieval_examples_path=retrieval_examples_path,
            models=models,
            parent_run_id=parent_run_id,
        )
        context.write()
        return context

    @classmethod
    def _build(
        cls,
        *,
        run_id: str,
        name: str,
        output_dir: Path,
        config: dict[str, Any],
        tasks_path: str | Path,
        retrieval_examples_path: str | Path | None,
        models: list[dict[str, Any]] | None = None,
        parent_run_id: str | None = None,
        index_path: Path | None = None,
    ) -> "RunContext":
        root = project_root()
        prompt_path = root / "src" / "embodied_gap" / "llm" / "prompts.py"
        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "name": name,
            "status": "running",
            "started_at": utc_now(),
            "completed_at": None,
            "output_dir": display_path(output_dir),
            "config": {
                "sha256": canonical_sha256(config),
                "payload": config,
            },
            "code": git_provenance(root),
            "environment": {
                "python_version": platform.python_version(),
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "uv_lock": file_fingerprint(root / "uv.lock"),
            },
            "data": {
                "tasks": taskset_fingerprint(tasks_path),
                "retrieval_examples": (
                    taskset_fingerprint(retrieval_examples_path)
                    if retrieval_examples_path
                    else None
                ),
            },
            "prompts": {
                "template": file_fingerprint(prompt_path),
                "versions": {
                    "planning": PLANNING_PROMPT_VERSION,
                    "repair": REPAIR_PROMPT_VERSION,
                },
            },
            "models": models or [],
            "telemetry": {},
            "results": {},
            "error": None,
        }
        return cls(run_id, name, output_dir, manifest, index_path=index_path)

    def write(self) -> None:
        atomic_write_json(self.output_dir / "run_manifest.json", self.manifest)

    def update(self, **fields: Any) -> None:
        self.manifest.update(fields)
        self.write()

    def finalize(
        self,
        status: str,
        *,
        results: dict[str, Any] | None = None,
        telemetry: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.manifest["status"] = status
        self.manifest["completed_at"] = utc_now()
        if results is not None:
            self.manifest["results"] = results
        if telemetry is not None:
            self.manifest["telemetry"] = telemetry
        if error is not None:
            self.manifest["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        self.write()
        if self.index_path is not None:
            self._append_index()

    def _append_index(self) -> None:
        assert self.index_path is not None
        entry = {
            "run_id": self.run_id,
            "name": self.name,
            "status": self.manifest["status"],
            "started_at": self.manifest["started_at"],
            "completed_at": self.manifest["completed_at"],
            "output_dir": display_path(self.output_dir),
            "config_sha256": self.manifest["config"]["sha256"],
        }
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def make_run_id(config_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    entropy = secrets.token_hex(8)
    suffix = hashlib.sha256(f"{config_hash}:{timestamp}:{entropy}".encode("utf-8")).hexdigest()[:8]
    return f"{timestamp}_{suffix}"


def git_provenance(root: Path) -> dict[str, Any]:
    commit = git_output(root, "rev-parse", "HEAD")
    branch = git_output(root, "rev-parse", "--abbrev-ref", "HEAD")
    status = git_output(root, "status", "--porcelain", "--untracked-files=no")
    index_text = git_output(root, "ls-files", "--stage") or ""
    submodules: dict[str, dict[str, Any]] = {}
    for line in index_text.splitlines():
        try:
            metadata, relative_path = line.split("\t", 1)
            mode, pinned_commit, _stage = metadata.split()
        except ValueError:
            continue
        if mode != "160000":
            continue
        checkout = root / relative_path
        checked_out_commit = git_output(checkout, "rev-parse", "HEAD")
        checkout_status = git_output(
            checkout,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
        if checked_out_commit is None:
            state = "missing"
        elif checked_out_commit != pinned_commit or checkout_status:
            state = "modified"
        else:
            state = "clean"
        submodules[relative_path] = {
            "pinned_commit": pinned_commit,
            "checked_out_commit": checked_out_commit,
            "state": state,
        }
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "submodules": submodules,
    }


def git_output(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def display_path(path: Path) -> str:
    path = path.expanduser()
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except (ValueError, FileNotFoundError):
        return str(path)
