from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def verify_final_protocol(
    protocol_path: str | Path,
    *,
    repo_root: str | Path = ".",
    require_git_tag: bool = True,
    require_clean_worktree: bool = True,
    require_unrun: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    protocol_file = _resolve(root, Path(protocol_path))
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))

    artifact_checks = []
    for artifact in protocol.get("artifacts", []):
        path = _resolve(root, Path(artifact["path"]))
        actual = _sha256(path) if path.exists() else None
        artifact_checks.append(
            {
                "role": artifact.get("role"),
                "path": artifact["path"],
                "exists": path.exists(),
                "expected_sha256": artifact["sha256"],
                "actual_sha256": actual,
                "valid": actual == artifact["sha256"],
            }
        )

    output_checks = []
    for experiment in protocol.get("experiments", []):
        output_root = _resolve(root, Path(experiment["output_root"]))
        index_path = output_root / "run_index.jsonl"
        output_checks.append(
            {
                "id": experiment["id"],
                "output_root": experiment["output_root"],
                "run_index_exists": index_path.exists(),
                "unrun": not index_path.exists(),
            }
        )

    git = _git_state(root, protocol.get("required_git_tag"))
    artifacts_valid = all(item["valid"] for item in artifact_checks)
    outputs_valid = all(item["unrun"] for item in output_checks)
    git_tag_valid = bool(git["tag_exists"] and git["tag_commit"] == git["commit"])
    clean_valid = not git["dirty"]
    valid = (
        artifacts_valid
        and (git_tag_valid or not require_git_tag)
        and (clean_valid or not require_clean_worktree)
        and (outputs_valid or not require_unrun)
    )
    return {
        "schema_version": 1,
        "protocol_id": protocol.get("protocol_id"),
        "protocol_path": _display_path(root, protocol_file),
        "valid": valid,
        "requirements": {
            "require_git_tag": require_git_tag,
            "require_clean_worktree": require_clean_worktree,
            "require_unrun": require_unrun,
        },
        "git": git,
        "artifacts_valid": artifacts_valid,
        "artifact_checks": artifact_checks,
        "outputs_unrun": outputs_valid,
        "output_checks": output_checks,
        "totals": protocol.get("totals", {}),
    }


def _git_state(root: Path, tag: str | None) -> dict[str, Any]:
    commit = _git(root, "rev-parse", "HEAD")
    dirty = bool(_git(root, "status", "--porcelain"))
    tag_commit = None
    if tag:
        result = subprocess.run(
            ["git", "rev-list", "-n", "1", tag],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            tag_commit = result.stdout.strip() or None
    return {
        "commit": commit,
        "dirty": dirty,
        "required_tag": tag,
        "tag_exists": tag_commit is not None,
        "tag_commit": tag_commit,
    }


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
