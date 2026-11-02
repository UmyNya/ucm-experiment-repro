#!/usr/bin/env python3
"""Snapshot selected historical evidence; never execute a historical controller."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

DEFAULT_HISTORY_MAP = Path(__file__).resolve().parents[1] / "config/history-sources.example.json"


def load_history(path, workspace):
    """Read explicit relative source directories without embedding machine history."""
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or not value:
        raise ValueError("History map must be a nonempty object")
    workspace = Path(workspace).resolve()
    for key, name in value.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError("History map keys must be simple names")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("History source must be a relative directory")
        directory = Path(name)
        resolved = (workspace / directory).resolve()
        if directory.is_absolute() or resolved == workspace or workspace not in resolved.parents:
            raise ValueError("History source must stay inside the workspace")
    return value


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo, *args):
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(repo), *args],
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"},
            text=True, capture_output=True, timeout=35,
        )
        # URLs/credential diagnostics are deliberately not included.
        return {"returncode": result.returncode, "stdout": result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "error_type": type(error).__name__}


def repository_state(path, remote=False):
    result = {"path": str(path)}
    for name, args in {
        "head": ("rev-parse", "HEAD"),
        "status": ("status", "--porcelain=v1", "--untracked-files=all"),
        "branch": ("branch", "--show-current"),
        "upstream": ("rev-parse", "--abbrev-ref", "@{upstream}"),
        "local_upstream_distance": ("rev-list", "--left-right", "--count", "HEAD...@{upstream}"),
    }.items():
        result[name] = git(path, *args)
    result["tracked_and_untracked_clean"] = (
        result["status"].get("returncode") == 0 and not result["status"]["stdout"])
    result["remote_checked"] = False
    if remote and result["branch"].get("stdout"):
        branch = result["branch"]["stdout"]
        remote_name = git(path, "config", "--get", f"branch.{branch}.remote").get("stdout")
        merge_ref = git(path, "config", "--get", f"branch.{branch}.merge").get("stdout")
        if remote_name and merge_ref:
            reply = git(path, "ls-remote", "--exit-code", remote_name, merge_ref)
            rows = [line.split() for line in reply.get("stdout", "").splitlines()]
            heads = [row[0] for row in rows if len(row) == 2 and row[1] == merge_ref]
            result["remote_checked"] = reply.get("returncode") == 0 and len(heads) == 1
            result["remote_query_returncode"] = reply.get("returncode")
            if result["remote_checked"]:
                result["remote_head"] = heads[0]
                result["matches_remote"] = heads[0] == result["head"].get("stdout")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--ucm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; must not exist")
    parser.add_argument("--history-map", type=Path, default=DEFAULT_HISTORY_MAP,
                        help="JSON mapping experiment names to directories relative to --workspace")
    parser.add_argument("--check-remote", action="store_true", help="Read-only ls-remote; no fetch/reset")
    args = parser.parse_args()
    os.umask(0o077)
    workspace, output = args.workspace.resolve(), args.output.resolve()
    history = load_history(args.history_map, workspace)
    protected = [(workspace / p).resolve() for p in history.values()] + [
        workspace / "aisbench-ucm-repro", workspace / "swe-rebench-ucm-bench", args.ucm.resolve()]
    if any(output == p or p in output.parents for p in protected):
        parser.error("Output must be outside every protected source directory")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source_workspace": str(workspace), "files": [], "repositories": {},
                "scope": "Selected protocols/configurations/reports only; not a full source or weight archive"}
    try:
        for key, directory in history.items():
            source = workspace / directory
            names = ["REPORT.md", "REPRODUCE.md", "experiment.json", "arm-specifications.json",
                     "ALL_CONFIGURATIONS.json", "configs/all-actual-configurations.json",
                     "configs/matrix.json", "performance.csv", "results.csv", "comparison.csv",
                     "pipeline-costs.csv"]
            names += [str(p.relative_to(source)) for p in sorted((source / "configs").glob("*.json"))]
            for name in sorted(set(names)):
                path = source / name
                if not path.is_file():
                    continue
                before = path.stat()
                sha = digest(path)
                destination = output / "reference" / key / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                after = path.stat()
                unchanged = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                if not unchanged or digest(destination) != sha:
                    raise RuntimeError(f"Source changed while reading: {path}")
                manifest["files"].append({"source": str(path), "copy": str(destination.relative_to(output)),
                                          "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
                                          "sha256": sha})
        for key, path in {"ucm": args.ucm.resolve(),
                          "aisbench": workspace / "aisbench-ucm-repro",
                          "swe": workspace / "swe-rebench-ucm-bench"}.items():
            manifest["repositories"][key] = repository_state(path, args.check_remote)
        manifest["capture_complete"] = True
    except Exception as error:
        manifest["capture_complete"] = False
        manifest["error"] = str(error)
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
