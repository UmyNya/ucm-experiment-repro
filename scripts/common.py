"""Small shared helpers; all writable artifacts belong to a new run directory."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import uuid

BUNDLE = Path(__file__).resolve().parents[1]


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def protected_paths(config):
    paths = config["paths"]
    return [Path(paths[name]).resolve() for name in ["ucm_repository", "aisbench_repository",
            "swe_repository", "model", "tokenizer"]]


def safe_output(path, config):
    path = Path(path).resolve()
    for protected in protected_paths(config):
        if within(path, protected):
            raise ValueError(f"Output overlaps protected source/model path: {path}")
    # User-designated historical directories under the original workspace.
    for source in (BUNDLE / "evidence/source-audit/manifest.json",):
        if source.is_file():
            for item in read_json(source)["files"]:
                p = Path(item["source"])
                historical = next((x for x in p.parents if x.name.startswith("experiment_")), None)
                if historical and within(path, historical):
                    raise ValueError(f"Output overlaps historical experiment: {path}")
    return path


def new_run(config, profile, purpose="run"):
    root = safe_output(config["paths"]["output_root"], config)
    root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = root / f"experiment_{profile}_{purpose}_{stamp}_{uuid.uuid4().hex[:8]}"
    run.mkdir(mode=0o700)
    code = run / "executed-code"
    code.mkdir()
    manifest = {}
    for source in (BUNDLE / "scripts").glob("*.py"):
        shutil.copyfile(source, code / source.name)
        manifest[source.name] = sha256(source)
    save_json(run / "executed-code-sha256.json", manifest)
    for entry in config.get("configuration_sources", []):
        source = Path(entry["path"])
        if sha256(source) != entry["sha256"]:
            raise RuntimeError(f"Configuration changed while creating run: {source}")
        target = run / "config-sources" / (entry["section"] + ".json")
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
    save_json(run / "configuration.json", config)
    return run


def http(base_url, path, body=None, timeout=45):
    payload = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base_url.rstrip("/") + path, data=payload,
                                 headers={"Content-Type": "application/json"})
    # Requests are local service control; do not route localhost through external proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else None
