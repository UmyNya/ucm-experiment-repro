"""Native physical-drain checks around a measurement; zero is never assumed for missing data."""
from __future__ import annotations

import time
from common import http, save_json

LIFECYCLE = {"unfinished_dump_tasks", "pending_load_tasks", "block_dump_batch_pending",
             "early_jobs", "early_request_mappings", "early_consumer_tasks", "early_active_consumers",
             "store_load_receipts", "consistency_load_contexts"}


def rpc(plan, method, args=(), timeout=60):
    reply = http(plan["base_url"], "/collective_rpc", {"method": method, "args": list(args), "timeout": timeout}, timeout + 30)
    rows = reply.get("results") if isinstance(reply, dict) else reply
    if not isinstance(rows, list) or not rows or not all(isinstance(x, dict) for x in rows):
        raise ValueError(f"Invalid collective RPC result for {method}")
    return rows


def drain_state(row):
    lifecycle = row["lifecycle"]
    if not LIFECYCLE.issubset(lifecycle):
        raise ValueError("Observer omitted required pending-work dimensions")
    counters, histograms = row["counters"], row["histogram_counts"]
    # After an empty service start there may be no histogram samples yet. A
    # collector-produced metric map and native task checks establish that case.
    if row.get("metrics_collected") is not True:
        raise ValueError("Drain requires a collected metrics snapshot")
    submitted = histograms.get("posix_dump_queue_wait_duration_ms", 0)
    completed = histograms.get("posix_dump_task_duration_ms", 0)
    written = counters.get("posix_h2s_bytes_total", 0)
    successful = counters.get("posix_dump_write_bytes_total", 0)
    errors = {key: value for key, value in counters.items() if value and any(
        part in key for part in ["errors_total", "failed_shards_total", "timeout_total", "queue_full_total",
                                "unhealthy_count_total", "passive_failures_total"])}
    native_complete = all(p["complete"] is True and p["sources_released"] is True for p in row["pending_dump_task_status"])
    stable = [submitted, completed, written, successful,
              counters.get("posix_dump_write_ops_total", 0), counters.get("posix_dump_full_block_ops_total", 0)]
    return {"pid": row["pid"], "signature": stable, "errors": errors,
            "physically_idle": native_complete and not any(lifecycle.values()) and submitted == completed and written == successful,
            "lifecycle": lifecycle}


def drain(plan, run, stage):
    deadline = time.monotonic() + plan["configuration"]["execution"]["drain_timeout_s"]
    history, previous = [], None
    while True:
        snapshots = rpc(plan, "bundle_snapshot", [True])
        states = sorted([drain_state(row) for row in snapshots], key=lambda x: x["pid"])
        history.append({"monotonic_ns": time.monotonic_ns(), "workers": states})
        save_json(run / (stage + "-drain-progress.json"), history)
        if any(s["errors"] for s in states):
            raise RuntimeError("Native I/O error observed during physical writeback drain")
        signature = [(s["pid"], s["signature"]) for s in states]
        if all(s["physically_idle"] for s in states) and signature == previous:
            reaped = rpc(plan, "bundle_reap_completed")
            save_json(run / (stage + "-reap.json"), reaped)
            if any(s["pending_after"] != 0 or s["retired_requests_after"] != 0 for s in reaped):
                raise RuntimeError("Physical completion did not clear connector receipts")
            final = rpc(plan, "bundle_snapshot", [True])
            if any(s["pending_dump_tasks"] or not drain_state(s)["physically_idle"] for s in final):
                raise RuntimeError("Work appeared after a supposedly stable drain")
            save_json(run / (stage + "-workers.json"), final)
            return final
        previous = signature
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Physical I/O drain timed out; see {stage}-drain-progress.json")
        time.sleep(1)
