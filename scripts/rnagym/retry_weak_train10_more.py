#!/usr/bin/env python3
"""Restart the remaining failed A-pilot train10 arms with audited archives."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.rnagym import mwu_retention_experiments as old
from scripts.rnagym import mwu_weak_experiments as weak

FAILED = {
    5: {
        "name": "mw_010_seed1234",
        "logged_epochs": 9,
        "checkpoints": [1, 2, 5],
        "reason": "UGE enforced the original 4h h_rt before validation/checkpoint completion",
    },
    9: {
        "name": "mw_001_seed1235",
        "logged_epochs": 1,
        "checkpoints": [],
        "reason": "checkpoint save failed while project file-count quota was exhausted",
    },
}


def validate_failed_state(out: Path, index: int) -> tuple[dict, Path]:
    old.verify(out)
    old.require(out, out / "preflight.json")
    plan = json.loads((out / "plan.json").read_text())
    spec = FAILED[index]
    entry = plan["entries"][index - 1]
    if entry["name"] != spec["name"]:
        raise RuntimeError(f"Unexpected A arm at task {index}: {entry['name']}")
    folder = out / "runs" / spec["name"]
    if not folder.is_dir() or (folder / "segment10_completed.json").exists():
        raise RuntimeError(f"Arm is missing or already successful: {folder}")
    rows = old.read_rows(folder / "components.tsv")
    epochs = [int(row["epoch"]) for row in rows]
    expected_epochs = list(range(1, spec["logged_epochs"] + 1))
    if epochs != expected_epochs:
        raise RuntimeError(f"Unexpected partial component log for {spec['name']}: {epochs}")
    checkpoints = sorted(
        int(p.name.removeprefix("epoch-"))
        for p in folder.glob("epoch-*")
        if p.name.removeprefix("epoch-").isdigit()
    )
    if checkpoints != spec["checkpoints"]:
        raise RuntimeError(f"Unexpected checkpoint inventory for {spec['name']}: {checkpoints}")
    return spec, folder


def restart(out: Path, index: int, gpu: int) -> None:
    spec, folder = validate_failed_state(out, index)
    archive = out / "runs" / "attempt1" / spec["name"]
    manifest = archive.parent / f"{spec['name']}.attempt1_manifest.json"
    if archive.exists() or manifest.exists():
        raise FileExistsError(f"Attempt archive already exists for {spec['name']}")
    files = []
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            files.append({
                "path": str(path.relative_to(folder)),
                "size": path.stat().st_size,
                "sha256": old.digest(path),
            })
    archive.parent.mkdir(parents=True, exist_ok=True)
    os.replace(folder, archive)
    old.write_json(manifest, {
        "success": True,
        "archived": old.now(),
        "arm": spec["name"],
        "logged_epochs": spec["logged_epochs"],
        "checkpoint_epochs": spec["checkpoints"],
        "reason": spec["reason"],
        "files": files,
        "source_sha256": old.digest(Path(__file__)),
    })
    print(f"Archived failed attempt for {spec['name']}; starting a fresh 10-epoch segment", flush=True)
    weak.train_arm(out, index - 1, gpu, end_epoch=10)
    marker = old.require(out, folder / "segment10_completed.json")
    if marker.get("optimizer_updates") != 31660:
        raise RuntimeError(f"Unexpected update count for repaired arm: {spec['name']}")
    print(f"Completed train10 segment: {spec['name']}", flush=True)


def submit(out: Path) -> None:
    old.verify(out)
    old.require(out, out / "preflight.json")
    record = out / "retry10b_submissions.json"
    submission_state = json.loads(record.read_text()) if record.exists() else {"submissions": []}
    submitted_tasks = {int(item["task"]) for item in submission_state["submissions"]}
    for index in FAILED:
        if index not in submitted_tasks:
            validate_failed_state(out, index)

    submissions = submission_state["submissions"]
    for index, spec in FAILED.items():
        if index in submitted_tasks:
            continue
        command = [
            "qsub", "-g", "tga-satolab-gtex", "-terse", "-cwd", "-j", "y",
            "-N", "mww_retry10b", "-l", "gpu_h=1", "-l", "h_rt=08:00:00",
            "-l", "hostname=!r22n6", "-o", str(out / "scheduler"),
            "-t", str(index), "-v", f"PROJECT_ROOT={ROOT},MWW_ROOT={out}",
            str(ROOT / "scripts/rnagym/qsub_mwu_weak_retry10_more.sh"),
        ]
        response = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
        job = response.split(".")[0]
        if not job.isdigit():
            raise RuntimeError(f"Unexpected qsub response for task {index}: {response}")
        submissions.append({
            "task": index,
            "arm": spec["name"],
            "job": job,
            "response": response,
            "submitted": old.now(),
            "command": command,
            "walltime": "08:00:00",
            "gpu_h": 1,
        })
        old.write_json(record, {"submissions": submissions})
    for item in submissions:
        print(f"Task {item['task']} ({item['arm']}): job {item['response']}", flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["submit", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    out = args.root.resolve()
    if args.action == "submit":
        submit(out)
    else:
        if args.index not in FAILED:
            raise ValueError(f"Unexpected retry task index: {args.index}")
        restart(out, args.index, args.gpu)


if __name__ == "__main__":
    main()
