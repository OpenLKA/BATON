#!/usr/bin/env python3
"""
run_all.py — Orchestrate all PassingCtrl baseline experiments.

Strategy for GPU utilization:
- Task1 (980K samples): runs alone on GPU (large model + data)
- Task2 + Task3 (57K + 71K): run simultaneously on the same GPU
  (each uses ~2-3GB VRAM, total fits in 16GB)
- CPU jobs (LR/XGB) run in background parallel with GPU

Usage:
  python3 run_all.py --phase 1              # Anchor results
  python3 run_all.py --phase 2              # Main tables
  python3 run_all.py --phase 3              # Supplementary
  python3 run_all.py --phase all            # Everything
  python3 run_all.py --phase 1 --dry-run    # Print commands only
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

BASELINE_DIR = Path(__file__).parent
RESULTS_DIR = BASELINE_DIR / "results"
SEEDS = [42, 123, 7]
TASKS = ["task1", "task2", "task3"]


def is_done(run_name):
    return (RESULTS_DIR / run_name / "results.json").exists()


# Modalities that load video features (heavy memory)
_VIDEO_MODALITIES = {"FV", "CV", "FV+CV", "Full-Struct+FV", "Full-Struct+CV",
                     "Full-Multimodal", "Full-All"}


def make_nn_cmd(task, modality, model, split, horizon, seed, extra_args=None):
    run_name = f"{task}_{modality}_{model}_{split}_h{horizon}_s{seed}"
    if is_done(run_name):
        return None, run_name
    cmd = [
        sys.executable, "train_nn.py",
        "--task", task, "--modality", modality, "--model", model,
        "--split", split, "--horizon", str(horizon), "--seed", str(seed),
    ]
    # Task1 always uses default 4 workers (runs alone, RouteCache is COW-shared)
    # Task2+Task3 get --num-workers 2 injected in run_grouped() when running in parallel
    if extra_args:
        cmd.extend(extra_args)
    return cmd, run_name


def make_classical_cmd(task, model, split, horizon, seed):
    run_name = f"{task}_Full-Struct+GPS_{model}_{split}_h{horizon}_s{seed}"
    if is_done(run_name):
        return None, run_name
    cmd = [
        sys.executable, "train_classical.py",
        "--task", task, "--model", model,
        "--split", split, "--horizon", str(horizon), "--seed", str(seed),
    ]
    return cmd, run_name


def run_single(cmd, name, dry_run=False):
    """Run a single GPU command, return (ok, elapsed)."""
    if cmd is None:
        print(f"  [SKIP] {name}")
        return True, 0
    if dry_run:
        print(f"  [DRY] {' '.join(cmd)}")
        return True, 0

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BASELINE_DIR))
    elapsed = time.time() - t0
    ok = result.returncode == 0
    print(f"  {'OK' if ok else 'FAILED'}: {name} ({elapsed:.0f}s)")
    return ok, elapsed


def run_parallel_gpu(cmd_pairs, dry_run=False):
    """Run multiple GPU commands in parallel (same GPU), wait for all."""
    procs = []
    for cmd, name in cmd_pairs:
        if cmd is None:
            print(f"  [SKIP] {name}")
            continue
        if dry_run:
            print(f"  [DRY‖] {' '.join(cmd)}")
            continue
        print(f"  [GPU‖] {name}")
        # Use parallel.log to avoid conflict with train_nn.py's own train.log FileHandler
        log_path = RESULTS_DIR / name / "parallel.log"
        (RESULTS_DIR / name).mkdir(parents=True, exist_ok=True)
        f = open(log_path, "w")
        p = subprocess.Popen(cmd, cwd=str(BASELINE_DIR), stdout=f, stderr=subprocess.STDOUT)
        procs.append((p, name, f))

    n_ok, n_fail = 0, 0
    for p, name, f in procs:
        p.wait()
        f.close()
        if p.returncode == 0:
            n_ok += 1
            print(f"  [DONE‖] {name}")
        else:
            n_fail += 1
            print(f"  [FAIL‖] {name}")
    return n_ok, n_fail


MAX_CPU_CONCURRENT = 3  # limit concurrent CPU jobs to prevent OOM


def run_cpu_background(cmds_with_names, dry_run=False):
    """Launch CPU commands with concurrency limit, return results."""
    queue = []
    for cmd, name in cmds_with_names:
        if cmd is None:
            print(f"  [SKIP] {name}")
            continue
        if dry_run:
            print(f"  [DRY/CPU] {' '.join(cmd)}")
            continue
        queue.append((cmd, name))

    if dry_run or not queue:
        return []

    # Run with concurrency limit
    n_ok, n_fail = 0, 0
    active = []

    def _reap():
        nonlocal n_ok, n_fail
        still_active = []
        for p, nm, f in active:
            if p.poll() is not None:
                f.close()
                if p.returncode == 0:
                    n_ok += 1
                    print(f"  [DONE] {nm}")
                else:
                    n_fail += 1
                    print(f"  [FAIL] {nm}")
            else:
                still_active.append((p, nm, f))
        return still_active

    for cmd, name in queue:
        # Wait until a slot opens
        while len(active) >= MAX_CPU_CONCURRENT:
            active = _reap()
            if len(active) >= MAX_CPU_CONCURRENT:
                time.sleep(2)

        print(f"  [BG] {name} ({len(active)+1}/{MAX_CPU_CONCURRENT} slots)")
        log_path = RESULTS_DIR / name / "bg.log"
        (RESULTS_DIR / name).mkdir(parents=True, exist_ok=True)
        f = open(log_path, "w")
        p = subprocess.Popen(cmd, cwd=str(BASELINE_DIR), stdout=f, stderr=subprocess.STDOUT)
        active.append((p, name, f))

    # Wait for remaining
    while active:
        active = _reap()
        if active:
            time.sleep(2)

    # Return empty list (results already collected)
    return (n_ok, n_fail)


def wait_bg(bg_result):
    """Compat wrapper — results already collected by run_cpu_background."""
    if isinstance(bg_result, tuple):
        return bg_result
    # Legacy: list of procs
    n_ok, n_fail = 0, 0
    for p, name, f in bg_result:
        p.wait()
        f.close()
        if p.returncode == 0:
            n_ok += 1
            print(f"  [DONE] {name}")
        else:
            n_fail += 1
            print(f"  [FAIL] {name}")
    return n_ok, n_fail


def run_grouped(groups, dry_run=False, phase_label=""):
    """Run experiment groups: task1 alone, then task2+task3 in parallel.

    Each group is a list of (cmd, name) tuples for [task1, task2, task3].
    """
    n_ok, n_fail, n_skip = 0, 0, 0
    total_groups = len(groups)

    for gi, group in enumerate(groups, 1):
        # Separate task1 (big) from task2+task3 (small, can run in parallel)
        task1_jobs = [(cmd, name) for cmd, name in group if name and "task1" in name]
        small_jobs = [(cmd, name) for cmd, name in group if name and "task1" not in name]

        label = group[0][1].replace("task1_", "").replace("task2_", "").replace("task3_", "")
        print(f"\n  ▸ [P{phase_label}] Group {gi}/{total_groups}: {label}")

        # Run task1 alone (uses most VRAM + RAM, full workers)
        for cmd, name in task1_jobs:
            ok, _ = run_single(cmd, name, dry_run)
            if cmd is None:
                n_skip += 1
            elif ok:
                n_ok += 1
            else:
                n_fail += 1

        # Run task2 + task3 in parallel on same GPU
        # Use --num-workers 2 each to limit memory (2 processes × 2 workers = 4 total)
        active = []
        for cmd, name in small_jobs:
            if cmd is None:
                print(f"  [SKIP] {name}")
                n_skip += 1
            else:
                # Inject --num-workers 3 for parallel safety (2 processes × 3 workers = 6 total)
                cmd_safe = cmd + ["--num-workers", "3"]
                active.append((cmd_safe, name))

        if active:
            if len(active) >= 2 and not dry_run:
                print(f"  Running {len(active)} small tasks in parallel on GPU (workers=3 each)...")
                ok, fail = run_parallel_gpu(active, dry_run)
                n_ok += ok
                n_fail += fail
            else:
                for cmd, name in active:
                    ok, _ = run_single(cmd, name, dry_run)
                    if ok:
                        n_ok += 1
                    else:
                        n_fail += 1

    return n_ok, n_fail, n_skip


# ═══════════════════════════════════════════════════════════
# PHASE DEFINITIONS (return grouped by modality+seed)
# ═══════════════════════════════════════════════════════════

def phase1_groups():
    """Anchor: Veh (CAN-only lower bound) + Full-All (upper bound) × 3 tasks × 3 seeds."""
    groups = []
    for seed in SEEDS:
        for modality in ["Veh", "Full-All"]:
            group = [make_nn_cmd(task, modality, "gru", "cross_driver", 3, seed)
                     for task in TASKS]
            groups.append(group)
    return groups, []


def phase2_groups():
    """Modality ablation (incremental) + model comparison.

    Ablation order:  FV → CV → FV+CV → FV+CV+CAN → FV+CV+CAN+GPS
    Full-All is already in Phase 1, so ablation covers the 4 sub-configs.
    """
    groups = []
    cpu_cmds = []

    # Incremental modality ablation
    ablation = ["FV", "CV", "FV+CV", "Full-Multimodal"]
    # Full-Multimodal = FV+CV+CAN (no GPS);  Full-All = FV+CV+CAN+GPS (Phase 1)
    for seed in SEEDS:
        for modality in ablation:
            group = [make_nn_cmd(task, modality, "gru", "cross_driver", 3, seed)
                     for task in TASKS]
            groups.append(group)

    # CAN-only baselines (no video) for reference
    for seed in SEEDS:
        for modality in ["Full-Struct", "Full-Struct+GPS"]:
            group = [make_nn_cmd(task, modality, "gru", "cross_driver", 3, seed)
                     for task in TASKS]
            groups.append(group)

    # Model comparison: TCN on Full-Struct+GPS
    for seed in SEEDS:
        group = [make_nn_cmd(task, "Full-Struct+GPS", "tcn", "cross_driver", 3, seed)
                 for task in TASKS]
        groups.append(group)

    # CPU jobs (LR, XGB on Full-Struct+GPS)
    for seed in SEEDS:
        for task in TASKS:
            cpu_cmds.append(make_classical_cmd(task, "lr", "cross_driver", 3, seed))
            cpu_cmds.append(make_classical_cmd(task, "xgb", "cross_driver", 3, seed))

    return groups, cpu_cmds


def phase3_groups():
    """Supplementary: horizon sensitivity + split comparison."""
    groups = []
    cpu_cmds = []

    # Horizon sensitivity (task2+task3 only, h=1,5)
    for seed in SEEDS:
        for h in [1, 5]:
            group = [make_nn_cmd(task, "Full-Struct+GPS", "gru", "cross_driver", h, seed)
                     for task in ["task2", "task3"]]
            groups.append(group)

    # Split comparison (random, cross_vehicle)
    for seed in SEEDS:
        for split in ["random", "cross_vehicle"]:
            group = [make_nn_cmd(task, "Full-Struct+GPS", "gru", split, 3, seed)
                     for task in TASKS]
            groups.append(group)

    return groups, cpu_cmds


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--phase", default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    phase_defs = {
        "1": ("Phase 1: Anchor (Veh CAN-only + Full-All)", phase1_groups),
        "2": ("Phase 2: Modality Ablation + Model Comparison", phase2_groups),
        "3": ("Phase 3: Supplementary (Horizon, Splits)", phase3_groups),
    }
    phase_keys = ["1", "2", "3"] if args.phase == "all" else [args.phase]

    t0 = time.time()
    total_ok, total_fail, total_skip = 0, 0, 0

    for pk in phase_keys:
        phase_label, phase_fn = phase_defs[pk]
        groups, cpu_cmds = phase_fn()

        n_gpu = sum(1 for g in groups for cmd, _ in g if cmd is not None)
        n_cpu = sum(1 for cmd, _ in cpu_cmds if cmd is not None)
        n_skip = (sum(1 for g in groups for cmd, _ in g if cmd is None)
                  + sum(1 for cmd, _ in cpu_cmds if cmd is None))

        print(f"\n{'━'*70}")
        print(f"  {phase_label}")
        print(f"  GPU: {n_gpu}  CPU: {n_cpu}  Skip: {n_skip}")
        print(f"{'━'*70}")

        if n_gpu + n_cpu == 0:
            print("  Nothing to run — all done.")
            total_skip += n_skip
            continue

        # Launch CPU jobs in background
        bg_procs = []
        if cpu_cmds:
            print(f"\n  Launching {n_cpu} CPU jobs in background...")
            bg_procs = run_cpu_background(cpu_cmds, args.dry_run)

        # Run GPU groups
        if groups:
            g_ok, g_fail, g_skip = run_grouped(groups, args.dry_run, phase_label=pk)
        else:
            g_ok, g_fail, g_skip = 0, 0, 0

        # Wait for CPU background jobs
        if bg_procs:
            print(f"\n  Waiting for CPU background jobs...")
            c_ok, c_fail = wait_bg(bg_procs)
        else:
            c_ok, c_fail = 0, 0

        p_ok = g_ok + c_ok
        p_fail = g_fail + c_fail
        total_ok += p_ok
        total_fail += p_fail
        total_skip += n_skip + g_skip

        elapsed_p = time.time() - t0
        print(f"\n  Phase {pk} done: {p_ok} ok, {p_fail} failed ({elapsed_p/3600:.1f}h elapsed)")

    elapsed = time.time() - t0
    print(f"\n{'━'*70}")
    print(f"  ALL DONE: {total_ok} ok, {total_fail} failed, {total_skip} skipped, {elapsed/3600:.1f}h total")
    print(f"{'━'*70}")


if __name__ == "__main__":
    main()
