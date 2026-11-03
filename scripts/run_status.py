"""Status of the research reference model runs on a cluster: one block per run.

    python3 scripts/run_status.py [--runs <runs folder>] [--epochs 8] [--watch 60]

For each run: its state (running, queued, stale, waiting, done), the live progress of the current
epoch (from the newest job log), the last epochs of log.jsonl (training loss and MPJPE, validation
MPJPE per Masking mode, EMA vs raw weights), the best epoch, an ETA, and the Test results once
evaluated (newest-frame MPJPE per Camera setting and Masking mode, prior perturbation slopes).
Standard library only, so it runs on a login node without any module.
"""
import argparse
import getpass
import json
import os
import re
import subprocess
import time
from pathlib import Path

# run folder -> (job name of its loop, step name in the job log, run it waits for)
RUNS = {"reference_backbone": ("pf_reference", "backbone", None),
        "reference_versatile": ("pf_reference", "versatile", "reference_backbone"),
        "reference_versatile_scratch": ("pf_scratch", "versatile from scratch", None),
        "reference_full_amass": ("pf_full", "versatile from scratch, full AMASS, size base", None),
        "reference_full_amass_medium": ("pf_full_medium", "versatile from scratch, full AMASS, size medium", None),
        "reference_full_amass_large": ("pf_full_large", "versatile from scratch, full AMASS, size large", None)}
DEFAULT_EPOCHS = {"reference_backbone": 50, "reference_versatile": 20, "reference_versatile_scratch": 50,
                  "reference_full_amass": 6, "reference_full_amass_medium": 6, "reference_full_amass_large": 4}
MODES = ["full_priors", "camera_estimated", "segments_masked", "no_priors"]
SHORT = {"full_priors": "full", "camera_estimated": "cam est", "segments_masked": "seg mask", "no_priors": "none"}
STALE_MINUTES = 20
TQDM = re.compile(r"epoch (\d+):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.]+)(it/s|s/it)"
                  r"(?:, mpjpe=([\d.]+))?")


def jobs():
    """{job name: [(id, state, elapsed)]} of the user's Slurm jobs ({} without Slurm)."""
    try:
        out = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%i|%j|%T|%M"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    found = {}
    for line in out.splitlines():
        jid, name, state, elapsed = line.split("|")
        found.setdefault(name, []).append((jid, state, elapsed))
    return found


def loops():
    try:
        out = subprocess.run(["pgrep", "-af", "chain_reference.sh"], capture_output=True, text=True).stdout
    except OSError:
        return []
    variants = {"scratch": "scratch", "full": "full"}
    return [variants.get(line.split()[-1], "two-stage") for line in out.splitlines() if "pgrep" not in line]


def newest_log(logs, job_name):
    files = sorted(logs.glob(f"{job_name}_*.out"), key=lambda f: f.stat().st_mtime)
    return files[-1] if files else None


def log_tail(path, size=20000):
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - size))
        return f.read().decode(errors="replace")


def log_head(path, size=20000):
    """The start of a job log, where each link prints its step (--- ...)."""
    with open(path, "rb") as f:
        return f.read(size).decode(errors="replace")


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass   # a line being written
    return rows


def hours(minutes):
    return f"{minutes / 60:.1f} h" if minutes >= 60 else f"{minutes:.0f} min"


def run_block(runs, logs, name, all_jobs, n_epochs):
    job_name, step, after = RUNS[name]
    folder = runs / name
    config = json.loads((folder / "config.json").read_text()) if (folder / "config.json").exists() else {}
    epochs = config.get("epochs", DEFAULT_EPOCHS[name])
    log = read_jsonl(folder / "log.jsonl")
    done_train = (folder / "DONE").exists()
    evals = sorted((folder / "eval_v2").glob("*.json")) if (folder / "eval_v2").exists() else []

    # state
    tail, running_here, progress = "", False, None
    log_file = newest_log(logs, job_name)
    if log_file:
        tail = log_tail(log_file)
        steps = re.findall(r"^--- (.+)$", log_head(log_file) + tail, re.M)   # the step line scrolls out of the tail
        running_here = bool(steps) and (steps[-1] == step or steps[-1].startswith(f"evaluate {name} "))
    job_list = all_jobs.get(job_name, [])
    running = [j for j in job_list if j[1] == "RUNNING"]
    pending = [j for j in job_list if j[1] == "PENDING"]
    if done_train and len(evals) >= (4 if name.startswith("reference_full_amass") else 2):   # + Unseen clips
        state = "DONE (trained and evaluated)"
    elif running and running_here:
        state = f"RUNNING  job {running[0][0]}, {running[0][2]} into this 2 h link"
    elif done_train:
        state = "TRAINED, evaluation " + ("queued" if job_list else "waiting for a link")
    elif after and not (runs / after / "DONE").exists():
        state = f"WAITING for {after}"
    elif pending or running:
        state = f"QUEUED   job {(pending or running)[0][0]} ({(pending or running)[0][1].lower()})"
    elif (folder / "last.pt").exists() and time.time() - (folder / "last.pt").stat().st_mtime > 60 * STALE_MINUTES:
        state = f"STALE    no job, last save {hours((time.time() - (folder / 'last.pt').stat().st_mtime) / 60)} ago"
    elif not folder.exists():
        state = "NOT STARTED"
    else:
        state = "BETWEEN LINKS (next link soon)"

    lines = [f"\033[1m{name}\033[0m  {state}"]
    if config:
        lines.append(f"  recipe: size {config.get('size') or 'base'}{' amp' if config.get('amp') else ''}, lr {config.get('lr')} {config.get('schedule', 'cosine')}, ema {config.get('ema')}, "
                     f"seg jitter {config.get('seg_jitter')}, {config.get('train_clips')} train clips")

    if running and running_here and tail:
        bars = [m for m in TQDM.finditer(tail.replace("\r", "\n"))]
        if bars:
            ep, pct, step_i, total, spent, left, rate, unit, mp = bars[-1].groups()
            speed = float(rate) if unit == "it/s" else 1 / float(rate)
            epoch_min = int(total) / speed / 60
            remaining = (int(total) - int(step_i)) / speed / 60 + max(0, epochs - int(ep)) * epoch_min
            progress = (f"  now: epoch {ep}/{epochs}  {pct}%  ({step_i}/{total} steps, {speed:.1f} it/s"
                        + (f", train MPJPE {1000 * float(mp):.1f} mm" if mp else "") + ")"
                        + f"   epoch ≈ {hours(epoch_min)}, training left ≈ {hours(remaining)}")
            lines.append(progress)

    if log:
        modes = [m for m in MODES if m in log[-1]["val_mpjpe"]]
        has_raw = "val_mpjpe_raw" in log[-1]
        head = "  epoch  train loss  train MPJPE | val MPJPE (mm): " + "  ".join(f"{SHORT[m]:>8}" for m in modes)
        lines += [head + ("     raw" if has_raw else "") + "   min"]
        for row in log[-n_epochs:]:
            vals = "  ".join(f"{1000 * row['val_mpjpe'][m]:8.1f}" for m in modes)
            raw = f"  {1000 * row['val_mpjpe_raw']['full_priors']:6.1f}" if has_raw else ""
            loss = f"{row['train_loss']:10.4f}" if "train_loss" in row else f"{'':>10}"
            lines.append(f"  {row['epoch']:5d}  {loss}  {1000 * row['train_mpjpe']:9.1f}   |"
                         f"{'':16}{vals}{raw}  {row['minutes']:5.0f}")
        best = min(log, key=lambda r: r["val_mpjpe"]["full_priors"])
        mean_min = sum(r["minutes"] for r in log[-5:]) / len(log[-5:])
        left = max(0, epochs - log[-1]["epoch"])
        lines.append(f"  {len(log)}/{epochs} epochs; best val (full priors) {1000 * best['val_mpjpe']['full_priors']:.1f} mm "
                     f"at epoch {best['epoch']}; ≈ {hours(mean_min)}/epoch"
                     + (f", ≈ {hours(left * mean_min)} of training left" if left else ""))
    elif folder.exists() and progress is None:
        lines.append("  no finished epoch yet")

    for path in evals:
        res = json.loads(path.read_text())
        for setting, summary in res["summary"].items():
            cells = "  ".join(f"{SHORT[m]} {summary[m]['mpjpe']['all_clips']['mean_mm']:.1f}" for m in MODES if m in summary)
            lines.append(f"  TEST {setting:8s} newest-frame MPJPE (mm): {cells}")
            fp = summary.get("full_priors", {})
            if fp:
                lines.append(f"  {'':14s}full priors: N-MPJPE {fp['n_mpjpe']['all_clips']['mean_mm']:.1f}, "
                             f"PA-MPJPE {fp['pa_mpjpe']['all_clips']['mean_mm']:.1f}, "
                             f"bone error {fp['bone_error']['all_clips']['mean_mm']:.1f}")
        for setting, sw in res.get("prior_sweep", {}).items():
            sl = {c: sw[c]["slope_mm_per_eps"] for c in ["segments", "focal"]}
            lines.append(f"  {'':14s}sweep slopes (mm per unit eps, +/-): segments {sl['segments']['positive']:+.0f} / "
                         f"{sl['segments']['negative']:+.0f}, focal {sl['focal']['positive']:+.0f} / {sl['focal']['negative']:+.0f}")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=os.path.expandvars("$SCRATCH/priorformer/runs"))
    p.add_argument("--epochs", type=int, default=8, help="last N epochs shown per run")
    p.add_argument("--watch", type=int, help="refresh every N seconds (Ctrl+C to quit)")
    args = p.parse_args()
    runs = Path(args.runs)
    while True:
        all_jobs = jobs()
        active = loops()
        out = [f"PriorFormer runs in {runs}   {time.strftime('%Y-%m-%d %H:%M')}   "
               f"submit loops: {', '.join(active) if active else 'NONE RUNNING'}", ""]
        for name in RUNS:
            out += run_block(runs, runs / "logs", name, all_jobs, args.epochs) + [""]
        if args.watch:
            print("\033[2J\033[H", end="")
        print("\n".join(out), flush=True)
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
