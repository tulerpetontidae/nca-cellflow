"""
Summarize experiments from checkpoint files.

Walks `checkpoints/` (or a user-specified dir), loads the 80k checkpoint and the
latest checkpoint for each experiment, extracts hyperparameters from `extra`
and FID/loss trajectories from `logs`, and writes a Markdown summary table
plus a per-experiment JSON dump.

Usage:
    python scripts/summarize_experiments.py                          # default paths
    python scripts/summarize_experiments.py --ckpt_dir checkpoints \\
        --out figures/experiments_summary.md \\
        --json_out figures/experiments_summary.json \\
        --exclude petridish
    python scripts/summarize_experiments.py --target_step 80000      # override target step

The output markdown has a sortable table + a short category-based discussion
hook (sorted by best FID). Re-run any time to refresh the numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

STEP_RE = re.compile(r"step_(\d+)\.pt$")


def list_steps(exp_dir: str) -> list[int]:
    """Return sorted list of step ints for step_*.pt files in exp_dir."""
    steps: list[int] = []
    if not os.path.isdir(exp_dir):
        return steps
    for fname in os.listdir(exp_dir):
        m = STEP_RE.match(fname)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def pick_checkpoints(steps: list[int], target_step: int) -> list[int]:
    """Return up to two steps to load: {target_step if present} ∪ {latest}.

    - If target_step exists and equals the latest, return [target_step].
    - If target_step exists but latest > target_step, return [target_step, latest].
    - If target_step missing, return [latest].
    - If no checkpoints, return [].
    """
    if not steps:
        return []
    latest = steps[-1]
    if target_step in steps:
        if latest == target_step:
            return [target_step]
        return [target_step, latest]
    return [latest]


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

# Hyperparameters we want surfaced in the summary (order matters).
HPARAM_KEYS = [
    "nca_type",
    "hidden_channels",
    "noise_channels",
    "z_dim",
    "channel_n",
    "nca_steps",
    "step_size",
    "dx_clip",
    "use_tanh",
    "use_alive_mask",
    "alive_threshold",
    "ema_decay",
    "cond_type",
    "d_type",
    "gamma",
    "lr_g",
    "lr_d",
    "diversity_weight",
    "intermediate_weight",
    "gradual_weight",
    "style_weight",
    "spectral_weight",
    "texture_d",
    "num_compounds",
    "wandb_run_id",
]


@dataclass
class CheckpointStats:
    step: int
    extra: dict[str, Any] = field(default_factory=dict)
    fid_trajectory: list[float] = field(default_factory=list)
    fid_last: float | None = None
    fid_min: float | None = None
    fid_min_idx: int | None = None  # index into trajectory (0-based)
    fid_every: int | None = None  # inferred step spacing between FID evals
    fid_min_step: int | None = None  # actual training step of min FID
    val_fid_last: float | None = None  # cellflux has val_fid/global
    val_fid_min: float | None = None
    G_loss_last: float | None = None
    D_loss_last: float | None = None
    D_real_last: float | None = None
    D_fake_last: float | None = None
    gp_real_last: float | None = None
    gp_fake_last: float | None = None
    log_keys: list[str] = field(default_factory=list)
    # Optional mean-per-compound FID, backfilled from wandb or read from logs
    mpc_trajectory: list[float] = field(default_factory=list)
    mpc_steps: list[int] = field(default_factory=list)
    mpc_last: float | None = None  # value at or closest-before `step`
    mpc_min: float | None = None
    mpc_min_step: int | None = None


def safe_last(arr) -> float | None:
    if arr is None or len(arr) == 0:
        return None
    v = arr[-1]
    try:
        return float(v)
    except Exception:
        return None


def extract_fid(logs: dict, step: int, key: str = "fid/global") -> tuple[list[float], int | None]:
    """Return (trajectory, fid_every) for the given FID log key.

    fid_every is inferred as step / len(trajectory) rounded to nearest 100.
    """
    vals = logs.get(key, [])
    traj = [float(v) for v in vals]
    fid_every = None
    if traj and step:
        raw = step / len(traj)
        # round to nearest 100 for tidiness
        fid_every = int(round(raw / 100.0) * 100)
    return traj, fid_every


def stats_from_ckpt(ck: dict) -> CheckpointStats:
    step = int(ck.get("step", 0))
    extra_raw = ck.get("extra", {}) or {}
    # strip unloadable state_dict fields from extra
    extra = {k: v for k, v in extra_raw.items() if not isinstance(v, dict)}
    logs = ck.get("logs", {}) or {}

    traj, fid_every = extract_fid(logs, step, "fid/global")
    val_traj, _ = extract_fid(logs, step, "val_fid/global")

    s = CheckpointStats(
        step=step,
        extra=extra,
        fid_trajectory=traj,
        fid_every=fid_every,
        log_keys=sorted(logs.keys()),
    )

    if traj:
        arr = np.asarray(traj)
        s.fid_last = float(arr[-1])
        s.fid_min = float(arr.min())
        s.fid_min_idx = int(arr.argmin())
        if fid_every:
            s.fid_min_step = (s.fid_min_idx + 1) * fid_every

    if val_traj:
        arr = np.asarray(val_traj)
        s.val_fid_last = float(arr[-1])
        s.val_fid_min = float(arr.min())

    # Generic loss / logit statistics (present in most runs, absent in non-NCA)
    s.G_loss_last = safe_last(logs.get("loss/G")) or safe_last(logs.get("loss/G_total"))
    s.D_loss_last = safe_last(logs.get("loss/D_total"))
    s.D_real_last = safe_last(logs.get("logits/D_real_mean"))
    s.D_fake_last = safe_last(logs.get("logits/D_fake_mean"))
    s.gp_real_last = safe_last(logs.get("penalty/gp_real_mean"))
    s.gp_fake_last = safe_last(logs.get("penalty/gp_fake_mean"))

    # Mean-per-compound FID (only present in runs trained after train.py was fixed)
    mpc_traj, _ = extract_fid(logs, step, "fid/mean_per_cpd")
    if mpc_traj and fid_every is not None:
        s.mpc_trajectory = mpc_traj
        s.mpc_steps = [(i + 1) * fid_every for i in range(len(mpc_traj))]
        apply_mpc_stats(s, target_step=step)
    return s


def apply_mpc_stats(s: CheckpointStats, target_step: int) -> None:
    """Populate mpc_last / mpc_min / mpc_min_step from trajectory and steps."""
    if not s.mpc_trajectory or not s.mpc_steps:
        return
    arr = np.asarray(s.mpc_trajectory)
    steps_arr = np.asarray(s.mpc_steps)
    s.mpc_min = float(arr.min())
    s.mpc_min_step = int(steps_arr[int(arr.argmin())])
    # Value at (or closest <=) target_step
    mask = steps_arr <= target_step
    if mask.any():
        idx = int(np.where(mask)[0][-1])
        s.mpc_last = float(arr[idx])
    else:
        s.mpc_last = float(arr[0])


# ---------------------------------------------------------------------------
# Wandb backfill (optional)
# ---------------------------------------------------------------------------


def load_wandb_cache(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def save_wandb_cache(path: str, cache: dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_wandb_mpc(
    run_ids: list[tuple[str, str]],  # list of (experiment_name, wandb_run_id)
    project: str,
    entity: str | None,
    cache_path: str,
    refresh: bool = False,
) -> dict[str, dict]:
    """Fetch `fid/mean_per_cpd` history for each run id. Caches results on disk.

    Returns dict keyed by experiment name, with {'steps': [...], 'values': [...], 'run_id': ...}.
    Runs that fail to fetch are stored as {'error': '...'} so we don't retry forever.
    """
    import wandb  # imported lazily so default path doesn't require wandb
    cache = load_wandb_cache(cache_path)
    api = wandb.Api()
    ent = entity or api.default_entity
    project_path = f"{ent}/{project}"

    keys = ["_step", "fid/mean_per_cpd", "fid/global"]

    for name, run_id in run_ids:
        if not refresh and name in cache and "error" not in cache[name]:
            continue
        try:
            print(f"[wandb] fetching {project_path}/{run_id}  ({name})")
            run = api.run(f"{project_path}/{run_id}")
            rows = list(run.scan_history(keys=keys))
            # Deduplicate by _step, prefer rows that have fid/mean_per_cpd
            by_step: dict[int, dict] = {}
            for r in rows:
                if "_step" not in r:
                    continue
                step = int(r["_step"])
                existing = by_step.get(step)
                if existing is None:
                    by_step[step] = r
                else:
                    # prefer one with mean_per_cpd present
                    if "fid/mean_per_cpd" in r and r["fid/mean_per_cpd"] is not None:
                        by_step[step] = r
            steps_sorted = sorted(by_step.keys())
            steps = []
            mpcs = []
            fids = []
            for st in steps_sorted:
                r = by_step[st]
                mpc = r.get("fid/mean_per_cpd")
                fg = r.get("fid/global")
                if mpc is None:
                    continue
                steps.append(st)
                mpcs.append(float(mpc))
                fids.append(float(fg) if fg is not None else None)
            cache[name] = {
                "run_id": run_id,
                "steps": steps,
                "mean_per_cpd": mpcs,
                "fid_global": fids,
            }
        except Exception as e:
            print(f"[wandb] !! failed for {name}/{run_id}: {e}")
            cache[name] = {"run_id": run_id, "error": str(e)}
    save_wandb_cache(cache_path, cache)
    return cache


def merge_wandb_into_stats(
    experiments: list[ExperimentSummary],
    wandb_cache: dict[str, dict],
    target_step: int,
) -> None:
    """Populate mpc_* fields on each experiment's checkpoints from wandb_cache."""
    for e in experiments:
        entry = wandb_cache.get(e.name)
        if not entry or "error" in entry or not entry.get("steps"):
            continue
        steps = entry["steps"]
        mpcs = entry["mean_per_cpd"]
        for ck in (e.target_ckpt, e.latest_ckpt):
            if ck is None:
                continue
            # filter trajectory up to ck.step
            traj = [(s, v) for s, v in zip(steps, mpcs) if s <= ck.step]
            if not traj:
                continue
            ck.mpc_steps = [s for s, _ in traj]
            ck.mpc_trajectory = [v for _, v in traj]
            apply_mpc_stats(ck, target_step=ck.step)


# ---------------------------------------------------------------------------
# Experiment aggregation
# ---------------------------------------------------------------------------


@dataclass
class ExperimentSummary:
    name: str
    target_ckpt: CheckpointStats | None  # 80k (or nearest fallback)
    latest_ckpt: CheckpointStats | None  # latest (only populated if > target)
    had_target: bool  # True iff target_step existed exactly

    def best_fid(self) -> float | None:
        """Best FID seen across loaded checkpoints."""
        cands = [c for c in (self.target_ckpt, self.latest_ckpt) if c is not None and c.fid_min is not None]
        if not cands:
            return None
        return min(c.fid_min for c in cands)

    def best_mpc(self) -> float | None:
        cands = [c for c in (self.target_ckpt, self.latest_ckpt) if c is not None and c.mpc_min is not None]
        if not cands:
            return None
        return min(c.mpc_min for c in cands)

    def headline_ckpt(self) -> CheckpointStats | None:
        """The checkpoint we treat as the 'headline' for the summary row."""
        return self.target_ckpt or self.latest_ckpt


def collect_experiments(
    ckpt_dir: str,
    target_step: int,
    exclude: list[str],
    only: list[str] | None = None,
) -> list[ExperimentSummary]:
    results: list[ExperimentSummary] = []
    for name in sorted(os.listdir(ckpt_dir)):
        exp_dir = os.path.join(ckpt_dir, name)
        if not os.path.isdir(exp_dir):
            continue
        if any(pat in name for pat in exclude):
            continue
        if only and not any(pat in name for pat in only):
            continue

        steps = list_steps(exp_dir)
        if not steps:
            continue

        pick = pick_checkpoints(steps, target_step)
        target_ck: CheckpointStats | None = None
        latest_ck: CheckpointStats | None = None
        had_target = target_step in steps

        for i, step in enumerate(pick):
            path = os.path.join(exp_dir, f"step_{step}.pt")
            print(f"[load] {name}/step_{step}.pt")
            try:
                ck = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"  !! failed to load {path}: {e}")
                continue
            stats = stats_from_ckpt(ck)
            if had_target and step == target_step:
                target_ck = stats
            elif i == 0 and not had_target:
                target_ck = stats  # fallback: latest-as-target
            else:
                latest_ck = stats

        results.append(
            ExperimentSummary(
                name=name,
                target_ckpt=target_ck,
                latest_ckpt=latest_ck,
                had_target=had_target,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def fmt_float(x: float | None, precision: int = 2) -> str:
    if x is None:
        return "—"
    if abs(x) >= 1000:
        return f"{x:.0f}"
    return f"{x:.{precision}f}"


def classify_model(name: str, extra: dict) -> str:
    """Return a coarse model category for grouping in the summary."""
    if "cellflux" in name:
        return "external:cellflux"
    if "impa" in name:
        return "external:impa"
    nca_type = extra.get("nca_type")
    if nca_type == "latent":
        return "nca:latent"
    if nca_type == "noise":
        return "nca:noise"
    if nca_type is None and "noise" in name:
        return "nca:noise-legacy"
    if nca_type is None:
        return "nca:base-legacy"
    return f"nca:{nca_type}"


def short_hparam_blurb(name: str, extra: dict) -> str:
    """One-line summary of the most distinguishing hyperparameters."""
    cat = classify_model(name, extra)
    if cat.startswith("external:"):
        return cat.split(":", 1)[1]

    parts = []
    nca_type = extra.get("nca_type")
    if nca_type:
        parts.append(str(nca_type))
    if "hidden_channels" in extra:
        parts.append(f"h{extra['hidden_channels']}")
    if "nca_steps" in extra:
        parts.append(f"T={extra['nca_steps']}")
    if "step_size" in extra:
        parts.append(f"ss={extra['step_size']}")
    if extra.get("use_tanh"):
        parts.append("tanh")
    if extra.get("use_alive_mask"):
        parts.append("alive")
    if extra.get("d_type") and extra["d_type"] != "global":
        parts.append(f"d={extra['d_type']}")
    if extra.get("cond_type") and extra["cond_type"] != "embedding":
        parts.append(f"cond={extra['cond_type']}")
    for key, label in [
        ("intermediate_weight", "inter"),
        ("diversity_weight", "div"),
        ("gradual_weight", "grad"),
        ("style_weight", "sty"),
        ("spectral_weight", "spec"),
    ]:
        v = extra.get(key)
        if v is not None and v != 0 and v != 0.0:
            parts.append(f"{label}{v}")
    if extra.get("texture_d"):
        parts.append("texD")
    return " ".join(parts) if parts else "(legacy)"


def render_markdown(
    experiments: list[ExperimentSummary],
    target_step: int,
) -> str:
    lines: list[str] = []
    lines.append(f"# Experiment summary (target step = {target_step})")
    lines.append("")
    lines.append(
        "Loaded the target checkpoint (fallback: latest available) and the latest checkpoint "
        "when training continued past the target. FID is `fid/global` computed every ~5000 "
        "steps on the test split against the EMA generator. Older runs pre-date FID logging "
        "and show `—`."
    )
    lines.append("")

    # Sort: experiments with FID first (ascending by best FID), then the rest.
    def sort_key(e: ExperimentSummary):
        best = e.best_fid()
        if best is None:
            return (1, e.name)
        return (0, best)

    experiments_sorted = sorted(experiments, key=sort_key)

    # Main table
    lines.append("## Results table")
    lines.append("")
    lines.append(
        "**FID_g** = global FID (all generated vs all real). "
        "**FID_c** = mean-per-compound FID (average of 34 per-compound FIDs — much harder). "
        "Ranking below is by best FID_g for backwards compatibility."
    )
    lines.append("")
    header = [
        "Rank",
        "Experiment",
        "Config",
        "Steps",
        "FID_g@target",
        "FID_g best",
        "FID_g best@step",
        "FID_g latest",
        "FID_c@target",
        "FID_c best",
        "G loss",
        "D loss",
        "D_real / D_fake",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    rank = 0
    for exp in experiments_sorted:
        head = exp.headline_ckpt()
        if head is None:
            continue

        best_fid = exp.best_fid()
        if best_fid is not None:
            rank += 1
            rank_str = str(rank)
        else:
            rank_str = "—"

        fid_target = head.fid_last
        fid_best_step = head.fid_min_step
        # If we also have a latest_ckpt, the "latest FID" is from latest_ckpt, and
        # the "best" may come from either.
        latest = exp.latest_ckpt
        latest_step = latest.step if latest is not None else head.step
        fid_latest = latest.fid_last if latest is not None else head.fid_last
        if latest is not None and latest.fid_min is not None:
            if head.fid_min is None or latest.fid_min < head.fid_min:
                fid_best_step = latest.fid_min_step

        steps_str = f"{head.step:,}" + (f" → {latest_step:,}" if latest is not None else "")

        dr = head.D_real_last
        df = head.D_fake_last
        drdf = "—"
        if dr is not None or df is not None:
            drdf = f"{fmt_float(dr, 2)} / {fmt_float(df, 2)}"

        # mean-per-compound FID from headline + optional latest
        mpc_target = head.mpc_last
        best_mpc = exp.best_mpc()

        row = [
            rank_str,
            f"`{exp.name}`",
            short_hparam_blurb(exp.name, head.extra),
            steps_str,
            fmt_float(fid_target),
            fmt_float(best_fid),
            f"{fid_best_step:,}" if fid_best_step else "—",
            fmt_float(fid_latest),
            fmt_float(mpc_target),
            fmt_float(best_mpc),
            fmt_float(head.G_loss_last, 3),
            fmt_float(head.D_loss_last, 3),
            drdf,
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    # Auto-generated notes
    lines.append("## Notes")
    lines.append("")

    with_fid = [e for e in experiments_sorted if e.best_fid() is not None]
    without_fid = [e for e in experiments_sorted if e.best_fid() is None]

    if with_fid:
        top5 = with_fid[:5]
        lines.append("**Best runs by FID:**")
        for e in top5:
            head = e.headline_ckpt()
            assert head is not None
            lines.append(
                f"- `{e.name}` — best FID **{e.best_fid():.2f}** "
                f"({short_hparam_blurb(e.name, head.extra)})"
            )
        lines.append("")

        worst5 = with_fid[-5:][::-1]
        lines.append("**Worst runs by FID (with FID logged):**")
        for e in worst5:
            head = e.headline_ckpt()
            assert head is not None
            lines.append(
                f"- `{e.name}` — best FID **{e.best_fid():.2f}** "
                f"({short_hparam_blurb(e.name, head.extra)})"
            )
        lines.append("")

    # Category breakdown: best FID per coarse category
    by_cat: dict[str, list[ExperimentSummary]] = {}
    for e in with_fid:
        head = e.headline_ckpt()
        assert head is not None
        cat = classify_model(e.name, head.extra)
        by_cat.setdefault(cat, []).append(e)

    if by_cat:
        lines.append("**By model category — global FID:**")
        lines.append("")
        lines.append("| Category | n | Best | Mean | Median | Worst | Winner |")
        lines.append("|---|---|---|---|---|---|---|")
        for cat in sorted(by_cat.keys()):
            runs = by_cat[cat]
            fids = [r.best_fid() for r in runs]  # type: ignore[misc]
            fids = [f for f in fids if f is not None]
            if not fids:
                continue
            winner = min(runs, key=lambda r: r.best_fid() or float("inf"))
            arr = np.asarray(fids)
            lines.append(
                f"| {cat} | {len(runs)} | {arr.min():.2f} | {arr.mean():.2f} | "
                f"{float(np.median(arr)):.2f} | {arr.max():.2f} | `{winner.name}` |"
            )
        lines.append("")

        lines.append("**By model category — mean-per-compound FID:**")
        lines.append("")
        lines.append("| Category | n w/ FID_c | Best | Mean | Median | Worst | Winner |")
        lines.append("|---|---|---|---|---|---|---|")
        for cat in sorted(by_cat.keys()):
            runs = by_cat[cat]
            mpcs = [r.best_mpc() for r in runs]
            mpcs = [m for m in mpcs if m is not None]
            if not mpcs:
                continue
            winner = min(
                (r for r in runs if r.best_mpc() is not None),
                key=lambda r: r.best_mpc() or float("inf"),
            )
            arr = np.asarray(mpcs)
            lines.append(
                f"| {cat} | {len(mpcs)} | {arr.min():.2f} | {arr.mean():.2f} | "
                f"{float(np.median(arr)):.2f} | {arr.max():.2f} | `{winner.name}` |"
            )
        lines.append("")

    if without_fid:
        lines.append(f"**Legacy runs (no FID logged, {len(without_fid)} runs):** "
                     + ", ".join(f"`{e.name}`" for e in without_fid))
        lines.append("")

    lines.append(AUTO_GEN_END)
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Preserve handwritten discussion below the auto-generated section
# ---------------------------------------------------------------------------

AUTO_GEN_END = "<!-- END AUTO-GENERATED: content below is preserved across re-runs -->"

DEFAULT_DISCUSSION = """## Discussion

_This section is preserved when the script is re-run. Edit freely._

- **What worked:** ...
- **What didn't:** ...
- **Open questions:** ...
"""


def preserve_manual_section(out_path: str) -> str | None:
    """If `out_path` already has an AUTO_GEN_END marker, return everything below it.

    Returns None when the file doesn't exist or has no marker.
    """
    if not os.path.exists(out_path):
        return None
    with open(out_path) as f:
        content = f.read()
    idx = content.find(AUTO_GEN_END)
    if idx == -1:
        return None
    tail = content[idx + len(AUTO_GEN_END):].lstrip("\n")
    return tail


def experiments_to_json(experiments: list[ExperimentSummary]) -> list[dict]:
    out = []
    for e in experiments:
        out.append({
            "name": e.name,
            "had_target": e.had_target,
            "target_ckpt": asdict(e.target_ckpt) if e.target_ckpt else None,
            "latest_ckpt": asdict(e.latest_ckpt) if e.latest_ckpt else None,
            "best_fid": e.best_fid(),
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_dir", default="checkpoints", help="Root checkpoint directory")
    parser.add_argument("--target_step", type=int, default=80000, help="Preferred checkpoint step")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["petridish", "configs"],
        help="Skip experiment names containing any of these substrings",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Only include experiment names containing any of these substrings",
    )
    parser.add_argument("--out", default="figures/experiments_summary.md")
    parser.add_argument("--json_out", default="figures/experiments_summary.json")
    parser.add_argument(
        "--wandb_backfill", action="store_true",
        help="Fetch fid/mean_per_cpd history from wandb for runs with wandb_run_id in extra",
    )
    parser.add_argument("--wandb_project", default="nca-cellflow")
    parser.add_argument("--wandb_entity", default=None, help="Defaults to api.default_entity")
    parser.add_argument(
        "--wandb_cache", default="figures/wandb_mpc_cache.json",
        help="On-disk cache of fetched mean-per-cpd histories",
    )
    parser.add_argument(
        "--wandb_refresh", action="store_true",
        help="Re-fetch all runs even if cached",
    )
    args = parser.parse_args()

    print(f"[scan] {args.ckpt_dir} (target step {args.target_step}, exclude {args.exclude})")
    experiments = collect_experiments(
        ckpt_dir=args.ckpt_dir,
        target_step=args.target_step,
        exclude=args.exclude,
        only=args.only,
    )
    print(f"[scan] loaded {len(experiments)} experiments")

    # Optional wandb backfill of fid/mean_per_cpd
    if args.wandb_backfill:
        run_ids: list[tuple[str, str]] = []
        for e in experiments:
            head = e.headline_ckpt()
            if head is None:
                continue
            wid = (head.extra or {}).get("wandb_run_id")
            if wid:
                run_ids.append((e.name, str(wid)))
        print(f"[wandb] backfilling {len(run_ids)} runs from {args.wandb_project}")
        wandb_cache = fetch_wandb_mpc(
            run_ids=run_ids,
            project=args.wandb_project,
            entity=args.wandb_entity,
            cache_path=args.wandb_cache,
            refresh=args.wandb_refresh,
        )
        merge_wandb_into_stats(experiments, wandb_cache, target_step=args.target_step)

    md = render_markdown(experiments, target_step=args.target_step)
    # Preserve any handwritten discussion beneath the auto-generated marker.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    manual_tail = preserve_manual_section(args.out)
    if manual_tail is None:
        manual_tail = DEFAULT_DISCUSSION
    full = md + manual_tail
    if not full.endswith("\n"):
        full += "\n"
    with open(args.out, "w") as f:
        f.write(full)
    print(f"[write] {args.out}")

    with open(args.json_out, "w") as f:
        json.dump(experiments_to_json(experiments), f, indent=2, default=str)
    print(f"[write] {args.json_out}")


if __name__ == "__main__":
    main()
