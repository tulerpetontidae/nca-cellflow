"""Submit 96px comparison jobs (NCA, CellFlux, IMPA) via bsub.

Usage:
    python submit_cellflow_comp.py                  # submit all jobs
    python submit_cellflow_comp.py --resume         # auto-find latest checkpoint
    python submit_cellflow_comp.py --only nca       # submit only jobs matching "nca"
    python submit_cellflow_comp.py --dry-run        # print commands without submitting
"""

import argparse
import glob
import os
import re
import subprocess

try:
    _netrc = open(os.path.expanduser("~/.netrc")).read()
    _wandb_key = re.search(r"password (\S+)", _netrc).group(1)
    _wandb_env = f"WANDB_API_KEY={_wandb_key} "
except Exception:
    _wandb_env = ""

bad_hosts = ["lsf22-gpu02", "lsf22-gpu03", "lsf22-gpu05", "lsf22-gpu07", "aih-superl40s-06"]

_host_select = " && ".join(f"hname!='{h}'" for h in bad_hosts)
_host_select_clause = f' -R "select[{_host_select}]"' if bad_hosts else ""
_project_dir = "/omics/groups/OE0606/internal/art1m/projects/nca-cellflow"
_python = "/omics/groups/OE0606/internal/art1m/micromamba/envs/voidtracer/bin/python"
bsub_template = f"bsub -gpu num=1:j_exclusive=yes:gmem=100G -R 'rusage[mem=32GB]'{_host_select_clause} -q gpu-pro 'cd {_project_dir} && OMP_NUM_THREADS=16 {_wandb_env}{{command}}'"

# Map model type → training script
_scripts = {
    "nca": f"{_python} scripts/train.py",
    "cellflux": f"{_python} scripts/train_cellflux.py",
    "impa": f"{_python} scripts/train_impa.py",
}

_ckpt_prefix = f"{_project_dir}/checkpoints"

configs = [
    # --- NCA (LatentNCA) 96px ---
    # 1. Leader config scaled to 96px, 60 steps
    {
        "model": "nca",
        "wandb_name": "nca-h24-nh256-w96D-96px",
        "config": "configs/latent-hidden24-ema-steps60-v2-ss008-plate-fp-balanced-tanh-w96D-nh256-96px-lb.yaml",
        "checkpoint_dir": f"{_ckpt_prefix}/latent-hidden24-ema-steps60-v2-ss008-plate-fp-balanced-tanh-w96D-nh256-96px-lb",
    },
    # 2. Same + intermediate regularization
    {
        "model": "nca",
        "wandb_name": "nca-h24-nh256-w96D-inter-96px",
        "config": "configs/latent-hidden24-ema-steps60-v2-ss008-plate-fp-balanced-tanh-w96D-nh256-inter-96px-lb.yaml",
        "checkpoint_dir": f"{_ckpt_prefix}/latent-hidden24-ema-steps60-v2-ss008-plate-fp-balanced-tanh-w96D-nh256-inter-96px-lb",
    },
    # --- CellFlux 96px ---
    {
        "model": "cellflux",
        "wandb_name": "cellflux-96px",
        "config": "configs/cellflux-bbbc021-96px-lb.yaml",
        "checkpoint_dir": f"{_ckpt_prefix}/cellflux-bbbc021-96px-lb",
    },
    # --- IMPA 96px ---
    {
        "model": "impa",
        "wandb_name": "impa-96px",
        "config": "configs/impa-bbbc021-96px-lb.yaml",
        "checkpoint_dir": f"{_ckpt_prefix}/impa-bbbc021-96px-lb",
    },
]

# Checkpoint naming patterns per model type
_ckpt_patterns = {
    "nca": "step_*.pt",
    "cellflux": "step_*.pt",
    "impa": "step_*.pt",
}


def find_latest_checkpoint(checkpoint_dir, model_type):
    """Find the latest checkpoint in a checkpoint dir."""
    pattern_name = _ckpt_patterns.get(model_type, "step_*.pt")
    pattern = os.path.join(checkpoint_dir, pattern_name)
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    def step_num(p):
        m = re.search(r"_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1
    return max(ckpts, key=step_num)


def build_command(config):
    model_type = config["model"]
    script = _scripts[model_type]
    cmd_parts = [script]
    for key, value in sorted(config.items()):
        if key == "model":
            continue
        if value is None:
            continue
        elif isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key}")
        else:
            cmd_parts.append(f"--{key} {value}")
    return " ".join(cmd_parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Auto-find latest checkpoint and resume each job")
    parser.add_argument("--only", type=str, default=None,
                        help="Only submit jobs whose wandb_name matches this string")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without submitting")
    args = parser.parse_args()

    for config in configs:
        name = config["wandb_name"]
        if args.only and args.only not in name:
            continue

        # Always pass --wandb
        config["wandb"] = True

        if args.resume:
            ckpt_dir = config.get("checkpoint_dir", "")
            latest = find_latest_checkpoint(ckpt_dir, config["model"])
            if latest:
                config["resume"] = latest
                print(f"[{name}] resuming from {latest}")
            else:
                print(f"[{name}] no checkpoint found in {ckpt_dir}, starting fresh")

        command = build_command(config)
        bsub_command = bsub_template.format(command=command)
        if args.dry_run:
            print(f"[dry-run] {bsub_command}")
        else:
            print(bsub_command)
            subprocess.run(bsub_command, shell=True)
