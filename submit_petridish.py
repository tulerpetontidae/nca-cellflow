"""Submit petri dish NCA training jobs via bsub.

Usage:
    python submit_petridish.py                  # submit all configs from scratch
    python submit_petridish.py --resume         # auto-find latest checkpoint per job
    python submit_petridish.py --resume --only petridish-base  # resume a single job
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

configs = [
    # 1. Baseline — hidden_dim=128
    {
        "wandb_name": "petridish-base",
        "config": "configs/petridish-base.yaml",
        "checkpoint_dir": f"{_project_dir}/checkpoints/petridish-base",
    },
    # 2. Baseline, lighter style — style_weight=0.5
    {
        "wandb_name": "petridish-base-style05",
        "config": "configs/petridish-base.yaml",
        "checkpoint_dir": f"{_project_dir}/checkpoints/petridish-base-style05",
        "style_weight": 0.5,
    },
    # 3. Wider — hidden_dim=256
    {
        "wandb_name": "petridish-w256",
        "config": "configs/petridish-base.yaml",
        "checkpoint_dir": f"{_project_dir}/checkpoints/petridish-w256",
        "nca_hidden_dim": 256,
    },
    # 4. Lower homeostasis weight
    {
        "wandb_name": "petridish-homeo1",
        "config": "configs/petridish-homeo1.yaml",
        "checkpoint_dir": f"{_project_dir}/checkpoints/petridish-homeo1",
    },
]


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest petridish_step_*.pt in a checkpoint dir."""
    pattern = os.path.join(checkpoint_dir, "petridish_step_*.pt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    # Extract step number and pick the highest
    def step_num(p):
        m = re.search(r"petridish_step_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1
    return max(ckpts, key=step_num)


def build_command(config):
    cmd_parts = [f"{_python} scripts/train_petridish.py"]
    for key, value in sorted(config.items()):
        if value is None:
            continue
        elif isinstance(value, bool):
            cmd_parts.append(f"--{key}" if value else f"--no-{key}")
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

        if args.resume:
            ckpt_dir = config.get("checkpoint_dir", "")
            latest = find_latest_checkpoint(ckpt_dir)
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
