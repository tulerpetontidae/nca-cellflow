"""Submit petri dish NCA training jobs via bsub."""

import subprocess
import re

try:
    _netrc = open(__import__("os").path.expanduser("~/.netrc")).read()
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
    # # 1. Baseline — hidden_dim=128
    # {
    #     "wandb_name": "petridish-base",
    #     "config": "configs/petridish-base.yaml",
    # },
    # 2. Wider — hidden_dim=256
    {
        "wandb_name": "petridish-w256",
        "config": "configs/petridish-base.yaml",
        "nca_hidden_dim": 256,
    },
]


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
    for config in configs:
        command = build_command(config)
        bsub_command = bsub_template.format(command=command)
        print(bsub_command)
        subprocess.run(bsub_command, shell=True)
