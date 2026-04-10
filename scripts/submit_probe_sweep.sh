#!/bin/bash
# Submit a batch of capacity-probe jobs to Sherlock.
# Usage:
#   bash scripts/submit_probe_sweep.sh             # submit all configs/probe/*.yaml
#   bash scripts/submit_probe_sweep.sh 'probe-d-*' # glob filter (quote it!)
#   bash scripts/submit_probe_sweep.sh 'probe-s-*'
#
# Iterates configs/probe/<filter>.yaml and calls submit_probe.sh for each.

set -e

FILTER="${1:-probe-*}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

count=0
for cfg in "${REPO_DIR}/configs/probe/"${FILTER}.yaml; do
    if [ ! -f "$cfg" ]; then
        echo "No configs match: ${FILTER}"
        exit 1
    fi
    name="$(basename "$cfg" .yaml)"
    echo ">>> ${name}"
    bash "${REPO_DIR}/scripts/submit_probe.sh" "${name}"
    count=$((count + 1))
done
echo "Submitted ${count} jobs."
