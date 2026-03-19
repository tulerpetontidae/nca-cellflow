#!/bin/bash
# Sync checkpoints from Sherlock to local machine
# Usage: bash scripts/sync_checkpoints.sh [experiment_name]
#   e.g. bash scripts/sync_checkpoints.sh baseline-fire05-lb
#   Without args: syncs all checkpoints

set -e

REMOTE="sherlock:/oak/stanford/groups/ccurtis2/users/alomakin/projects/nca-cellflow/checkpoints"
LOCAL="checkpoints"

mkdir -p "${LOCAL}"

if [ -n "$1" ]; then
    rsync -avz --progress "${REMOTE}/$1/" "${LOCAL}/$1/"
else
    rsync -avz --progress "${REMOTE}/" "${LOCAL}/"
fi
