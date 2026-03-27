#!/bin/bash
# Sync project to Sherlock cluster with progress bar
# Usage: bash scripts/deploy.sh

REMOTE="sherlock:/oak/stanford/groups/ccurtis2/users/alomakin/projects/nca-cellflow"

echo "Deploying to ${REMOTE} ..."

rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'wandb/' \
    --exclude 'tests/checkpoints/' \
    --exclude 'checkpoints/' \
    --exclude '*.egg-info' \
    --exclude '.DS_Store' \
    --exclude 'data/*' \
    --exclude 'notebooks/*' \
    --exclude 'figures/*' \
    ./ "${REMOTE}/"

echo "Deploy complete."
