#!/usr/bin/env bash
set -e

# ---- Logging ----
LOG_FILE="deps_install.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "===== Dependency install started: $(date) ====="

# ---- Setup venv ----
python3 -m venv venv
source venv/bin/activate

# ---- Install packages ----
pip install --upgrade pip
pip install torch torchvision matplotlib pandas

# ---- Freeze versions ----
pip freeze > requirements.txt

echo "===== Dependency install finished: $(date) ====="
