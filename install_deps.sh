#!/usr/bin/env bash
set -e

# ---- System update and OS package install ----
echo "===== System update and OS package install started: $(date) ====="
sudo yum update -y
sudo yum install -y \
  git \
  python3 \
  python3-devel \
  python3-pip \
  python3-virtualenv \
  gcc \
  gcc-c++ \
  make \
  libffi-devel \
  openssl-devel
echo "===== System packages installed ====="

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
