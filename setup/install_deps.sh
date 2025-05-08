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
# Free pip cache to save space
pip cache purge

# Install CPU-only PyTorch and TorchVision
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu \
  -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install matplotlib pandas
pip install "urllib3<2.0"
pip freeze > requirements.txt
# Clear pip cache again
pip cache purge

pip freeze > requirements.txt
# ---- Freeze versions ----

echo "===== Dependency install finished: $(date) ====="
