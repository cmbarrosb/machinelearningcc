#!/usr/bin/env bash
set -e

# Worker node IPs (hardcoded)
WORKER_IPS=(
  "3.214.184.67"
  "44.210.21.75"
)

# Path to your SSH key
KEY_PATH="$HOME/.ssh/mlcc-key.pem"

for HOST in "${WORKER_IPS[@]}"; do
  echo "=== Setting up worker at $HOST ==="
  ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" ec2-user@"$HOST" bash -s << 'EOF'
    set -e

    # Install OS packages on the worker
    sudo yum update -y
    sudo yum install -y git python3 python3-devel python3-pip python3-virtualenv gcc gcc-c++ make libffi-devel openssl-devel

    # Clone the repo if missing
    if [ ! -d ~/machinelearningcc ]; then
      git clone https://github.com/cmbarrosb/machinelearningcc.git ~/machinelearningcc
    fi
    cd ~/machinelearningcc

    # Install dependencies
    bash setup/install_deps.sh 2>&1 | tee -a logs/server_setup_worker.log
EOF
done