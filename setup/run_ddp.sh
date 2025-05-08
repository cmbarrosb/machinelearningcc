#!/usr/bin/env bash
set -e

# Distributed training settings
MASTER_IP="3.94.129.141"
MASTER_PORT="29500"
WORLD_SIZE=3
KEY_PATH="$HOME/.ssh/mlcc-key.pem"
WORKER_IPS=("3.214.184.67" "44.210.21.75")

# Function to start training on a node
run_node() {
  HOST=$1
  RANK=$2
  echo "Starting rank $RANK on $HOST"
  ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" ec2-user@"$HOST" bash -s << EOF
    set -e
    cd ~/machinelearningcc
    source venv/bin/activate
    export MASTER_ADDR=$MASTER_IP
    export MASTER_PORT=$MASTER_PORT
    export WORLD_SIZE=$WORLD_SIZE
    export RANK=$RANK
    python3 train_ddp.py 2>&1 | tee logs/train_ddp_rank${RANK}.log
EOF
}

# Launch master (rank 0) in background
run_node "$MASTER_IP" 0 &

# Give master time to initialize
sleep 5

# Launch worker nodes
run_node "${WORKER_IPS[0]}" 1 &
run_node "${WORKER_IPS[1]}" 2

echo "Distributed training launched. Logs in logs/train_ddp_rank0.log, train_ddp_rank1.log, train_ddp_rank2.log"