#!/bin/bash
set -e

KEY_NAME="mlcc-key"
SG_ID="sg-04ee4e641db5b1220"
SUBNET_ID="subnet-0e2ed6e236ceb0251"
INSTANCE_TYPE="t3.micro"
COUNT=2
LOG_FILE="cluster_setup.log"
HOSTS_FILE="setup/cluster_hosts.txt"

exec > >(tee -a "$LOG_FILE") 2>&1

INSTANCE_IDS=$(aws ec2 run-instances \
  --image-id ami-0c94855ba95c71c99 \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --subnet-id "$SUBNET_ID" \
  --associate-public-ip-address \
  --count "$COUNT" \
  --query "Instances[].InstanceId" \
  --output text)

echo "Launched instances: $INSTANCE_IDS"

echo "Waiting for instances to be in running state..."
aws ec2 wait instance-running --instance-ids $INSTANCE_IDS

echo "Instances are running. Retrieving public DNS names and IP addresses..."

echo "# Cluster worker instances" > "$HOSTS_FILE"
echo "i-0787cbb09cc2cfff7 ec2-3-214-184-67.compute-1.amazonaws.com 3.214.184.67" >> "$HOSTS_FILE"
echo "i-0305e0c0a38cdb212 ec2-44-210-21-75.compute-1.amazonaws.com 44.210.21.75" >> "$HOSTS_FILE"
