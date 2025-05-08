#!/bin/bash
set -e

KEY_NAME="mlcc-key"
SG_ID="sg-04ee4e641db5b1220"
SUBNET_ID="subnet-0e2ed6e236ceb0251"
INSTANCE_TYPE="t3.micro"
COUNT=2
LOG_FILE="cluster_setup.log"

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

for INSTANCE_ID in $INSTANCE_IDS; do
  DNS_NAME=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].PublicDnsName" --output text)
  IP_ADDR=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
  echo "Instance ID: $INSTANCE_ID"
  echo "  Public DNS: $DNS_NAME"
  echo "  Public IP: $IP_ADDR"
done
