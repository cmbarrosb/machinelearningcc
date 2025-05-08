#!/usr/bin/env bash
set -e

LOG_FILE="aws_setup.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "===== EC2 setup started: $(date) ====="

# 1. Get your IP for SSH ingress
MY_IP=$(curl -s http://checkip.amazonaws.com)/32
echo "Your public IP: $MY_IP"

# 2. Create Security Group
SG_NAME="mlcc-sg"
echo "Creating security group $SG_NAME..."
aws ec2 create-security-group \
  --group-name "$SG_NAME" \
  --description "SSH-only access for ML project"
echo "Authorizing SSH ingress..."
aws ec2 authorize-security-group-ingress \
  --group-name "$SG_NAME" \
  --protocol tcp \
  --port 22 \
  --cidr "$MY_IP"

# 3. Launch EC2 instance
AMI_ID="ami-0c94855ba95c71c99"  # Amazon Linux 2 in us-east-1
INSTANCE_TYPE="t3.medium"
KEY_NAME="mlcc-key"
echo "Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-groups "$SG_NAME" \
  --query "Instances[0].InstanceId" \
  --output text)

# 4. Wait for running and retrieve public DNS
echo "Waiting for instance ($INSTANCE_ID) to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
PUBLIC_DNS=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].PublicDnsName" \
  --output text)

echo "Instance is running!"
echo "Public DNS: $PUBLIC_DNS"
echo
echo "SSH in with:"
echo "  ssh -i ~/.ssh/mlcc-key.pem ec2-user@$PUBLIC_DNS"

echo "===== EC2 setup finished: $(date) ====="
