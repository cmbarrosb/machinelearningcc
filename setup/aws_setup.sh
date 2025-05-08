#!/usr/bin/env bash
# Built-in log to aws_setup.log
LOG_FILE="aws_setup.log"
exec > >(tee "$LOG_FILE") 2>&1

set -e

# 1. Get your IP for SSH ingress
MY_IP=$(curl -s http://checkip.amazonaws.com)/32
echo "Your public IP: $MY_IP"

# Configuration: use default VPC and predefined Security Group
VPC_ID=vpc-0dabef63c6da78fde
SG_ID=sg-04ee4e641db5b1220
echo "Using VPC_ID=$VPC_ID, SG_ID=$SG_ID"

# Lookup a subnet in the VPC
SUBNET_ID=$(aws ec2 describe-subnets \
  --filters Name=vpc-id,Values="$VPC_ID" \
  --query "Subnets[0].SubnetId" --output text)
echo "Using SUBNET_ID=$SUBNET_ID"

# Authorize SSH ingress on predefined SG (ignore duplicates)
echo "Authorizing SSH ingress..."
aws ec2 authorize-security-group-ingress \
  --group-id "$SG_ID" \
  --protocol tcp --port 22 --cidr "$MY_IP" 2>/dev/null || true

#
# 3. Launch EC2 instance
AMI_ID="ami-0c94855ba95c71c99"  # Amazon Linux 2 in us-east-1
INSTANCE_TYPE="t3.medium"
KEY_NAME="mlcc-key"
echo "Launching EC2 instance in subnet $SUBNET_ID..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SG_ID" \
  --associate-public-ip-address \
  --count 1 \
  --query "Instances[0].InstanceId" --output text)
echo "Instance ID: $INSTANCE_ID"

# 5. Wait for running and retrieve public DNS or IP
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
# Retrieve public DNS and IP
INSTANCE_INFO=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].[PublicDnsName,PublicIpAddress]" \
  --output text)
DNS=$(echo "$INSTANCE_INFO" | awk '{print $1}')
IP=$(echo "$INSTANCE_INFO" | awk '{print $2}')
ADDRESS=${DNS:-$IP}

echo "Instance is running!"
echo "Address: $ADDRESS"
echo "SSH in with: ssh -i ~/.ssh/mlcc-key.pem ec2-user@$ADDRESS"
