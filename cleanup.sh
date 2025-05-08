#!/usr/bin/env bash
set -e

# ————————————————————————————
# 1. Terminate all EC2 instances
# ————————————————————————————
INSTANCE_IDS=(
  i-0d29cbcbd4c217fae   # master
  i-0787cbb09cc2cfff7   # worker1
  i-0305e0c0a38cdb212   # worker2
)
echo "Terminating instances: ${INSTANCE_IDS[*]}"
aws ec2 terminate-instances --instance-ids "${INSTANCE_IDS[@]}"
aws ec2 wait instance-terminated --instance-ids "${INSTANCE_IDS[@]}"
echo "Instances terminated."

# ————————————————————————————
# 2. Delete the custom key-pair
# ————————————————————————————
echo "Deleting key-pair mlcc-key"
aws ec2 delete-key-pair --key-name mlcc-key
# (Optional) remove local .pem file:
# rm -f ~/.ssh/mlcc-key.pem

# ————————————————————————————
# 3. Delete the custom security group
# ————————————————————————————
echo "Deleting security group sg-04ee4e641db5b1220"
aws ec2 delete-security-group --group-id sg-04ee4e641db5b1220

# ————————————————————————————
# 4. Tear down the custom VPC and its networking
# ————————————————————————————
VPC_ID=vpc-0dabef63c6da78fde
SUBNET_ID=subnet-0e2ed6e236ceb0251

# Internet Gateway
IGW_ID=$(aws ec2 describe-internet-gateways \
  --filters Name=attachment.vpc-id,Values="$VPC_ID" \
  --query "InternetGateways[0].InternetGatewayId" \
  --output text)
echo "Detaching & deleting IGW $IGW_ID"
aws ec2 detach-internet-gateway --internet-gateway-id "$IGW_ID" --vpc-id "$VPC_ID"
aws ec2 delete-internet-gateway --internet-gateway-id "$IGW_ID"

# Route Table
RTB_ID=$(aws ec2 describe-route-tables \
  --filters Name=vpc-id,Values="$VPC_ID" \
  --query "RouteTables[?Associations[?Main==\`false\`]].RouteTableId" \
  --output text)
echo "Deleting route table(s): $RTB_ID"
for r in $RTB_ID; do
  aws ec2 delete-route-table --route-table-id "$r"
done

# Subnet
echo "Deleting subnet $SUBNET_ID"
aws ec2 delete-subnet --subnet-id "$SUBNET_ID"

# VPC
echo "Deleting VPC $VPC_ID"
aws ec2 delete-vpc --vpc-id "$VPC_ID"

echo "All custom AWS resources have been deleted."
