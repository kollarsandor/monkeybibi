#!/usr/bin/env bash
#
# GCP VPC Peering Configuration for Dragonfly Cloud
# Production setup - GCP us-central1 to Dragonfly network
#

set -e

# Dragonfly Cloud Network Details (REAL VALUES)
DFCLOUD_ACCOUNT_ID="df-prod-1"
DFCLOUD_VPC_ID="dfcloud-prodv2-gqkpuckrp"
DFCLOUD_NETWORK_NAME="net_gqkpuckrp"
DFCLOUD_CIDR="192.168.0.0/16"
DFCLOUD_REGION="us-central1"
DFCLOUD_PROVIDER="gcp"

# Your GCP Project Details (TO BE CONFIGURED)
GCP_PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project-id}"
GCP_VPC_NAME="${GCP_VPC_NAME:-default}"
GCP_REGION="${GCP_REGION:-us-central1}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║     GCP VPC Peering Setup for Dragonfly Cloud                ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Dragonfly Network:"
echo "  - Account ID: $DFCLOUD_ACCOUNT_ID"
echo "  - VPC ID: $DFCLOUD_VPC_ID"
echo "  - CIDR: $DFCLOUD_CIDR"
echo "  - Region: $DFCLOUD_REGION"
echo ""

# Step 1: Verify GCP authentication
echo "🔑 Verifying GCP authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with GCP. Run: gcloud auth login"
    exit 1
fi
echo "✅ GCP authenticated"

# Step 2: Set GCP project
echo "📋 Setting GCP project to: $GCP_PROJECT_ID"
gcloud config set project "$GCP_PROJECT_ID"

# Step 3: Create VPC peering from your side to Dragonfly
PEERING_NAME="vpc-to-dragonfly-$(date +%s)"

echo "🔗 Creating VPC peering connection..."
gcloud compute networks peerings create "$PEERING_NAME" \
    --network="$GCP_VPC_NAME" \
    --peer-project="$DFCLOUD_ACCOUNT_ID" \
    --peer-network="$DFCLOUD_VPC_ID" \
    --auto-create-routes

echo "✅ VPC peering created: $PEERING_NAME"

# Step 4: Add firewall rule to allow Dragonfly traffic
FIREWALL_RULE="allow-dragonfly-traffic"

echo "🔥 Creating firewall rule for Dragonfly..."
gcloud compute firewall-rules create "$FIREWALL_RULE" \
    --network="$GCP_VPC_NAME" \
    --allow=tcp:6385 \
    --source-ranges="$DFCLOUD_CIDR" \
    --description="Allow Dragonfly DB traffic from peered network"

echo "✅ Firewall rule created: $FIREWALL_RULE"

# Step 5: Verify peering status
echo ""
echo "🔍 Checking peering status..."
gcloud compute networks peerings list --network="$GCP_VPC_NAME"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║     ✅ GCP VPC Peering Setup Complete!                        ║"
echo "║                                                                ║"
echo "║  Next Steps:                                                   ║"
echo "║  1. Accept peering in Dragonfly Cloud console                 ║"
echo "║  2. Create private endpoint datastore                         ║"
echo "║  3. Update connection strings in your app                     ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
