#!/bin/bash
set -euo pipefail

ENVIRONMENT="${1:-dev}"
REGION="${2:-us-central1}"

echo "🏗️  Destroying LangChain FastAPI GCP resources..."
echo "Environment: $ENVIRONMENT"
echo "Region: $REGION"

read -p "⚠️  Are you sure you want to destroy $ENVIRONMENT resources? This cannot be undone. (yes/no): " CONFIRMATION
if [ "$CONFIRMATION" != "yes" ]; then
    echo "❌ Destruction cancelled."
    exit 1
fi

cd infra/gcp/terraform

# Destroy resources
terraform destroy \
    -var-file="environments/${ENVIRONMENT}.tfvars" \
    -auto-approve

cd ../../..

echo "✅ GCP resources destroyed successfully!"
