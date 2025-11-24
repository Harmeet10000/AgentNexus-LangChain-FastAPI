#!/bin/bash
set -euo pipefail

ENVIRONMENT="${1:-dev}"
REGION="${2:-us-central1}"

echo "📊 Getting Terraform outputs for $ENVIRONMENT..."

cd infra/gcp/terraform

terraform output \
    -var-file="environments/${ENVIRONMENT}.tfvars"

cd ../../..
