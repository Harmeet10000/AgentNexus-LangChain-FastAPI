#!/bin/bash
set -euo pipefail

# Quick Setup Script for GCP Deployment
# This script automates the initial setup for GCP deployment

PROJECT_ID="${1:-}"
ENVIRONMENT="${2:-dev}"
REGION="${3:-us-central1}"

if [ -z "$PROJECT_ID" ]; then
    echo "Usage: $0 <PROJECT_ID> [ENVIRONMENT] [REGION]"
    echo "Example: $0 my-gcp-project production us-central1"
    exit 1
fi

echo "🚀 Quick Setup for GCP Deployment"
echo "=================================="
echo "Project ID: $PROJECT_ID"
echo "Environment: $ENVIRONMENT"
echo "Region: $REGION"
echo ""

# Step 1: Update tfvars files
echo "📝 Updating Terraform variables..."
for env_file in infra/gcp/terraform/environments/*.tfvars; do
    env_name=$(basename "$env_file" .tfvars)
    sed -i "s/your-project-id/$PROJECT_ID/g" "$env_file"
    echo "✓ Updated $(basename $env_file)"
done

# Step 2: Create backend.tf
echo ""
echo "🔧 Setting up Terraform backend..."
STATE_BUCKET="${PROJECT_ID}-terraform-state"
cat > infra/gcp/terraform/backend.tf <<EOF
terraform {
  backend "gcs" {
    bucket = "${STATE_BUCKET}"
    prefix = "langchain-fastapi"
  }
}
EOF
echo "✓ Backend configuration created"

# Step 3: Create .env template
echo ""
echo "📋 Creating environment template..."
cat > infra/gcp/.env.example <<'EOF'
# GCP Configuration
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1

# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:password@host:5432/langchain_app

# Redis
REDIS_URL=redis://:auth@host:6379

# LangChain Configuration
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
EOF
echo "✓ .env.example created"

# Step 4: Display next steps
echo ""
echo "✅ Quick setup complete!"
echo ""
echo "📚 Next Steps:"
echo "1. Update secrets in infra/gcp/terraform/environments/${ENVIRONMENT}.tfvars"
echo "2. Copy and configure .env file: cp infra/gcp/.env.example infra/gcp/.env"
echo "3. Initialize Terraform: cd infra/gcp/terraform && terraform init"
echo "4. Deploy: cd ../.. && ./infra/gcp/deploy.sh $PROJECT_ID $ENVIRONMENT $REGION"
echo ""
echo "📖 For more details, see: infra/gcp/README.md"
