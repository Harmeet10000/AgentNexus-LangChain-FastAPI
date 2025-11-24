#!/bin/bash
set -euo pipefail

PROJECT_ID="${1:-your-project-id}"
ENVIRONMENT="${2:-dev}"
REGION="${3:-us-central1}"

echo "🚀 Deploying LangChain FastAPI to GCP..."
echo "Project: $PROJECT_ID"
echo "Environment: $ENVIRONMENT"
echo "Region: $REGION"

# Set GCP project
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo "📡 Enabling required GCP APIs..."
gcloud services enable \
    run.googleapis.com \
    sqladmin.googleapis.com \
    redis.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    aiplatform.googleapis.com \
    compute.googleapis.com \
    servicenetworking.googleapis.com \
    cloudresourcemanager.googleapis.com

# Create Terraform state bucket if it doesn't exist
STATE_BUCKET="${PROJECT_ID}-terraform-state"
if ! gsutil ls "gs://${STATE_BUCKET}" &> /dev/null; then
    echo "📦 Creating Terraform state bucket: $STATE_BUCKET"
    gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://${STATE_BUCKET}"
    gsutil versioning set on "gs://${STATE_BUCKET}"
else
    echo "✓ Terraform state bucket already exists: $STATE_BUCKET"
fi

# Create Artifact Registry repository
ARTIFACT_REPO="langchain-fastapi-docker"
if ! gcloud artifacts repositories describe "$ARTIFACT_REPO" --location="$REGION" &> /dev/null; then
    echo "🏗️  Creating Artifact Registry repository: $ARTIFACT_REPO"
    gcloud artifacts repositories create "$ARTIFACT_REPO" \
        --repository-format=docker \
        --location="$REGION" \
        --description="Docker repository for LangChain FastAPI"
else
    echo "✓ Artifact Registry repository already exists: $ARTIFACT_REPO"
fi

# Initialize and configure Terraform
echo "🔧 Initializing Terraform..."
cd infra/gcp/terraform

# Update backend configuration
cat > backend.tf <<EOF
terraform {
  backend "gcs" {
    bucket = "${STATE_BUCKET}"
    prefix = "langchain-fastapi/${ENVIRONMENT}"
  }
}
EOF

# Initialize Terraform backend
terraform init -upgrade

# Validate Terraform
echo "✓ Validating Terraform configuration..."
terraform validate

# Plan Terraform deployment
echo "📋 Planning Terraform deployment..."
terraform plan \
    -var-file="environments/${ENVIRONMENT}.tfvars" \
    -var="gcp_project_id=${PROJECT_ID}" \
    -out=tfplan

# Ask for confirmation
read -p "Do you want to apply this plan? (yes/no): " CONFIRMATION
if [ "$CONFIRMATION" != "yes" ]; then
    echo "❌ Deployment cancelled."
    exit 1
fi

# Apply Terraform
echo "✨ Applying Terraform configuration..."
terraform apply tfplan

# Get outputs
echo ""
echo "🎉 Deployment complete! Outputs:"
terraform output -json | jq '.'

# Return to root directory
cd ../../..

echo ""
echo "✅ LangChain FastAPI deployed successfully to GCP!"
echo ""
echo "Next steps:"
echo "1. Push your Docker image: gcloud builds submit --config=infra/gcp/cloudbuild.yaml"
echo "2. Check Cloud Run service: gcloud run services describe langchain-fastapi-${ENVIRONMENT} --region=${REGION}"
echo "3. Monitor logs: gcloud run services logs read langchain-fastapi-${ENVIRONMENT} --region=${REGION}"
