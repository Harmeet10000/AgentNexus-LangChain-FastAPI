# GCP Infrastructure Deployment Guide

## Overview

This directory contains Terraform configuration for deploying the LangChain FastAPI application to Google Cloud Platform (GCP).

### Architecture Components

- **Cloud Run**: Serverless container runtime for the FastAPI application
- **Cloud SQL**: PostgreSQL database for application data
- **Memorystore for Redis**: In-memory data cache
- **VPC Network**: Private network for inter-service communication
- **Cloud Storage**: Object storage for documents and embeddings
- **Artifact Registry**: Docker image registry
- **Secret Manager**: Secure secret storage for credentials
- **Cloud Build**: CI/CD pipeline automation

## Prerequisites

1. **GCP Project Setup**
   ```bash
   gcloud projects create langchain-fastapi-prod
   gcloud config set project langchain-fastapi-prod
   ```

2. **Install Required Tools**
   ```bash
   # Install Terraform
   terraform -version  # Should be >= 1.0
   
   # Install Google Cloud SDK
   gcloud --version
   
   # Authenticate with GCP
   gcloud auth login
   ```

3. **Enable Billing** on your GCP project

## Project Structure

```
infra/gcp/
├── terraform/
│   ├── main.tf              # Provider and API configuration
│   ├── variables.tf         # Input variables
│   ├── outputs.tf           # Output values
│   ├── cloud_run.tf         # Cloud Run service
│   ├── database.tf          # Cloud SQL configuration
│   ├── redis.tf             # Redis cache configuration
│   ├── network.tf           # VPC and networking
│   ├── iam.tf               # Service accounts and IAM roles
│   ├── storage.tf           # GCS buckets and Artifact Registry
│   ├── secrets.tf           # Secret Manager
│   ├── backend.tf           # Terraform state backend
│   └── environments/
│       ├── dev.tfvars       # Development environment
│       ├── staging.tfvars   # Staging environment
│       └── production.tfvars # Production environment
├── cloudbuild.yaml          # Cloud Build pipeline
├── deploy.sh                # Deployment script
├── destroy.sh               # Resource cleanup script
└── outputs.sh               # Fetch Terraform outputs
```

## Deployment Steps

### 1. Prepare Environment Variables

Update the environment-specific `.tfvars` files in `infra/gcp/terraform/environments/`:

```bash
# For development
sed -i 's/your-project-id/YOUR_ACTUAL_PROJECT_ID/g' infra/gcp/terraform/environments/dev.tfvars
```

### 2. Initialize Terraform

```bash
cd infra/gcp/terraform
terraform init
terraform validate
```

### 3. Deploy Infrastructure

**Manual Deployment:**
```bash
# Plan the deployment
terraform plan -var-file="environments/dev.tfvars" -var="gcp_project_id=YOUR_PROJECT_ID"

# Apply the configuration
terraform apply -var-file="environments/dev.tfvars" -var="gcp_project_id=YOUR_PROJECT_ID"
```

**Using Deploy Script:**
```bash
cd infra/gcp
chmod +x deploy.sh
./deploy.sh YOUR_PROJECT_ID dev us-central1
```

### 4. Build and Push Docker Image

```bash
gcloud builds submit \
  --config=infra/gcp/cloudbuild.yaml \
  --region=us-central1 \
  --substitutions=_ENVIRONMENT=dev
```

### 5. Verify Deployment

```bash
# Check Cloud Run service
gcloud run services describe langchain-fastapi-dev --region=us-central1

# Check service status
gcloud run services logs read langchain-fastapi-dev --region=us-central1 --limit=50

# Get service URL
gcloud run services describe langchain-fastapi-dev --format='value(status.url)' --region=us-central1
```

## Environment Configuration

### Development (`dev`)
- Minimal resources for testing
- Single instance Cloud Run
- db-f1-micro Cloud SQL instance
- 1GB Redis cache
- Public access enabled

### Staging (`staging`)
- Standard resources
- Regional HA enabled for database
- 2GB Redis cache
- Up to 10 Cloud Run instances
- Public access enabled

### Production (`production`)
- High-availability setup
- Regional failover for database
- Read replicas enabled
- 5GB Redis cache
- Up to 50 Cloud Run instances
- Deletion protection enabled
- Enhanced monitoring and logging

## Configuration Management

### Database

The database URL is automatically constructed and stored in Secret Manager:

```
postgresql://user:password@private_ip:5432/langchain_app?sslmode=require
```

### Redis

Redis connection string stored in Secret Manager:

```
redis://:auth_token@host:port
```

### Application Secrets

Store sensitive configuration in `.env` or Secret Manager:

1. **Using Cloud Console**: Manually create secrets in Secret Manager
2. **Using gcloud CLI**:
   ```bash
   echo -n "secret_value" | gcloud secrets create SECRET_NAME --data-file=-
   ```

## Scaling Configuration

Adjust scalability in the environment tfvars files:

```hcl
cloud_run_min_instances = 1  # Minimum warm instances
cloud_run_max_instances = 50 # Maximum instances
cloud_run_cpu    = "2"       # CPU allocation
cloud_run_memory = "2Gi"     # Memory allocation
```

## Monitoring and Logging

### View Application Logs

```bash
gcloud run services logs read langchain-fastapi-dev \
  --region=us-central1 \
  --limit=100
```

### Set Up Monitoring Alerts

```bash
# Create alert policy (via Cloud Console or gcloud)
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="Cloud Run High Error Rate"
```

## Database Migrations

Run Alembic migrations via Cloud Run Jobs:

```bash
gcloud run jobs create langchain-migrate \
  --image=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/langchain-fastapi-docker/langchain-fastapi:latest \
  --region=us-central1 \
  --set-env-vars=COMMAND=migrate \
  --execute
```

## Backup and Recovery

### Database Backups

Automated backups are configured in `database.tf`:
- Daily backups retained for 7 days (production) or 1 day (dev)
- Point-in-time recovery enabled
- Transaction logs retained for 7 days (production only)

### Manual Backup

```bash
gcloud sql backups create \
  --instance=langchain-db-production \
  --description="Pre-deployment backup"
```

### Restore from Backup

```bash
gcloud sql backups restore BACKUP_ID \
  --instance=langchain-db-production
```

## Cleanup

### Destroy All Resources

```bash
cd infra/gcp
chmod +x destroy.sh
./destroy.sh production us-central1
```

Or manually:

```bash
cd infra/gcp/terraform
terraform destroy -var-file="environments/production.tfvars"
```

## Cost Estimation

### Approximate Monthly Costs (Development)

| Service | Instance Type | Estimated Cost |
|---------|--------------|-----------------|
| Cloud Run | 1 instance, 1 CPU | $5-10 |
| Cloud SQL | db-f1-micro | $10-15 |
| Redis | 1GB basic | $7-10 |
| Storage | 10GB | $0.20 |
| **Total** | | **~$25-35** |

### Approximate Monthly Costs (Production)

| Service | Instance Type | Estimated Cost |
|---------|--------------|-----------------|
| Cloud Run | 2-50 instances, 2 CPU | $100-200 |
| Cloud SQL | db-n1-standard-2, HA | $300-400 |
| Redis | 5GB standard HA | $100-150 |
| Storage | 100GB | $2-5 |
| **Total** | | **~$500-800** |

## Troubleshooting

### Common Issues

**Cloud Run can't connect to Cloud SQL**
- Check VPC connector is properly configured
- Verify IAM roles for service account
- Check Cloud SQL instance accepts connections on private IP

**Redis connection failures**
- Verify Redis private IP is accessible from Cloud Run
- Check firewall rules allow Redis traffic
- Verify auth string in Secret Manager

**Deployment failures**
- Check Cloud Build logs: `gcloud builds log BUILD_ID --stream`
- Verify Docker image exists in Artifact Registry
- Check Terraform state: `terraform state list`

### Useful Commands

```bash
# Refresh Terraform state
terraform refresh -var-file="environments/production.tfvars"

# Import existing resource
terraform import google_cloud_run_v2_service.fastapi_app projects/PROJECT_ID/locations/REGION/services/SERVICE_NAME

# Validate configuration
terraform validate

# Format Terraform files
terraform fmt -recursive
```

## Security Best Practices

1. **Never commit credentials** to Git
2. **Use Secret Manager** for all sensitive data
3. **Enable VPC Service Controls** for additional isolation
4. **Use Cloud Armor** for DDoS protection
5. **Enable Cloud SQL Auth Proxy** for database access
6. **Implement IAM conditions** for time-based access
7. **Encrypt data in transit** (TLS 1.2+)
8. **Enable audit logging** for compliance

## Additional Resources

- [Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud SQL Best Practices](https://cloud.google.com/sql/docs/postgres/best-practices)
- [GCP Security Best Practices](https://cloud.google.com/security/best-practices)
