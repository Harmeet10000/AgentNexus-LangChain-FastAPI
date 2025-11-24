gcp_project_id       = "your-project-id"
gcp_region           = "us-central1"
app_name             = "langchain-fastapi"
environment          = "dev"
docker_image         = "us-central1-docker.pkg.dev/your-project-id/langchain-fastapi-docker/langchain-fastapi:latest"
db_username          = "postgres"
db_password          = "change-me-to-strong-password"
redis_memory_size_gb = 1
cloud_run_min_instances = 1
cloud_run_max_instances = 5
cloud_run_cpu        = "1"
cloud_run_memory     = "1Gi"
enable_public_access = true

tags = {
  environment = "dev"
  managed_by  = "terraform"
  project     = "langchain-fastapi"
}
