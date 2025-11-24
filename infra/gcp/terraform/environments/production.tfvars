gcp_project_id       = "your-project-id"
gcp_region           = "us-central1"
app_name             = "langchain-fastapi"
environment          = "production"
docker_image         = "us-central1-docker.pkg.dev/your-project-id/langchain-fastapi-docker/langchain-fastapi:latest"
db_username          = "postgres"
db_password          = "change-me-to-very-strong-password"
redis_memory_size_gb = 5
cloud_run_min_instances = 2
cloud_run_max_instances = 50
cloud_run_cpu        = "2"
cloud_run_memory     = "2Gi"
enable_public_access = true

tags = {
  environment = "production"
  managed_by  = "terraform"
  project     = "langchain-fastapi"
}
