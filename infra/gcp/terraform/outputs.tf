output "cloud_run_url" {
  value       = google_cloud_run_v2_service.fastapi_app.uri
  description = "Cloud Run service URL"
}

output "cloud_run_service_name" {
  value       = google_cloud_run_v2_service.fastapi_app.name
  description = "Cloud Run service name"
}

output "database_private_ip" {
  value       = google_sql_database_instance.postgres.private_ip_address
  description = "Cloud SQL private IP address"
  sensitive   = true
}

output "database_instance_name" {
  value       = google_sql_database_instance.postgres.name
  description = "Cloud SQL instance name"
}

output "redis_host" {
  value       = google_redis_instance.cache.host
  description = "Redis instance host"
  sensitive   = true
}

output "redis_port" {
  value       = google_redis_instance.cache.port
  description = "Redis instance port"
}

output "service_account_email" {
  value       = google_service_account.cloud_run_sa.email
  description = "Cloud Run service account email"
}

output "artifact_registry_repository" {
  value       = google_artifact_registry_repository.docker_repo.repository_id
  description = "Artifact Registry repository name"
}

output "vpc_network_id" {
  value       = google_compute_network.vpc.id
  description = "VPC network ID"
}

output "vpc_subnet_id" {
  value       = google_compute_subnetwork.subnet.id
  description = "VPC subnet ID"
}

output "gcs_bucket_name" {
  value       = google_storage_bucket.app_data.name
  description = "GCS bucket name"
}
