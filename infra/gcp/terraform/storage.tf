resource "google_storage_bucket" "app_data" {
  name          = "${var.gcp_project_id}-${var.app_name}-${var.environment}"
  location      = var.gcp_region
  force_destroy = var.environment != "production"

  uniform_bucket_level_access = true

  versioning {
    enabled = var.environment == "production"
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 3
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  logging {
    log_bucket = google_storage_bucket.app_logs.id
  }

  cors {
    origin          = ["*"]
    method          = ["GET", "POST", "PUT", "DELETE"]
    response_header = ["Content-Type", "Authorization"]
    max_age_seconds = 3600
  }

  depends_on = [
    google_project_service.required_apis,
  ]
}

resource "google_storage_bucket" "app_logs" {
  name          = "${var.gcp_project_id}-${var.app_name}-logs-${var.environment}"
  location      = var.gcp_region
  force_destroy = var.environment != "production"

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [
    google_project_service.required_apis,
  ]
}

resource "google_storage_bucket_iam_member" "app_data_viewer" {
  bucket = google_storage_bucket.app_data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

resource "google_storage_bucket_iam_member" "app_data_editor" {
  bucket = google_storage_bucket.app_data.name
  role   = "roles/storage.objectEditor"
  member = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.gcp_region
  repository_id = "${var.app_name}-docker"
  description   = "Docker repository for ${var.app_name}"
  format        = "DOCKER"

  depends_on = [
    google_project_service.required_apis,
  ]
}
