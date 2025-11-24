resource "google_cloud_run_v2_service" "fastapi_app" {
  name     = "${var.app_name}-${var.environment}"
  location = var.gcp_region
  protocol = "HTTP2"

  template {
    max_retries = 1
    timeout     = "300s"

    scaling {
      min_instance_count = var.cloud_run_min_instances
      max_instance_count = var.cloud_run_max_instances
    }

    containers {
      image = var.docker_image

      ports {
        container_port = 8000
      }

      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }

      env {
        name  = "LOG_LEVEL"
        value = var.environment == "production" ? "INFO" : "DEBUG"
      }

      env {
        name = "DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_url.id
            version = "latest"
          }
        }
      }

      env {
        name = "REDIS_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.redis_url.id
            version = "latest"
          }
        }
      }

      env {
        name  = "GCP_PROJECT_ID"
        value = var.gcp_project_id
      }

      resources {
        limits = {
          cpu    = var.cloud_run_cpu
          memory = var.cloud_run_memory
        }
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        failure_threshold = 3
        period_seconds    = 10
        timeout_seconds   = 5
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        failure_threshold = 3
        period_seconds    = 30
        timeout_seconds   = 5
      }
    }

    service_account = google_service_account.cloud_run_sa.email

    vpc_access {
      connector = google_vpc_access_connector.connector.id
      egress    = "ALL_TRAFFIC"
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_project_service.required_apis,
    google_sql_database_instance.postgres,
    google_redis_instance.cache,
    google_secret_manager_secret_version.db_url,
    google_secret_manager_secret_version.redis_url,
  ]
}

resource "google_cloud_run_service_iam_binding" "public_access" {
  count    = var.enable_public_access ? 1 : 0
  service  = google_cloud_run_v2_service.fastapi_app.name
  location = google_cloud_run_v2_service.fastapi_app.location
  role     = "roles/run.invoker"

  members = [
    "allUsers",
  ]
}

resource "google_vpc_access_connector" "connector" {
  name          = "${var.app_name}-vpc-connector-${var.environment}"
  region        = var.gcp_region
  ip_cidr_range = "10.8.0.0/28"
  network       = google_compute_network.vpc.name
  depends_on = [
    google_project_service.required_apis,
  ]
}
