resource "google_secret_manager_secret" "db_url" {
  secret_id = "${var.app_name}-db-url-${var.environment}"

  labels = {
    environment = var.environment
    app         = var.app_name
  }

  depends_on = [
    google_project_service.required_apis,
  ]
}

resource "google_secret_manager_secret_version" "db_url" {
  secret      = google_secret_manager_secret.db_url.id
  secret_data = "postgresql://${var.db_username}:${urlencode(var.db_password)}@${google_sql_database_instance.postgres.private_ip_address}:5432/${google_sql_database.app_db.name}?sslmode=require"

  depends_on = [
    google_sql_database_instance.postgres,
  ]
}

resource "google_secret_manager_secret" "redis_url" {
  secret_id = "${var.app_name}-redis-url-${var.environment}"

  labels = {
    environment = var.environment
    app         = var.app_name
  }

  depends_on = [
    google_project_service.required_apis,
  ]
}

resource "google_secret_manager_secret_version" "redis_url" {
  secret      = google_secret_manager_secret.redis_url.id
  secret_data = "redis://:${urlencode(google_redis_instance.cache.auth_string)}@${google_redis_instance.cache.host}:${google_redis_instance.cache.port}"

  depends_on = [
    google_redis_instance.cache,
  ]
}

resource "google_secret_manager_secret_iam_member" "db_url_accessor" {
  secret_id = google_secret_manager_secret.db_url.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

resource "google_secret_manager_secret_iam_member" "redis_url_accessor" {
  secret_id = google_secret_manager_secret.redis_url.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}
