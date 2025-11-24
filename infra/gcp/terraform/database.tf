resource "google_sql_database_instance" "postgres" {
  name             = "${var.db_instance_name}-${var.environment}"
  database_version = "POSTGRES_15"
  region           = var.gcp_region

  settings {
    tier              = var.environment == "production" ? "db-n1-standard-2" : "db-f1-micro"
    availability_type = var.environment == "production" ? "REGIONAL" : "ZONAL"

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      transaction_log_retention_days = var.environment == "production" ? 7 : 1
      backup_retention_settings {
        retained_backups = 30
        retention_unit   = "COUNT"
      }
    }

    ip_configuration {
      require_ssl        = true
      ipv4_enabled       = false
      private_network    = google_compute_network.vpc.id
      enable_private_path_for_cloudsql_cloud_sql_instances = true
    }

    database_flags {
      name  = "max_connections"
      value = var.environment == "production" ? "200" : "50"
    }

    database_flags {
      name  = "log_statement"
      value = "all"
    }

    maintenance_window {
      day          = 6  # Saturday
      hour         = 2
      update_track = "stable"
    }

    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
    }
  }

  deletion_protection = var.environment == "production" ? true : false

  depends_on = [
    google_project_service.required_apis,
    google_service_networking_connection.private_vpc_connection,
  ]
}

resource "google_sql_database" "app_db" {
  name     = "langchain_app"
  instance = google_sql_database_instance.postgres.name

  depends_on = [
    google_sql_database_instance.postgres,
  ]
}

resource "google_sql_user" "db_user" {
  name     = var.db_username
  instance = google_sql_database_instance.postgres.name
  password = var.db_password
  type     = "BUILT_IN"

  depends_on = [
    google_sql_database_instance.postgres,
  ]
}

resource "google_sql_database_instance_replica" "read_replica" {
  count                    = var.environment == "production" ? 1 : 0
  name                     = "${var.db_instance_name}-${var.environment}-replica"
  master_instance_name     = google_sql_database_instance.postgres.name
  region                   = var.gcp_region
  database_version         = "POSTGRES_15"
  replica_configuration {
    kind                   = "REPLICA"
  }
}

resource "random_password" "db_password_backup" {
  length  = 32
  special = true
}
