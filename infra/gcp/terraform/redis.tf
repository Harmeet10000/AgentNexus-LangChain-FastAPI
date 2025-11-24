resource "google_redis_instance" "cache" {
  name               = "${var.app_name}-redis-${var.environment}"
  tier               = var.environment == "production" ? "standard_ha" : "basic"
  memory_size_gb     = var.redis_memory_size_gb
  region             = var.gcp_region
  redis_version      = "7.2"
  display_name       = "LangChain FastAPI Redis Cache"
  connect_mode       = "PRIVATE_SERVICE_ACCESS"
  authorized_network = google_compute_network.vpc.id

  redis_configs = {
    "timeout"                     = "300"
    "maxmemory-policy"            = "allkeys-lru"
    "notify-keyspace-events"      = "Ex"
    "appendonly"                  = "yes"
    "appendfsync"                 = "everysec"
  }

  auth_enabled           = true
  transit_encryption_mode = "SERVER_AUTHENTICATION"

  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      hour = 3
    }
  }

  depends_on = [
    google_project_service.required_apis,
    google_service_networking_connection.private_vpc_connection,
  ]
}

resource "google_redis_instance_auth_string" "cache_auth" {
  redis_instance = google_redis_instance.cache.id
}
