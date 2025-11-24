resource "google_compute_network" "vpc" {
  name                    = "${var.app_name}-vpc-${var.environment}"
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"

  depends_on = [
    google_project_service.required_apis,
  ]
}

resource "google_compute_subnetwork" "subnet" {
  name          = "${var.app_name}-subnet-${var.environment}"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.gcp_region
  network       = google_compute_network.vpc.id

  private_ip_google_access = true
  log_config {
    aggregation_interval = "INTERVAL_5_SEC"
    flow_sampling        = 0.5
    metadata             = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_firewall" "allow_internal" {
  name    = "${var.app_name}-allow-internal-${var.environment}"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  source_ranges = ["10.0.0.0/24"]
}

resource "google_compute_firewall" "allow_health_checks" {
  name    = "${var.app_name}-allow-health-checks-${var.environment}"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

  source_ranges = ["35.191.0.0/16", "130.211.0.0/22"]
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]

  depends_on = [
    google_project_service.required_apis,
  ]
}

resource "google_compute_global_address" "private_ip_address" {
  name          = "${var.app_name}-private-ip-${var.environment}"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc.id

  depends_on = [
    google_project_service.required_apis,
  ]
}

resource "google_compute_network_peering_routes_config" "peering_routes" {
  peering              = google_service_networking_connection.private_vpc_connection.peering
  network              = google_compute_network.vpc.name
  import_custom_routes = true
  export_custom_routes = true
}
