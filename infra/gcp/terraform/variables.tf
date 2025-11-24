variable "gcp_project_id" {
  type        = string
  description = "GCP Project ID"
}

variable "gcp_region" {
  type        = string
  default     = "us-central1"
  description = "GCP region for resources"
}

variable "app_name" {
  type        = string
  default     = "langchain-fastapi"
  description = "Application name"
}

variable "environment" {
  type        = string
  description = "Environment name (dev, staging, production)"
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "docker_image" {
  type        = string
  description = "Docker image URI for Cloud Run"
}

variable "db_instance_name" {
  type        = string
  default     = "langchain-db"
  description = "Cloud SQL instance name"
}

variable "db_username" {
  type        = string
  default     = "postgres"
  sensitive   = true
  description = "Database username"
}

variable "db_password" {
  type        = string
  sensitive   = true
  description = "Database password"
}

variable "redis_memory_size_gb" {
  type        = number
  default     = 2
  description = "Redis memory size in GB"
  validation {
    condition     = var.redis_memory_size_gb >= 1 && var.redis_memory_size_gb <= 300
    error_message = "Redis memory size must be between 1 and 300 GB."
  }
}

variable "cloud_run_min_instances" {
  type        = number
  default     = 1
  description = "Minimum number of Cloud Run instances"
}

variable "cloud_run_max_instances" {
  type        = number
  default     = 10
  description = "Maximum number of Cloud Run instances"
}

variable "cloud_run_cpu" {
  type        = string
  default     = "2"
  description = "Cloud Run CPU allocation"
}

variable "cloud_run_memory" {
  type        = string
  default     = "2Gi"
  description = "Cloud Run memory allocation"
}

variable "enable_public_access" {
  type        = bool
  default     = true
  description = "Enable public access to Cloud Run service"
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "Additional tags for resources"
}
