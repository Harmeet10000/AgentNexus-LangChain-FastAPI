#!/usr/bin/env bash
# openchamber.sh — OpenChamber CLI quick reference

# ── Basic commands ──────────────────────────────────────────────────────────
openchamber --ui-password 
openchamber status
openchamber logs
openchamber restart
openchamber stop

# ── Startup (systemd / launchd) ─────────────────────────────────────────────
openchamber startup enable
openchamber startup status
openchamber startup disable

# Headless server for remote clients
OPENCHAMBER_UI_PASSWORD='secret' openchamber startup enable
openchamber startup enable --port 3000 --api-only --host 0.0.0.0 --ui-password secret

# ── Environment variables ───────────────────────────────────────────────────
# OpenChamber snapshots the current env on `startup enable`.
# Rerun enable after changing variables you want the service to use.

# OpenChamber server
# OPENCHAMBER_HOST              Bind address (use 0.0.0.0 for remote access)
# OPENCHAMBER_UI_PASSWORD       Browser UI password (required for non-localhost)
# OPENCHAMBER_API_ONLY          true/1 → headless mode, no browser UI
# OPENCHAMBER_DATA_DIR          Override data dir (default: ~/.config/openchamber)
# OPENCHAMBER_COMPRESS_API      true/1 force on, false/0 force off
# OPENCHAMBER_SKIP_API_COMPRESSION  true/1 disables compression (takes precedence)
# OPENCHAMBER_VERBOSE_REQUEST_LOGS  true/1 enables verbose HTTP logs
# OPENCHAMBER_UPDATE_API_URL    Override update-check endpoint (rare)
# OPENCHAMBER_PACKAGE_MANAGER   Force package manager for updates

# OpenCode server
# OPENCODE_HOST                 http/https origin with port, no path (takes precedence over OPENCODE_PORT)
# OPENCODE_PORT                 Managed port, or external server port if OPENCODE_SKIP_START=true
# OPENCODE_SKIP_START           true → don't start managed OpenCode server
# OPENCHAMBER_OPENCODE_HOSTNAME Bind hostname for managed OpenCode (default: 127.0.0.1)
# OPENCODE_BINARY               Path to opencode executable
# OPENCODE_CONFIG               Path to OpenCode config file
# OPENCODE_CONFIG_DIR           Path to config dir (agents, skills, snippets)
# OPENCODE_DATA_DIR             Custom data dir for managed OpenCode
# OPENCODE_WSL_DISTRO           WSL distro for Windows integration
# OPENCHAMBER_OPENCODE_WSL_DISTRO  OpenChamber alias (OPENCODE_WSL_DISTRO takes precedence)
# OPENCODE_JWT_SECRET           Sign UI auth tokens (use long random value)

# Terminal and Git
# OPENCHAMBER_TERMINAL_SHELL    Shell for terminal sessions
# OPENCHAMBER_GIT_BINARY        Git executable for OpenChamber features
# GIT_BINARY                   Alternative Git override (prefer OPENCHAMBER_GIT_BINARY)
# OPENCHAMBER_GIT_READ_CACHE_TTL_MS  TTL in ms for cached Git reads (0 to disable)

# Voice and tunnels
# OPENAI_API_KEY               API key for voice features
# OPENCHAMBER_ALLOW_REMOTE_OPENAI_COMPAT_URLS  true/1 → allow remote base URLs
# NGROK_AUTHTOKEN              ngrok auth token

# Runtime helpers
# BUN_BINARY                   Bun executable for daemon processes
# BUN_INSTALL                  Bun root (used to find bin/bun)
# VITE_OPENCODE_URL            Build-time API base URL (rarely set)

# ── Quick start ─────────────────────────────────────────────────────────────
openchamber

# ── Tunnels ─────────────────────────────────────────────────────────────────
# Cloudflare (quick mode)
openchamber tunnel start --provider cloudflare --mode quick

# Ngrok (quick mode)
openchamber tunnel start --provider ngrok --mode quick

# Status
openchamber tunnel status

# Managed remote (Cloudflare token + hostname)
openchamber tunnel start --provider cloudflare --mode managed-remote \
  --token-file ~/.secrets/cf-token --hostname app.example.com

# Managed local (local cloudflared config)
openchamber tunnel start --provider cloudflare --mode managed-local \
  --config ~/.cloudflared/config.yml

# ── Tunnel profiles ─────────────────────────────────────────────────────────
openchamber tunnel profile add --provider cloudflare --mode managed-remote \
  --name prod-main --hostname app.example.com --token-file ~/.secrets/cf-token
openchamber tunnel start --profile prod-main

# ── Useful commands ─────────────────────────────────────────────────────────
openchamber tunnel providers
openchamber tunnel ready --provider cloudflare
openchamber tunnel ready --provider ngrok
openchamber tunnel doctor --provider cloudflare
openchamber tunnel doctor --provider ngrok
openchamber tunnel stop --port 3000
