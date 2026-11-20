"""Temporary import compatibility for the integration-owned API roots."""

from importlib import import_module

router = import_module("app.features.profile.adapters.in.http.router").router
