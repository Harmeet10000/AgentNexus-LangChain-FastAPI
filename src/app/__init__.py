"""LangChain FastAPI Production - Main application package."""

from importlib import import_module


def __getattr__(name: str):
    if name == "app":
        return import_module("app.main").app
    raise AttributeError(name)
