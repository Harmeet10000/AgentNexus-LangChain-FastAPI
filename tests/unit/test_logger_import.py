import importlib
import sys


def test_logger_module_imports_with_contextvars() -> None:
    for module_name in [
        "app.utils.logger",
        "app.utils",
        "app.utils.cache",
        "app.utils.cache.redis_func",
    ]:
        sys.modules.pop(module_name, None)

    module = importlib.import_module("app.utils.logger")

    assert module.request_state is not None
    assert module.execution_path is not None


def test_prompt_module_imports_and_builds_prompt() -> None:
    prompts_module = importlib.import_module("app.shared.langchain_layer.prompts")
    prompt = prompts_module.SystemPromptParts(
        identity="You are a reliable assistant for testing imports.",
        objective="Answer the user's request clearly and correctly.",
    )

    assert prompt.build().startswith("IDENTITY")
