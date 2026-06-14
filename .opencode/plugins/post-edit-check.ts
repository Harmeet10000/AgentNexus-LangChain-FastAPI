import type { Plugin } from "@opencode-ai/plugin";

export default (async ({ $ }) => {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let dirty = false;

  const runVerification = async () => {
    timer = null;
    if (!dirty) return;
    dirty = false;

    try {
      await $`uv run ruff format src/ 2>&1 | tail -3`;
    } catch {
      // non-blocking
    }

    try {
      await $`uv run ruff check --fix --exit-zero src/ 2>&1 | tail -5`;
    } catch {
      // non-blocking
    }
  };

  return {
    "tool.execute.after": async (input: Record<string, unknown>) => {
      if (input.name === "edit" || input.name === "write") {
        dirty = true;
        if (timer) clearTimeout(timer);
        timer = setTimeout(runVerification, 1500);
      }
    },
  };
}) satisfies Plugin;
