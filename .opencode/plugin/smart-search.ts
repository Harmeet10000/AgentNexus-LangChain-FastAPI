import type { Plugin } from "@opencode-ai/plugin"
import { existsSync } from "node:fs"
import { join } from "node:path"

export default (async ({ directory }) => {
  const graphExists = existsSync(join(directory, "graphify-out", "graph.json"))

  if (!graphExists) return {}

  return {
    "tool.execute.before": async (input, output) => {
      if (input.tool === "grep") {
        const pattern = String(output.args?.pattern ?? output.args?.query ?? "")
        const explorationHints = [
          "architecture", "structure", "component", "module",
          "service", "repository", "router", "how does", "where is",
          "relationship", "dependency", "flow",
        ]
        const isExploration = explorationHints.some((h) =>
          pattern.toLowerCase().includes(h)
        )
        if (isExploration) {
          output.args = {
            ...output.args,
            _hint: `[graphify available] Consider running graphify query "${pattern}" first for scoped results (~200 tokens vs ~2K+ for grep+read). Use grep as fallback.`,
          }
        }
      }
    },
  }
}) satisfies Plugin
