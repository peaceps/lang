import { tools } from "@langchain/openai";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { exec } from "child_process";
import { promisify } from "util";
import { TavilySearch } from "@langchain/tavily";
import { z } from "zod";

const execAsync = promisify(exec);

/**
 * Do not use `tools.localShell` from @langchain/openai here: those are "built-in" tool shapes
 * (`type` !== "function") and force LangChain to use the OpenAI **Responses** API path, which
 * sends `role: "tool"` in a form incompatible with DashScope compatible-mode (expects
 * user|assistant|system|function for that route). A normal `function` tool uses Chat Completions.
 */
export const localShellTool = new DynamicStructuredTool({
  name: "local_shell",
  description:
    "Run a single shell command on the local machine and return combined stdout and stderr.",
  schema: z.object({
    command: z
      .string()
      .describe("Full shell command to run (e.g. on Windows: dir, on Unix: ls -la)"),
    working_directory: z
      .string()
      .optional()
      .describe("Directory to run the command in; defaults to current working directory"),
    timeout_ms: z
      .number()
      .int()
      .positive()
      .optional()
      .describe("Optional timeout in milliseconds"),
  }),
  func: async ({ command, working_directory, timeout_ms }) => {
    const { stdout, stderr } = await execAsync(command, {
      cwd: working_directory ?? process.cwd(),
      env: { ...process.env },
      timeout: timeout_ms,
    });
    return stdout + stderr;
  },
});

export const webSearchTool = tools.webSearch();

export const tavilyTool = new TavilySearch({
    // searchDepth: "advanced",
    // includeAnswer: true,
    maxResults: 3,
});