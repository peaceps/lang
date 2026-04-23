import { tools } from "@langchain/openai";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

// With execute callback for automatic command handling
export const localShell = tools.localShell({
  execute: async (action) => {
    const { command, env, working_directory, timeout_ms } = action;
    const result = await execAsync(command.join(" "), {
      cwd: working_directory ?? process.cwd(),
      env: { ...process.env, ...env },
      timeout: timeout_ms ?? undefined,
    });
    return result.stdout + result.stderr;
  },
});
