import { tools } from '@langchain/openai';
import { exec as execCallback } from 'node:child_process';
import { promisify } from 'node:util';

export const exec = promisify(execCallback);

// With execute callback for automatic command handling
export const shellTool = tools.shell({
    execute: async (action) => {
      const outputs = await Promise.all(
        action.commands.map(async (cmd) => {
          try {
            const { stdout, stderr } = await exec(cmd, {
              timeout: action.timeout_ms ?? undefined,
            });
            return {
              stdout,
              stderr,
              outcome: { type: "exit" as const, exit_code: 0 },
            };
          } catch (error: any) {
            const timedOut = error.killed && error.signal === "SIGTERM";
            return {
              stdout: error.stdout ?? "",
              stderr: error.stderr ?? String(error),
              outcome: timedOut
                ? { type: "timeout" as const }
                : { type: "exit" as const, exit_code: error.code ?? 1 },
            };
          }
        })
      );
      return {
        output: outputs,
        maxOutputLength: action.max_output_length,
      };
    },
  });