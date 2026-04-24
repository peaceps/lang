import path from 'path';
import dotenv from 'dotenv';

import { ChatOpenAI } from '@langchain/openai';


dotenv.config({ path: path.join(process.cwd(), '.env') });

const gw = {
    embeddingModel: process.env.EMBENDING_MODEL_ID,
    model: process.env.MODEL_ID || 'GPT5',
    baseURL: process.env.OPENAI_BASE_URL || '',
    apiKey: process.env.OPENAI_API_KEY || '',
    headers: {
        'api-key': process.env.OPENAI_API_KEY || '',
        'workspacename': process.env.WORKSPACE_NAME || '',
    },
    timeoutMs: 300 * 1000, // JSON: seconds → client: ms
    temperature: 0.1,
    maxTokens: 8000,
    tavilyApiKey: process.env.TAVILY_API_KEY || '',
}

export function getChatOpenAI(tools: any[] = []) {
    return new ChatOpenAI({
        model: gw.model,
        apiKey: gw.apiKey,
        // OpenAI "built-in" tools (e.g. tools.localShell) force the Responses API; non-OpenAI
        // gateways (e.g. DashScope) often break on that path. Keep Completions for compat-mode.
        useResponsesApi: false,
        configuration: {
            baseURL: gw.baseURL.replace(/\/+$/, ''),
            defaultHeaders: gw.headers,
        },
        temperature: gw.temperature,
        maxTokens: gw.maxTokens,
        timeout: gw.timeoutMs,
    }).bindTools(tools, {tool_choice: 'auto'});
}