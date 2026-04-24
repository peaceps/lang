import { ChatOpenAI } from '@langchain/openai';
import { llmGateway } from './llmgw-config.json';

const gw = {
    embeddingModel: llmGateway.embeddingModel ?? 'text-embedding-3-small',
    model: llmGateway.llmgwModel ?? 'GPT5',
    baseURL: llmGateway.llmgwApiBase.replace(/\/+$/, ''),
    apiKey: llmGateway.llmgwApiKey,
    headers: {
        'api-key': llmGateway.llmgwApiKey,
        'workspacename': ('llmgwWorkspace' in llmGateway) ? llmGateway.llmgwWorkspace as string : ''
    },
    timeoutMs: (llmGateway.timeout ?? 300) * 1000, // JSON: seconds → client: ms
    temperature: llmGateway.temperature ?? 0.1,
    maxTokens: llmGateway.maxTokens ?? 8000,
    tavilyApiKey: llmGateway.tavilyApiKey
}

process.env.TAVILY_API_KEY = gw.tavilyApiKey;

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