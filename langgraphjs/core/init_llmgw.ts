import { ChatOpenAI } from '@langchain/openai';
import { llmGateway, } from './llmgw_config.json';

const gw = {
    embeddingModel: llmGateway.embeddingModel ?? 'text-embedding-3-small',
    model: llmGateway.llmgwModel ?? 'GPT5',
    baseURL: llmGateway.llmgwApiBase.replace(/\/+$/, ''),
    apiKey: llmGateway.llmgwApiKey,
    headers: {'api-key': llmGateway.llmgwApiKey, 'workspacename': llmGateway.llmgwWorkspace},
    timeout: llmGateway.timeout ?? 300,
    temperature: llmGateway.temperature ?? 0.1,
    maxTokens: llmGateway.maxTokens ?? 8000,
    tavilyApiKey: llmGateway.tavilyApiKey
}

export function getChatOpenAI(tools?: any[]) {
    const model = new ChatOpenAI({
        model: gw.model,
        apiKey: gw.apiKey,
        configuration: {
            baseURL: gw.baseURL.replace(/\/+$/, ''),
            defaultHeaders: gw.headers,
        },
        temperature: gw.temperature,
        maxTokens: gw.maxTokens,
        timeout: gw.timeout,
    });
    if (tools && tools.length > 0) {
        model.bindTools(tools);
    }
    return model;
}
