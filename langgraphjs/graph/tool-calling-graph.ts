import {
    END,
    ExtractStateType,
    ExtractUpdateType,
    START,
    StateGraph,
    StateSchema,
    MessagesValue,
} from "@langchain/langgraph";
import { Runnable } from "@langchain/core/runnables";
import { SystemMessage, ToolMessage, AIMessage, HumanMessage } from "@langchain/core/messages";

import { getChatOpenAI } from "../llm/init-llmgw";
import { tavilyTool, localShellTool } from "./tools";
import { readUserInput } from "./user-input";

const DEFALT_SYSTEM_PROMPT = `
你是一个智能的研究助手。使用tavily搜索引擎来查找信息。\
你可以进行多次调用（可以同时进行，也可以按顺序进行）。\
尽可能都用tavily进行搜索以获得准确的结果。\
如果在提出后续问题之前需要先查找一些信息，你也可以这样做！

重要：对**同一个用户问题**，在已经收到至少一条工具返回的观测结果后，你必须用**一条最终回复**直接回答用户，\
整合工具结果即可。
`;

const ToolCallingState = new StateSchema({
    messages: MessagesValue
});

export class ToolCallingGraph {
    private tools = [localShellTool, tavilyTool];
    private toolsMap = new Map<string, any>(this.tools.map(tool => [tool.name, tool]));
    private model: Runnable;
    private graph;

    constructor(private systemPrompt: string = DEFALT_SYSTEM_PROMPT) {
        this.model = getChatOpenAI(this.tools);
        this.graph = new StateGraph(ToolCallingState)
            .addNode("llm", this.llmNode.bind(this))
            .addNode("tools", this.toolNode.bind(this))
            .addConditionalEdges("llm", this.shouldContinue.bind(this), ["tools", END])
            .addEdge(START, "llm")
            .addEdge("tools", "llm")
            .compile();
    }

    private async llmNode(state: ExtractStateType<typeof ToolCallingState>): Promise<ExtractUpdateType<typeof ToolCallingState>> {
        const response = await this.model.invoke([
            new SystemMessage(this.systemPrompt),
            ...state.messages
        ])
        return {messages: [response]}
    }

    private async toolNode(state: ExtractStateType<typeof ToolCallingState>): Promise<ExtractUpdateType<typeof ToolCallingState>> {
        const lastMessage = state.messages.at(-1);
        if (lastMessage == null || !AIMessage.isInstance(lastMessage)) {
          return { messages: [] };
        }
        const result: ToolMessage[] = [];
        for (const toolCall of lastMessage.tool_calls ?? []) {
          const tool: any = this.toolsMap.get(toolCall.name);
          const observation = await tool.invoke(toolCall);
          result.push(observation);
        }
        return { messages: result };
    };

    private shouldContinue(state: ExtractStateType<typeof ToolCallingState>): typeof END | "tools"  {
        const lastMessage = state.messages.at(-1);
        if (!lastMessage || !AIMessage.isInstance(lastMessage)) {
          return END;
        }
        if (lastMessage.tool_calls?.length) {
          return "tools";
        }
        return END;
    };

    async invoke_sync(input: string): Promise<any> {
        const result = await this.graph.invoke(this.formatInput(input));
        console.log(result.messages.at(-1)?.content);
    }

    async invoke(input: string): Promise<any> {
        const result = this.graph.streamEvents(this.formatInput(input), {version: "v2", streamMode: ["updates"]});
        for await (const event of result) {
            if (event.event === "on_chat_model_stream") {
                const content = event.data.chunk.content;
                if (content?.trim().length > 0) {
                    process.stdout.write(content.trim());
                }
            }
        }
    }

    async invoke_chat(): Promise<void> {
        console.log("输入你想问的内容，或者输入'exit'退出");
        const {read, close} = readUserInput();
        while (true) {
            const input = await read(">>> ");
            if (input === "exit") {
                break;
            }
            await this.invoke(input);
        }
        await close();
        console.log("再见！");
    }

    private formatInput(input: string): ExtractStateType<typeof ToolCallingState> {
        return {messages: [new HumanMessage(input)]};
    }
}
