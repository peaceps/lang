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
import { getChatOpenAI } from "./core/init-llmgw";
import { tavilyTool, localShellTool } from "./tools/tools";

const MyState = new StateSchema({
    messages: MessagesValue
});

export class MyGraph {
    private tools = [localShellTool, tavilyTool];
    private toolsMap = new Map<string, any>(this.tools.map(tool => [tool.name, tool]));
    private model: Runnable;
    private graph;

    constructor(private systemPrompt: string) {
        this.model = getChatOpenAI(this.tools);
        this.graph = new StateGraph(MyState)
            .addNode("llm", this.llmNode.bind(this))
            .addNode("tools", this.toolNode.bind(this))
            .addConditionalEdges("llm", this.shouldContinue.bind(this), ["tools", END])
            .addEdge(START, "llm")
            .addEdge("tools", "llm")
            .compile();
    }

    private async llmNode(state: ExtractStateType<typeof MyState>): Promise<ExtractUpdateType<typeof MyState>> {
        const response = await this.model.invoke([
            new SystemMessage(this.systemPrompt),
            ...state.messages
        ])
        return {messages: [response]}
    }

    private async toolNode(state: ExtractStateType<typeof MyState>): Promise<ExtractUpdateType<typeof MyState>> {
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

    private shouldContinue(state: ExtractStateType<typeof MyState>): typeof END | "tools"  {
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
        const result = await this.graph.invoke(this.formatInput(input) as ExtractStateType<typeof MyState> ?? undefined);
        console.log(result.messages.at(-1)?.content);
    }

    async invoke(input: string): Promise<any> {
        const result = this.graph.streamEvents(this.formatInput(input) as ExtractStateType<typeof MyState>, {version: "v2", streamMode: ["updates"]});
        for await (const event of result) {
            if (event.event === "on_chat_model_stream") {
                const content = event.data.chunk.content;
                if (content?.trim().length > 0) {
                    process.stdout.write(content.trim());
                }
            }
        }
    }

    private formatInput(input: string): ExtractStateType<typeof MyState> | undefined {
        return !input ? undefined : {messages: [new HumanMessage(input)]};
    }
}
