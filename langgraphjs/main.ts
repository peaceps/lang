import { ToolCallingGraph } from "./graph/tool-calling-graph";

const graph = new ToolCallingGraph();
const user = { configurable: { thread_id: "1" } };
await graph.invoke_chat(user);