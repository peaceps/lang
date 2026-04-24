import { ToolCallingGraph } from "./graph/tool-calling-graph";

const graph = new ToolCallingGraph();
await graph.invoke_chat();