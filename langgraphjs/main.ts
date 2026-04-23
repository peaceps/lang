
// import { shellTool } from './tools/shell-tool';
import {localShell} from './tools/local-shell-tool';
import { MyGraph } from './graph';


const system = `
你是一个智能的研究助手。使用搜索引擎来查找信息。\
你可以进行多次调用（可以同时进行，也可以按顺序进行）。\
只有在你明确知道自己想查什么时才进行搜索。\
如果在提出后续问题之前需要先查找一些信息，你也可以这样做！

重要：对**同一个用户问题**，在已经收到至少一条工具返回的观测结果后，你必须用**一条最终回复**直接回答用户，\
整合工具结果即可；不要为「核实、纠错、反复确认」而**再次调用搜索工具**，除非用户明确提出了**新的、独立的事实问题**。\
若工具返回与常识不符，可在回答中简要说明不确定性，但仍须结束本轮工具循环并给出结论。
`;

const s2 = `
You are an agent。
`;

const graph = new MyGraph(s2);
const res = await graph.invoke("你有哪些内置的tools？比如查询工具？计算工具？执行命令工具？文件查找工具？");
console.log(res);