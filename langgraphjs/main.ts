import { getChatOpenAI } from './core/init_llmgw';
import { HumanMessage } from "@langchain/core/messages";

const model = getChatOpenAI();

const res = await model.invoke([new HumanMessage("你好")])
console.log(res);