import 'dotenv/config';
import { ChatOpenRouter } from "@langchain/openrouter";

const model = new ChatOpenRouter({
    model: "nvidia/nemotron-3-ultra-550b-a55b:free",
    apiKey: process.env.OPENROUTER_API_KEY,
});

const response = await model.invoke("Say hello!");
console.log(response.content);