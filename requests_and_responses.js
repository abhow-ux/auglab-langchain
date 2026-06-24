import 'dotenv/config';
import { ChatOpenRouter } from "@langchain/openrouter";

// ── Models ────────────────────────────────────────────────────────────────────
const planner = new ChatOpenRouter({
    model: "liquid/lfm-2.5-1.2b-thinking:free",       // plans the subtopics
    apiKey: process.env.OPENROUTER_API_KEY,
});

const explainer = new ChatOpenRouter({
    model: "poolside/laguna-xs.2:free",             // explains each subtopic
    apiKey: process.env.OPENROUTER_API_KEY,
});

const quizzer = new ChatOpenRouter({
    model: "nvidia/nemotron-3-ultra-550b-a55b:free", // generates quiz questions
    apiKey: process.env.OPENROUTER_API_KEY,
});

// ── Topic ─────────────────────────────────────────────────────────────────────
const TOPIC = "How electric motors work in electric vehicles";

console.log("=".repeat(60));
console.log("TOPIC:", TOPIC);
console.log("=".repeat(60));

// ── STEP 1 (Plan): Ask the planner model to break the topic into subtopics ────
console.log("\n[Step 1] Planning subtopics...\n");

const planResponse = await planner.invoke(
    `You are a helpful engineering tutor. A student wants to learn about: "${TOPIC}".
     Produce a JSON array of exactly 4 subtopics to cover, ordered from basic to advanced.
     Respond ONLY with a valid JSON array of strings, no explanation, no markdown, no backticks.
     Example format: ["Subtopic 1", "Subtopic 2", "Subtopic 3", "Subtopic 4"]`
);

// Parse the plan — strip any accidental markdown fences just in case
const cleanPlan = planResponse.content.replace(/```json|```/g, "").trim();
const subtopics = JSON.parse(cleanPlan);

console.log("Subtopics planned:");
subtopics.forEach((s, i) => console.log(`  ${i + 1}. ${s}`));

// ── STEP 2 (Execute): Explain each subtopic individually ──────────────────────
console.log("\n[Step 2] Explaining each subtopic...\n");

const explanations = [];
for (const subtopic of subtopics) {
    console.log(`  Explaining: "${subtopic}"...`);
    const result = await explainer.invoke(
        `You are an engineering tutor explaining concepts to a first-year university student.
         Write a clear, concise explanation (3-4 sentences) of this subtopic: "${subtopic}".
         Context: this is part of a lesson on "${TOPIC}".`
    );
    explanations.push({ subtopic, explanation: result.content });
}

// ── STEP 3 (Chain): Feed the full explanation into the quizzer ───────────────
// This is the chain — the output of step 2 feeds directly into step 3
console.log("\n[Step 3] Generating quiz questions from the explanations...\n");

const fullContent = explanations
    .map(e => `Subtopic: ${e.subtopic}\n${e.explanation}`)
    .join("\n\n");

const quizResponse = await quizzer.invoke(
    `You are an engineering instructor. Based on the following lesson content,
     write 4 multiple-choice quiz questions (one per subtopic). 
     For each question provide 4 answer choices (A, B, C, D) and mark the correct answer.
     
     Lesson content:
     ${fullContent}`
);

// ── Output ────────────────────────────────────────────────────────────────────
console.log("=".repeat(60));
console.log("FULL LESSON OUTPUT");
console.log("=".repeat(60));

explanations.forEach(({ subtopic, explanation }, i) => {
    console.log(`\n[${i + 1}] ${subtopic}`);
    console.log("-".repeat(40));
    console.log(explanation);
});

console.log("\n" + "=".repeat(60));
console.log("QUIZ QUESTIONS");
console.log("=".repeat(60));
console.log(quizResponse.content);