import 'dotenv/config';
import * as z from 'zod';
import { tool } from '@langchain/core/tools';
import { ChatOpenRouter } from '@langchain/openrouter';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';

// ─── Tool 1: Get the current ISS position ───────────────────────────────────
// Uses the free Open Notify API — no key required.
const getISSLocation = tool(
    async (_args, config) => {
        if (config?.writer) config.writer('🛸 Pinging the ISS tracker API...\n');

        const res = await fetch('http://api.open-notify.org/iss-now.json');
        const data = await res.json();

        const lat = parseFloat(data.iss_position.latitude);
        const lon = parseFloat(data.iss_position.longitude);
        const timestamp = new Date(data.timestamp * 1000).toUTCString();

        if (config?.writer) {
            config.writer(`✅ ISS located at ${lat.toFixed(4)}°, ${lon.toFixed(4)}° (as of ${timestamp})\n`);
        }

        return JSON.stringify({ latitude: lat, longitude: lon, timestamp });
    },
    {
        name: 'get_iss_location',
        description:
            'Returns the current latitude, longitude, and timestamp of the International Space Station. ' +
            'Call this first whenever someone asks about the ISS position or what is below the ISS.',
        schema: z.object({}), // No inputs needed — the API always returns the live position
    }
);

// ─── Tool 2: Get weather at given coordinates ────────────────────────────────
// Uses the free Open-Meteo API — no key required.
const getWeather = tool(
    async ({ latitude, longitude }, config) => {
        if (config?.writer) {
            config.writer(`🌍 Fetching weather for (${latitude.toFixed(4)}, ${longitude.toFixed(4)})...\n`);
        }

        const url =
            `https://api.open-meteo.com/v1/forecast` +
            `?latitude=${latitude}&longitude=${longitude}` +
            `&current=temperature_2m,wind_speed_10m,weather_code,cloud_cover` +
            `&temperature_unit=fahrenheit`;

        const res = await fetch(url);
        const data = await res.json();
        const c = data.current;

        // Translate WMO weather code to a human-readable description
        const weatherDesc = interpretWeatherCode(c.weather_code);

        if (config?.writer) config.writer(`✅ Weather data received.\n`);

        return (
            `Conditions: ${weatherDesc} | ` +
            `Temperature: ${c.temperature_2m}°F | ` +
            `Wind: ${c.wind_speed_10m} mph | ` +
            `Cloud cover: ${c.cloud_cover}%`
        );
    },
    {
        name: 'get_weather',
        description:
            'Get the current temperature, wind speed, cloud cover, and general conditions ' +
            'for any location given its latitude and longitude.',
        schema: z.object({
            latitude: z.number().describe('Latitude of the location'),
            longitude: z.number().describe('Longitude of the location'),
        }),
    }
);

// WMO weather interpretation codes → readable string
function interpretWeatherCode(code) {
    if (code === 0) return 'Clear sky';
    if (code <= 2) return 'Partly cloudy';
    if (code === 3) return 'Overcast';
    if (code <= 49) return 'Foggy / hazy';
    if (code <= 59) return 'Drizzle';
    if (code <= 69) return 'Rain';
    if (code <= 79) return 'Snow';
    if (code <= 82) return 'Rain showers';
    if (code <= 86) return 'Snow showers';
    if (code <= 99) return 'Thunderstorm';
    return 'Unknown';
}

// ─── Model setup ─────────────────────────────────────────────────────────────
const model = new ChatOpenRouter({
    model: 'nvidia/nemotron-3-ultra-550b-a55b:free',
    apiKey: process.env.OPENROUTER_API_KEY,
});

const modelWithTools = model.bindTools([getISSLocation, getWeather]);

// ─── Prompt that requires BOTH tools ─────────────────────────────────────────
const prompt =
    "Where is the International Space Station right now, and what are the current " +
    "weather conditions directly below it on Earth's surface? " +
    "Include the coordinates and give a vivid picture of what a person standing there would experience.";

console.log('━'.repeat(60));
console.log('🚀 ISS Weather Tracker');
console.log('━'.repeat(60));
console.log(`\nQuestion: ${prompt}\n`);
console.log('─'.repeat(60));

// ─── Agentic loop: keep calling tools until the model is done ─────────────────
const messages = [new HumanMessage(prompt)];
let response = await modelWithTools.invoke(messages);
messages.push(response);

// The model may chain multiple tool calls, so loop until no more are requested.
while (response.tool_calls && response.tool_calls.length > 0) {
    for (const toolCall of response.tool_calls) {
        const toolMap = {
            get_iss_location: getISSLocation,
            get_weather: getWeather,
        };

        const selectedTool = toolMap[toolCall.name];
        if (!selectedTool) {
            console.warn(`Unknown tool requested: ${toolCall.name}`);
            continue;
        }

        // Pass a writer so the tool can emit progress to stdout in real time
        const result = await selectedTool.invoke(toolCall, {
            writer: (msg) => process.stdout.write(msg),
        });

        messages.push(result);
    }

    // Ask the model for its next move (another tool call, or a final answer)
    response = await modelWithTools.invoke(messages);
    messages.push(response);
}

// ─── Stream the final answer token-by-token ───────────────────────────────────
console.log('\n' + '─'.repeat(60));
process.stdout.write('\n📡 Answer: ');

const stream = await modelWithTools.streamEvents(messages, { version: 'v2' });

for await (const event of stream) {
    if (event.event === 'on_chat_model_stream') {
        const chunk = event.data?.chunk?.content;
        if (chunk) process.stdout.write(chunk);
    }
}

console.log('\n\n' + '━'.repeat(60));