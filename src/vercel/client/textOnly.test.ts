import { expect, test } from "vitest";
import { z } from "zod/v4";
import { stepCountIs } from "ai";
import {
  Agent,
  createThread,
  createTool,
  listMessages,
  listUIMessages,
} from "../index.js";
import { mockModel } from "./mockModel.js";
import { components, initConvexTest } from "./setup.test.js";

const signature = { mock: { signature: "s".repeat(4_000) } };
const agent = new Agent(components.agent, {
  name: "text-only",
  languageModel: mockModel({
    contentSteps: [
      [
        { type: "reasoning", text: "hmm", providerMetadata: signature },
        {
          type: "tool-call",
          toolCallId: "call-1",
          toolName: "search",
          input: JSON.stringify({ query: "q" }),
        },
      ],
      [
        { type: "reasoning", text: "done", providerMetadata: signature },
        { type: "text", text: "answer" },
      ],
    ],
  }),
  tools: {
    search: createTool({
      description: "Search",
      inputSchema: z.object({ query: z.string() }),
      execute: async () => "r".repeat(20_000),
    }),
  },
  stopWhen: stepCountIs(2),
});

async function generate() {
  const t = initConvexTest();
  const threadId = await t.action(async (ctx) => {
    const threadId = await createThread(ctx, components.agent, {});
    await agent.generateText(ctx, { threadId }, { prompt: "question" });
    return threadId;
  });
  return { t, threadId };
}

const paginationOpts = { cursor: null, numItems: 10 };

test("listUIMessages with textOnly renders role and text only", async () => {
  const { t, threadId } = await generate();
  const { page } = await t.run((ctx) =>
    listUIMessages(ctx, components.agent, {
      threadId,
      paginationOpts,
      textOnly: true,
    }),
  );
  expect(page.map((m) => [m.role, m.parts])).toEqual([
    ["user", [{ type: "text", text: "question", state: "done" }]],
    ["assistant", [{ type: "text", text: "answer", state: "done" }]],
  ]);
});

test("listMessages with textOnly lists what excludeToolMessages does, as text", async () => {
  const { t, threadId } = await generate();
  const [full, text, fromAgent] = await t.run((ctx) =>
    Promise.all([
      listMessages(ctx, components.agent, {
        threadId,
        paginationOpts,
        excludeToolMessages: true,
      }),
      listMessages(ctx, components.agent, {
        threadId,
        paginationOpts,
        textOnly: true,
      }),
      agent.listMessages(ctx, { threadId, paginationOpts, textOnly: true }),
    ]),
  );
  expect(text).toEqual(fromAgent);
  expect(text.page.map((m) => m._id)).toEqual(full.page.map((m) => m._id));
  expect(text.page.map((m) => m.message)).toEqual([
    { role: "assistant", content: "answer" },
    { role: "user", content: "question" },
  ]);
  expect(full.page[0].reasoning).toBe("done");
  expect(text.page[0]).not.toHaveProperty("reasoning");
});
