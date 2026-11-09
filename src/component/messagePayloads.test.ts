/// <reference types="vite/client" />

import { omit, pick } from "convex-helpers";
import { convexTest } from "convex-test";
import { convexToJson, getDocumentSize, type Value } from "convex/values";
import { describe, expect, test } from "vitest";
import { deriveReasoning, extractText } from "../shared.js";
import type { Message, MessageDoc } from "../validators.js";
import { docsToModelMessages } from "../vercel/mapping.js";
import { api } from "./_generated/api.js";
import type { Doc, Id } from "./_generated/dataModel.js";
import schema from "./schema.js";
import { initConvexTest, modules } from "./setup.test.js";

type T = ReturnType<typeof initConvexTest>;
type Ctx = Parameters<Parameters<T["run"]>[0]>[0];

const SIGNATURE = "s".repeat(6_000);
const toolInput = { query: "q".repeat(2_000) };
const heavyAssistant = {
  message: {
    role: "assistant" as const,
    content: [
      {
        type: "reasoning" as const,
        text: "thinking",
        providerOptions: { openai: { reasoningEncryptedContent: SIGNATURE } },
      },
      { type: "text" as const, text: "the answer" },
    ],
  },
  providerMetadata: { google: { groundingMetadata: "g".repeat(4_000) } },
  usage: { promptTokens: 10, completionTokens: 5, raw: { input_tokens: 10 } },
  warnings: [{ type: "other" as const, message: "careful" }],
  sources: [
    {
      type: "source" as const,
      sourceType: "url" as const,
      id: "s1",
      url: "https://example.com",
    },
  ],
  finishReason: "stop" as const,
  model: "gpt-5",
  provider: "openai",
};
const heavyToolCall = {
  message: {
    role: "assistant" as const,
    content: [
      {
        type: "tool-call" as const,
        toolCallId: "call-1",
        toolName: "search",
        input: toolInput,
        args: toolInput,
      },
    ],
  },
};
const heavyToolResult = {
  message: {
    role: "tool" as const,
    content: [
      {
        type: "tool-result" as const,
        toolCallId: "call-1",
        toolName: "search",
        output: { type: "text" as const, value: "r".repeat(20_000) },
      },
    ],
  },
};
const user = { message: { role: "user" as const, content: "question" } };

const grounded = (bytes: number) => ({
  ...heavyAssistant,
  providerMetadata: { google: { grounding: "g".repeat(bytes) } },
});

async function createThread(t: T, userId?: string) {
  const thread = await t.mutation(api.threads.createThread, { userId });
  return thread._id as Id<"threads">;
}

async function setup(t: T, userId = "u") {
  const threadId = await createThread(t, userId);
  const { messages } = await t.mutation(api.messages.addMessages, {
    threadId,
    messages: [user, heavyToolCall, heavyToolResult, heavyAssistant],
  });
  return { threadId, messages };
}

const list = (
  t: T,
  threadId: Id<"threads">,
  opts: {
    view?: "text" | "model" | "full";
    excludeToolMessages?: boolean;
  } = {},
) =>
  t.query(api.messages.listMessagesByThreadId, {
    threadId,
    order: "asc",
    paginationOpts: { cursor: null, numItems: 100 },
    ...opts,
  });

const rawMessages = (t: T) =>
  t.run((ctx) => ctx.db.query("messages").collect());

async function measure<R>(t: T, fn: (ctx: Ctx) => Promise<R>) {
  return t.run(async (ctx) => {
    const before = await ctx.meta.getTransactionMetrics();
    const result = await fn(ctx);
    const after = await ctx.meta.getTransactionMetrics();
    return {
      result,
      bytesRead: after.bytesRead.used - before.bytesRead.used,
      bytesWritten: after.bytesWritten.used - before.bytesWritten.used,
    };
  });
}

// Each tier row belongs to one message, exists only while it holds a field,
// and nothing is left behind by a write that removed or replaced a message.
async function expectConsistentStorage(t: T) {
  await t.run(async (ctx) => {
    const messages = await ctx.db.query("messages").collect();
    for (const table of ["messagePayloads", "messageRecords"] as const) {
      const linked = messages.flatMap((m) => {
        const id = table === "messagePayloads" ? m.payloadId : m.recordId;
        return id ? [id as string] : [];
      });
      expect(new Set(linked).size).toBe(linked.length);
      const rows = await ctx.db.query(table).collect();
      expect(rows.map((r) => r._id as string).sort()).toEqual(linked.sort());
      for (const row of rows) {
        expect(
          Object.values(omit(row, ["_id", "_creationTime"])).some(
            (value) => value !== undefined,
          ),
        ).toBe(true);
      }
    }
  });
}

// What a message read as a row saved before this storage change returns.
async function readAsUnsplitRow(t: T, doc: MessageDoc) {
  const threadId = await createThread(t);
  const id = await t.run((ctx) =>
    ctx.db.insert("messages", {
      ...omit(doc, ["_id", "_creationTime", "threadId", "embeddingId"]),
      threadId,
      fileIds: doc.fileIds as Id<"files">[] | undefined,
    } as Omit<Doc<"messages">, "_id" | "_creationTime">),
  );
  const [read] = await t.query(api.messages.getMessagesByIds, {
    messageIds: [id],
  });
  return omit(read!, ["_id", "_creationTime", "threadId"]);
}

const asModel = (doc: MessageDoc) =>
  omit(doc, ["providerMetadata", "usage", "sources", "warnings"]);

const comparable = (doc: MessageDoc) =>
  omit(doc, ["_id", "_creationTime", "threadId"]);

describe("message rows stay small", () => {
  test("a heavy message's row holds about what a text view needs", async () => {
    const t = initConvexTest();
    await setup(t);
    const rows = await rawMessages(t);
    for (const stepOrder of [1, 2, 3]) {
      const row = rows.find((m) => m.stepOrder === stepOrder)!;
      expect(getDocumentSize(row)).toBeLessThan(800);
    }
  });

  test("every new message keeps its content apart, whatever its size", async () => {
    const t = initConvexTest();
    await setup(t);
    const rows = await rawMessages(t);
    for (const row of rows) {
      expect(row.message).toBeUndefined();
      expect(row.payloadId).toBeDefined();
    }
    expect(rows.map((row) => row.recordId !== undefined)).toEqual([
      false,
      false,
      false,
      true,
    ]);
    await expectConsistentStorage(t);
  });

  test("a saved message records no parent message id", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    const { messages } = await t.mutation(api.messages.addMessages, {
      threadId,
      messages: [user],
    });
    await t.mutation(api.messages.addMessages, {
      threadId,
      promptMessageId: messages[0]._id as Id<"messages">,
      messages: [{ message: { role: "assistant", content: "hi" } }],
    });
    for (const row of await rawMessages(t)) {
      expect(row).not.toHaveProperty("parentMessageId");
    }
  });
});

describe("full reads return the whole message", () => {
  test("every read path returns what a row saved whole would", async () => {
    const t = initConvexTest();
    const { threadId, messages } = await setup(t);
    const { page } = await list(t, threadId);
    expect(page).toEqual(messages);
    expect(page[3]).toMatchObject({ ...heavyAssistant, text: "the answer" });
    expect(page[2].message).toEqual(heavyToolResult.message);
    for (const doc of page) {
      expect(comparable(doc)).toEqual(await readAsUnsplitRow(t, doc));
    }
    const byId = await t.query(api.messages.getMessagesByIds, {
      messageIds: messages.map((m) => m._id as Id<"messages">),
    });
    expect(byId).toEqual(messages);
  });

  test("rows saved before this change read whole, and as text", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    await t.run((ctx) =>
      ctx.db.insert("messages", {
        ...heavyAssistant,
        threadId,
        order: 0,
        stepOrder: 0,
        status: "success",
        tool: false,
        text: "the answer",
      }),
    );
    const full = await list(t, threadId);
    expect(full.page[0]).toMatchObject(heavyAssistant);
    const model = await list(t, threadId, { view: "model" });
    expect(model.page[0]).toEqual(full.page[0]);
    const text = await list(t, threadId, { view: "text" });
    expect(text.page[0].message).toEqual({
      role: "assistant",
      content: "the answer",
    });
    expect(text.page[0]).not.toHaveProperty("providerMetadata");
  });
});

describe("text-only reads", () => {
  test("return each non-tool message as its role and text", async () => {
    const t = initConvexTest();
    const { threadId } = await setup(t);
    const { page } = await list(t, threadId, { view: "text" });
    expect(page.map((m) => m.message)).toEqual([
      { role: "user", content: "question" },
      { role: "assistant", content: "the answer" },
    ]);
    for (const field of [
      "providerMetadata",
      "usage",
      "sources",
      "warnings",
      "reasoning",
      "reasoningDetails",
    ]) {
      expect(page[1]).not.toHaveProperty(field);
    }
    expect(page[1]).toMatchObject({
      finishReason: "stop",
      model: "gpt-5",
      text: "the answer",
    });
  });

  test("give the text a full read would extract", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    await t.mutation(api.messages.addMessages, {
      threadId,
      messages: [
        user,
        {
          message: {
            role: "user",
            content: [
              { type: "text", text: "look " },
              {
                type: "image",
                image: "i".repeat(3_000),
                mediaType: "image/png",
              },
              { type: "text", text: "here" },
            ],
          },
        },
        heavyAssistant,
        {
          message: {
            role: "assistant",
            content: [
              {
                type: "reasoning",
                text: "only thinking",
                providerOptions: { a: { signature: SIGNATURE } },
              },
            ],
          },
        },
      ],
    });
    const hidden = await t.mutation(api.messages.addMessages, {
      threadId,
      hideFromUserIdSearch: true,
      messages: [heavyAssistant],
    });
    expect(hidden.messages[0].text).toBeUndefined();
    const full = await list(t, threadId);
    const text = await list(t, threadId, { view: "text" });
    expect(text.page.map((m) => m._id)).toEqual(full.page.map((m) => m._id));
    expect(text.page.map((m) => m.message)).toEqual(
      full.page.map((m) => ({
        role: m.message!.role,
        content: extractText(m.message!) ?? "",
      })),
    );
  });

  test("skip tool messages, as excludeToolMessages does", async () => {
    const t = initConvexTest();
    const { threadId, messages } = await setup(t);
    const prompt = messages[0]._id as Id<"messages">;
    const text = await t.query(api.messages.listMessagesByThreadId, {
      threadId,
      order: "desc",
      view: "text",
      upToAndIncludingMessageId: messages[3]._id as Id<"messages">,
    });
    expect(text.page.map((m) => m._id)).toEqual(
      [messages[3]._id, prompt].map(String),
    );
  });

  test("a text page over heavy messages reads a fraction of a full page", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t, "u");
    for (let i = 0; i < 5; i++) {
      await t.mutation(api.messages.addMessages, {
        threadId,
        messages: [user, heavyToolCall, heavyToolResult, heavyAssistant],
      });
    }
    const args = {
      threadId,
      order: "desc" as const,
      paginationOpts: { cursor: null, numItems: 20 },
    };
    const full = await measure(t, (ctx) =>
      ctx.runQuery(api.messages.listMessagesByThreadId, args),
    );
    const excluded = await measure(t, (ctx) =>
      ctx.runQuery(api.messages.listMessagesByThreadId, {
        ...args,
        excludeToolMessages: true,
      }),
    );
    const text = await measure(t, (ctx) =>
      ctx.runQuery(api.messages.listMessagesByThreadId, {
        ...args,
        view: "text",
      }),
    );
    expect(text.result.page).toHaveLength(10);
    expect(text.bytesRead).toBeLessThan(excluded.bytesRead / 5);
    expect(text.bytesRead).toBeLessThan(full.bytesRead / 20);
  });
});

describe("model reads", () => {
  test("return what the model reads, without what only a full read needs", async () => {
    const t = initConvexTest();
    const { threadId, messages } = await setup(t);
    const { page } = await list(t, threadId, { view: "model" });
    expect(page).toEqual(messages.map(asModel));
  });

  test("search returns messages and their neighbours as the model reads them", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t, "u");
    const vector = Array.from({ length: 128 }, (_, i) => (i === 0 ? 1 : 0));
    const { messages } = await t.mutation(api.messages.addMessages, {
      threadId,
      messages: [user, heavyAssistant, { message: user.message }],
      embeddings: {
        model: "m",
        dimension: 128,
        vectors: [null, vector, null],
      },
    });
    const found = await t.action(api.messages.searchMessages, {
      threadId,
      embedding: vector,
      embeddingModel: "m",
      vectorSearch: true,
      limit: 1,
    });
    expect(found).toEqual(messages.slice(0, 3).map(asModel));
    const byText = await t.query(api.messages.textSearch, {
      threadId,
      text: "answer",
      limit: 10,
    });
    expect(byText).toEqual([asModel(messages[1])]);
  });

  test("give the model what a full read gives it", async () => {
    const t = initConvexTest();
    const { threadId, messages } = await setup(t);
    await t.mutation(api.messages.updateMessage, {
      messageId: messages[3]._id as Id<"messages">,
      patch: { providerOptions: { openai: { store: false } } },
    });
    const model = await list(t, threadId, { view: "model" });
    const full = await list(t, threadId);
    expect(docsToModelMessages(model.page)).toEqual(
      docsToModelMessages(full.page),
    );
  });

  test("a model page over heavy metadata reads a fraction of a full page", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    for (let i = 0; i < 5; i++) {
      await t.mutation(api.messages.addMessages, {
        threadId,
        messages: [user, grounded(50_000)],
      });
    }
    const args = {
      threadId,
      order: "desc" as const,
      paginationOpts: { cursor: null, numItems: 10 },
    };
    const full = await measure(t, (ctx) =>
      ctx.runQuery(api.messages.listMessagesByThreadId, args),
    );
    const model = await measure(t, (ctx) =>
      ctx.runQuery(api.messages.listMessagesByThreadId, {
        ...args,
        view: "model",
      }),
    );
    expect(model.result.page).toHaveLength(10);
    expect(model.bytesRead).toBeLessThan(full.bytesRead / 5);
  });
});

describe("saves and updates stay cheap", () => {
  test("a save after heavy messages reads only small rows", async () => {
    const t = initConvexTest();
    const { threadId } = await setup(t);
    const { bytesRead } = await measure(t, (ctx) =>
      ctx.runMutation(api.messages.addMessages, {
        threadId,
        messages: [{ message: { role: "assistant", content: "next" } }],
      }),
    );
    expect(bytesRead).toBeLessThan(3_000);
  });

  test("a status update rewrites only the small row", async () => {
    const t = initConvexTest();
    const { messages } = await setup(t);
    const { bytesWritten, result } = await measure(t, (ctx) =>
      ctx.runMutation(api.messages.updateMessage, {
        messageId: messages[3]._id as Id<"messages">,
        patch: { status: "failed", error: "nope" },
      }),
    );
    expect(bytesWritten).toBeLessThan(1_000);
    expect(result).toEqual({ ...messages[3], status: "failed", error: "nope" });
  });
});

describe("writes keep payloads in step", () => {
  test("deleting by id, by order, by thread and by user", async () => {
    const t = initConvexTest();
    const byId = await setup(t);
    await t.mutation(api.messages.deleteByIds, {
      messageIds: byId.messages.map((m) => m._id as Id<"messages">),
    });
    await expectConsistentStorage(t);

    const byOrder = await setup(t);
    await t.mutation(api.messages.deleteByOrder, {
      threadId: byOrder.threadId,
      startOrder: 0,
      endOrder: 1,
    });
    await expectConsistentStorage(t);

    const byThread = await setup(t);
    await t.action(api.threads.deleteAllForThreadIdSync, {
      threadId: byThread.threadId,
    });
    await expectConsistentStorage(t);

    await setup(t, "gone");
    await setup(t, "gone");
    await t.action(api.users.deleteAllForUserId, { userId: "gone" });
    await expectConsistentStorage(t);
    expect(await rawMessages(t)).toEqual([]);
  });

  test("updating content keeps the other fields and the storage consistent", async () => {
    const t = initConvexTest();
    const { threadId, messages } = await setup(t);
    const big = {
      role: "assistant" as const,
      content: [{ type: "text" as const, text: "x".repeat(5_000) }],
    };
    const small = { role: "assistant" as const, content: "short" };
    const row = async () =>
      (await rawMessages(t)).find((m) => m._id === messages[3]._id)!;
    const tiers = pick(await row(), ["payloadId", "recordId"]);
    for (const message of [big, small, heavyAssistant.message]) {
      const updated = await t.mutation(api.messages.updateMessage, {
        messageId: messages[3]._id as Id<"messages">,
        patch: { message },
      });
      expect(updated).toEqual({
        ...omit(messages[3], ["reasoning", "reasoningDetails"]),
        ...deriveReasoning(message),
        message,
        text: extractText(message),
      });
      expect((await list(t, threadId)).page[3]).toEqual(updated);
      expect(pick(await row(), ["payloadId", "recordId"])).toEqual(tiers);
      await expectConsistentStorage(t);
    }
    const grown = await t.mutation(api.messages.updateMessage, {
      messageId: messages[0]._id as Id<"messages">,
      patch: {
        message: {
          role: "user",
          content: [
            { type: "text", text: "see" },
            { type: "image", image: "i".repeat(5_000), mediaType: "image/png" },
          ],
        },
      },
    });
    expect((await list(t, threadId)).page[0]).toEqual(grown);
    await expectConsistentStorage(t);
  });

  test("a clone reads like its source and outlives it", async () => {
    const t = initConvexTest();
    const { threadId, messages } = await setup(t);
    const targetThreadId = await createThread(t);
    await t.action(api.messages.cloneThread, {
      sourceThreadId: threadId,
      targetThreadId,
    });
    await t.action(api.threads.deleteAllForThreadIdSync, { threadId });
    const { page } = await list(t, targetThreadId);
    expect(page.map(comparable)).toEqual(messages.map(comparable));
    await expectConsistentStorage(t);
  });

  test("a clone of a row saved whole is stored as a save stores it", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    await t.run((ctx) =>
      ctx.db.insert("messages", {
        ...heavyAssistant,
        threadId,
        order: 0,
        stepOrder: 0,
        status: "success",
        tool: false,
        text: "the answer",
      }),
    );
    const source = (await list(t, threadId)).page;
    const targetThreadId = await createThread(t);
    await t.action(api.messages.cloneThread, {
      sourceThreadId: threadId,
      targetThreadId,
    });
    const [copy] = await t.run((ctx) =>
      ctx.db
        .query("messages")
        .withIndex("threadId_status_tool_order_stepOrder", (q) =>
          q.eq("threadId", targetThreadId),
        )
        .collect(),
    );
    expect(copy).not.toHaveProperty("message");
    expect(copy).not.toHaveProperty("providerMetadata");
    expect(copy).toMatchObject({ role: "assistant", text: "the answer" });
    expect((await list(t, targetThreadId)).page.map(comparable)).toEqual(
      source.map(comparable),
    );
    await expectConsistentStorage(t);
  });

  test("finalizing a heavy pending message keeps its content", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    const { messages } = await t.mutation(api.messages.addMessages, {
      threadId,
      messages: [
        user,
        { ...heavyAssistant, status: "pending" },
        { ...heavyAssistant, status: "pending" },
      ],
    });
    await t.mutation(api.messages.finalizeMessage, {
      messageId: messages[1]._id as Id<"messages">,
      result: { status: "success" },
    });
    await t.mutation(api.messages.finalizeMessage, {
      messageId: messages[2]._id as Id<"messages">,
      result: { status: "failed", error: "boom" },
    });
    const { page } = await list(t, threadId);
    expect(page[1]).toEqual({ ...messages[1], status: "success" });
    expect(page[2]).toEqual({
      ...messages[2],
      status: "failed",
      error: "boom",
    });
    await expectConsistentStorage(t);
  });

  test("replacing a pending message keeps one payload", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    const { messages } = await t.mutation(api.messages.addMessages, {
      threadId,
      messages: [user, { ...heavyAssistant, status: "pending" }],
    });
    const pendingMessageId = messages[1]._id as Id<"messages">;
    for (const replacement of [
      heavyToolCall,
      { message: { role: "assistant" as const, content: "small" } },
      heavyAssistant,
    ]) {
      await t.mutation(api.messages.updateMessage, {
        messageId: pendingMessageId,
        patch: { status: "pending" },
      });
      const saved = await t.mutation(api.messages.addMessages, {
        threadId,
        pendingMessageId,
        messages: [replacement],
      });
      expect(saved.messages[0]).toMatchObject({
        _id: pendingMessageId,
        message: replacement.message,
      });
      expect((await list(t, threadId)).page[1]).toEqual(saved.messages[0]);
      await expectConsistentStorage(t);
    }
  });

  test("failing pending steps keeps their content", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    const { messages } = await t.mutation(api.messages.addMessages, {
      threadId,
      messages: [user, { ...heavyAssistant, status: "pending" }],
    });
    await t.mutation(api.messages.addMessages, {
      threadId,
      promptMessageId: messages[0]._id as Id<"messages">,
      failPendingSteps: true,
      messages: [{ message: { role: "assistant", content: "retry" } }],
    });
    const { page } = await list(t, threadId);
    expect(page[1]).toEqual({
      ...messages[1],
      status: "failed",
      error: "Restarting",
    });
  });

  test("approvals find a request whose message is stored apart", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    const { messages } = await t.mutation(api.messages.addMessages, {
      threadId,
      messages: [
        user,
        {
          message: {
            role: "assistant",
            content: [
              ...heavyToolCall.message.content,
              {
                type: "tool-approval-request",
                approvalId: "a1",
                toolCallId: "call-1",
              },
            ],
          },
        },
      ],
    });
    const { messageId } = await t.mutation(
      api.messages.respondToToolCallApprovals,
      { threadId, decisions: [{ approvalId: "a1", approved: true }] },
    );
    const [response] = await t.query(api.messages.getMessagesByIds, {
      messageIds: [messageId],
    });
    expect(response).toMatchObject({
      order: messages[1].order,
      message: {
        role: "tool",
        content: [{ type: "tool-approval-response", approvalId: "a1" }],
      },
    });
  });
});

test("approvals read no content but the request's", async () => {
  const t = initConvexTest();
  const threadId = await createThread(t);
  await t.mutation(api.messages.addMessages, {
    threadId,
    messages: [
      user,
      {
        message: {
          role: "assistant",
          content: [
            ...heavyToolCall.message.content,
            {
              type: "tool-approval-request",
              approvalId: "a1",
              toolCallId: "call-1",
            },
          ],
        },
      },
      ...Array.from({ length: 20 }, () => heavyToolResult),
    ],
  });
  const { bytesRead } = await measure(t, (ctx) =>
    ctx.runMutation(api.messages.respondToToolCallApprovals, {
      threadId,
      decisions: [{ approvalId: "a1", approved: true }],
    }),
  );
  expect(bytesRead).toBeLessThan(
    heavyToolResult.message.content[0].output.value.length,
  );
});

test("approvals find a request in a row saved whole", async () => {
  const t = initConvexTest();
  const threadId = await createThread(t);
  await t.run((ctx) =>
    ctx.db.insert("messages", {
      message: {
        role: "assistant",
        content: [
          {
            type: "tool-approval-request",
            approvalId: "a1",
            toolCallId: "call-1",
          },
        ],
      },
      threadId,
      order: 0,
      stepOrder: 0,
      status: "success",
      tool: true,
    }),
  );
  const { messageId } = await t.mutation(
    api.messages.respondToToolCallApprovals,
    { threadId, decisions: [{ approvalId: "a1", approved: true }] },
  );
  expect(messageId).toBeDefined();
  await expect(
    t.mutation(api.messages.respondToToolCallApprovals, {
      threadId,
      decisions: [{ approvalId: "a1", approved: false }],
    }),
  ).rejects.toThrow(/already handled/);
});

test("approvals find a request saved in its own message", async () => {
  const t = initConvexTest();
  const threadId = await createThread(t);
  await t.mutation(api.messages.addMessages, {
    threadId,
    messages: [
      user,
      heavyToolCall,
      {
        message: {
          role: "assistant",
          content: [
            {
              type: "tool-approval-request",
              approvalId: "a1",
              toolCallId: "call-1",
              providerOptions: { a: { signature: SIGNATURE } },
            },
          ],
        },
      },
    ],
  });
  const { messageId } = await t.mutation(
    api.messages.respondToToolCallApprovals,
    { threadId, decisions: [{ approvalId: "a1", approved: true }] },
  );
  expect(messageId).toBeDefined();
});

// The thread's messages again, in a new thread, as rows saved whole.
async function copyAsWholeRows(t: T, threadId: Id<"threads">) {
  const whole = await createThread(t);
  for (const doc of (await list(t, threadId)).page) {
    await t.run((ctx) =>
      ctx.db.insert("messages", {
        ...omit(doc, ["_id", "_creationTime", "threadId"]),
        threadId: whole,
      } as Omit<Doc<"messages">, "_id" | "_creationTime">),
    );
  }
  return whole;
}

// A page by the positions it holds, ends and splits at, which a thread and
// its whole-row copy share.
async function budgetedPage(
  t: T,
  threadId: Id<"threads">,
  paginationOpts: {
    cursor: string | null;
    endCursor?: string;
    maximumBytesRead?: number;
    numItems: number;
  },
) {
  const at = (cursor?: string) => cursor && JSON.parse(cursor).slice(0, 2);
  const result = await t.query(api.messages.listMessagesByThreadId, {
    threadId,
    order: "asc",
    paginationOpts,
  });
  return {
    positions: result.page.map((m) => [m.order, m.stepOrder]),
    isDone: result.isDone,
    pageStatus: result.pageStatus,
    ends: at(result.continueCursor),
    splitsAt: at(result.splitCursor ?? undefined),
    continueCursor: result.continueCursor,
  };
}

async function budgetedPages(
  t: T,
  threadId: Id<"threads">,
  opts: { numItems: number; maximumBytesRead: number },
) {
  const pages = [];
  let cursor: string | null = null;
  for (let i = 0; i < 10; i++) {
    const page = await budgetedPage(t, threadId, { cursor, ...opts });
    pages.push(omit(page, ["continueCursor"]));
    if (page.isDone) break;
    cursor = page.continueCursor;
  }
  return pages;
}

async function boundedPage(
  t: T,
  threadId: Id<"threads">,
  opts: { numItems: number; maximumBytesRead: number },
) {
  const range = await budgetedPage(t, threadId, {
    cursor: null,
    numItems: opts.numItems,
  });
  const page = await budgetedPage(t, threadId, {
    cursor: null,
    endCursor: range.continueCursor,
    ...opts,
  });
  return omit(page, ["continueCursor"]);
}

test("a byte budget ends a full page where it ends rows saved whole", async () => {
  const t = initConvexTest();
  const split = await createThread(t);
  for (let i = 0; i < 5; i++) {
    await t.mutation(api.messages.addMessages, {
      threadId: split,
      messages: [grounded(300_000)],
    });
  }
  const whole = await copyAsWholeRows(t, split);
  const opts = { numItems: 5, maximumBytesRead: 500_000 };
  const pages = await budgetedPages(t, split, opts);
  expect(pages.length).toBeGreaterThan(2);
  expect(pages.flatMap((p) => p.positions)).toEqual(
    [0, 1, 2, 3, 4].map((i) => [0, i]),
  );
  expect(pages).toEqual(await budgetedPages(t, whole, opts));
  const bounded = await boundedPage(t, split, opts);
  expect(bounded.pageStatus).toBe("SplitRequired");
  expect(bounded).toEqual(await boundedPage(t, whole, opts));
});

test("a byte budget counts payloads the scan read ahead, as rows saved whole", async () => {
  const t = initConvexTest();
  const split = await createThread(t);
  const { messages } = await t.mutation(api.messages.addMessages, {
    threadId: split,
    messages: [grounded(600_000), grounded(600_000), grounded(600_000)],
  });
  await t.mutation(api.messages.updateMessage, {
    messageId: messages[2]._id as Id<"messages">,
    patch: { status: "failed", error: "nope" },
  });
  const whole = await copyAsWholeRows(t, split);
  const opts = { numItems: 3, maximumBytesRead: 1_000_000 };
  const pages = await budgetedPages(t, split, opts);
  expect(pages[0].positions).toEqual([[0, 0]]);
  expect(pages).toEqual(await budgetedPages(t, whole, opts));
  expect(await boundedPage(t, split, opts)).toEqual(
    await boundedPage(t, whole, opts),
  );
});

test("a byte budget counts the rows a page scanned and the payloads it joins", async () => {
  const t = initConvexTest();
  const threadId = await createThread(t);
  for (let i = 0; i < 10; i++) {
    await t.mutation(api.messages.addMessages, {
      threadId,
      messages: [
        {
          ...heavyAssistant,
          providerMetadata: { google: { grounding: "g".repeat(200_000) } },
        },
      ],
    });
  }
  await t.run(async (ctx) => {
    for (let i = 0; i < 4; i++) {
      await ctx.db.insert("messages", {
        ...heavyAssistant,
        providerMetadata: { google: { grounding: "g".repeat(300_000) } },
        threadId,
        order: 1,
        stepOrder: i,
        status: "success",
        tool: false,
        text: "the answer",
      });
    }
  });
  const budget = 1_000_000;
  const seen = [];
  let cursor: string | null = null;
  for (let pages = 0; pages < 20; pages++) {
    const { result, bytesRead } = await measure(t, (ctx) =>
      ctx.runQuery(api.messages.listMessagesByThreadId, {
        threadId,
        order: "asc",
        paginationOpts: { cursor, numItems: 20, maximumBytesRead: budget },
      }),
    );
    expect(bytesRead).toBeLessThan(budget + 2 * 310_000);
    seen.push(...result.page.map((m) => [m.order, m.stepOrder]));
    if (result.isDone) break;
    cursor = result.continueCursor;
  }
  expect(seen).toEqual([
    ...Array.from({ length: 10 }, (_, i) => [0, i]),
    ...Array.from({ length: 4 }, (_, i) => [1, i]),
  ]);
});

test("a byte budget counts rows the scan read ahead", async () => {
  const t = initConvexTest();
  const threadId = await createThread(t);
  for (let i = 0; i < 2; i++) {
    await t.mutation(api.messages.addMessages, {
      threadId,
      messages: [
        {
          ...heavyAssistant,
          providerMetadata: { google: { grounding: "g".repeat(600_000) } },
        },
      ],
    });
  }
  await t.run((ctx) =>
    ctx.db.insert("messages", {
      ...heavyAssistant,
      providerMetadata: { google: { grounding: "g".repeat(900_000) } },
      threadId,
      order: 1,
      stepOrder: 0,
      status: "failed",
      tool: false,
      text: "the answer",
    }),
  );
  const budget = 1_000_000;
  const { result, bytesRead } = await measure(t, (ctx) =>
    ctx.runQuery(api.messages.listMessagesByThreadId, {
      threadId,
      order: "asc",
      paginationOpts: { cursor: null, numItems: 2, maximumBytesRead: budget },
    }),
  );
  expect(result.page.map((m) => m.stepOrder)).toEqual([0]);
  expect(bytesRead).toBeLessThan(budget + 650_000);
});

test("a byte budget counts only the tiers a view reads", async () => {
  const t = initConvexTest();
  const threadId = await createThread(t);
  const thinking = (hidden: boolean) =>
    t.mutation(api.messages.addMessages, {
      threadId,
      hideFromUserIdSearch: hidden,
      messages: [0, 1, 2, 3, 4].map(() => ({
        message: {
          role: "assistant" as const,
          content: [
            {
              type: "reasoning" as const,
              text: "r".repeat(400_000),
            },
            { type: "text" as const, text: "the answer" },
          ],
        },
        providerMetadata: { google: { grounding: "g".repeat(400_000) } },
      })),
    });
  await thinking(false);
  const page = async (view: "text" | "model" | "full", order: "asc" | "desc") =>
    (
      await t.query(api.messages.listMessagesByThreadId, {
        threadId,
        order,
        view,
        paginationOpts: {
          cursor: null,
          numItems: 5,
          maximumBytesRead: 1_000_000,
        },
      })
    ).page.length;
  expect(await page("full", "asc")).toBe(2);
  expect(await page("model", "asc")).toBe(3);
  expect(await page("text", "asc")).toBe(5);
  // Text hidden from search isn't on the row, so a text view reads the body.
  await thinking(true);
  expect(await page("text", "desc")).toBe(3);
});

test("a large full page stays within the index range limit", async () => {
  const t = convexTest({ schema, modules, transactionLimits: true });
  const threadId = await createThread(t);
  const count = 2_100;
  for (let i = 0; i < count; i += 100) {
    await t.mutation(api.messages.addMessages, {
      threadId,
      messages: Array.from({ length: 100 }, () => ({
        message: { role: "assistant" as const, content: "hi" },
        usage: { promptTokens: 1, completionTokens: 1 },
      })),
    });
  }
  let read = 0;
  let cursor: string | null = null;
  for (let pages = 0; pages < 10; pages++) {
    const result: Awaited<ReturnType<typeof list>> = await t.query(
      api.messages.listMessagesByThreadId,
      {
        threadId,
        order: "asc",
        paginationOpts: { cursor, numItems: count },
      },
    );
    read += result.page.length;
    if (result.isDone) break;
    cursor = result.continueCursor;
  }
  expect(read).toBe(count);
});

test("a large clone batch stays within the index range limit", async () => {
  const t = convexTest({ schema, modules, transactionLimits: true });
  const threadId = await createThread(t);
  const count = 1_500;
  const vector = Array.from({ length: 128 }, (_, i) => (i === 0 ? 1 : 0));
  for (let i = 0; i < count; i += 100) {
    await t.mutation(api.messages.addMessages, {
      threadId,
      messages: Array.from({ length: 100 }, () => ({
        message: { role: "assistant" as const, content: "hi" },
        usage: { promptTokens: 1, completionTokens: 1 },
      })),
      embeddings: {
        model: "m",
        dimension: 128,
        vectors: Array.from({ length: 100 }, () => vector),
      },
    });
  }
  const targetThreadId = await createThread(t);
  const copied = await t.action(api.messages.cloneThread, {
    sourceThreadId: threadId,
    targetThreadId,
    batchSize: count,
  });
  expect(copied).toBe(count);
});

test("a clone batch of messages with files stays within the index range limit", async () => {
  const t = convexTest({ schema, modules, transactionLimits: true });
  const threadId = await createThread(t);
  const count = 1_024;
  const vector = Array.from({ length: 128 }, (_, i) => (i === 0 ? 1 : 0));
  for (let i = 0; i < count; i += 100) {
    const length = Math.min(100, count - i);
    const fileIds = await t.run((ctx) =>
      Promise.all(
        Array.from({ length }, (_, j) =>
          ctx.db.insert("files", {
            storageId: `s${i + j}`,
            hash: `h${i + j}`,
            refcount: 0,
            lastTouchedAt: 0,
          }),
        ),
      ),
    );
    await t.mutation(api.messages.addMessages, {
      threadId,
      messages: fileIds.map((fileId) => ({
        message: { role: "assistant" as const, content: "hi" },
        usage: { promptTokens: 1, completionTokens: 1 },
        fileIds: [fileId],
      })),
      embeddings: {
        model: "m",
        dimension: 128,
        vectors: Array.from({ length }, () => vector),
      },
    });
  }
  const targetThreadId = await createThread(t);
  const copied = await t.action(api.messages.cloneThread, {
    sourceThreadId: threadId,
    targetThreadId,
    batchSize: count,
  });
  expect(copied).toBe(count);
  const files = await t.run((ctx) => ctx.db.query("files").collect());
  expect(files.every((file) => file.refcount === 2)).toBe(true);
});

test("a clone batch of heavy rows saved whole, with many files, reads each row once", async () => {
  const t = convexTest({ schema, modules, transactionLimits: true });
  const threadId = await createThread(t);
  const count = 200;
  for (let i = 0; i < count; i += 25) {
    await t.run(async (ctx) => {
      for (let j = i; j < i + 25; j++) {
        const fileIds = await Promise.all(
          Array.from({ length: 20 }, (_, k) =>
            ctx.db.insert("files", {
              storageId: `s${j}-${k}`,
              hash: `h${j}-${k}`,
              refcount: 1,
              lastTouchedAt: 0,
            }),
          ),
        );
        await ctx.db.insert("messages", {
          ...grounded(50_000),
          fileIds,
          threadId,
          order: j,
          stepOrder: 0,
          status: "success",
          tool: false,
          text: "the answer",
        });
      }
    });
  }
  const targetThreadId = await createThread(t);
  const copied = await t.action(api.messages.cloneThread, {
    sourceThreadId: threadId,
    targetThreadId,
    batchSize: count,
  });
  expect(copied).toBe(count);
  const files = await t.run((ctx) => ctx.db.query("files").collect());
  expect(files.every((file) => file.refcount === 2)).toBe(true);
});

test("a clone batch of messages with heavy metadata stays within the read limit", async () => {
  const t = convexTest({ schema, modules, transactionLimits: true });
  const threadId = await createThread(t);
  const count = 100;
  for (let i = 0; i < count; i += 10) {
    await t.mutation(api.messages.addMessages, {
      threadId,
      messages: Array.from({ length: 10 }, () => grounded(200_000)),
    });
  }
  const targetThreadId = await createThread(t);
  const copied = await t.action(api.messages.cloneThread, {
    sourceThreadId: threadId,
    targetThreadId,
    batchSize: count,
  });
  expect(copied).toBe(count);
});

test("a read of many full messages keeps under 1,000 operations at once", async () => {
  const t = initConvexTest();
  const threadId = await createThread(t);
  for (let i = 0; i < 1_024; i += 128) {
    await t.mutation(api.messages.addMessages, {
      threadId,
      messages: Array.from({ length: 128 }, () => ({
        message: { role: "assistant" as const, content: "hi" },
        usage: { promptTokens: 1, completionTokens: 1 },
      })),
    });
  }
  const { readMessages } = await import("./messagePayloads.js");
  const { inFlight, read } = await t.run(async (ctx) => {
    const rows = await ctx.db.query("messages").collect();
    let current = 0;
    let most = 0;
    const db = new Proxy(ctx.db, {
      get(target, key) {
        const value = Reflect.get(target, key);
        if (key !== "get") return value;
        return async (...args: unknown[]) => {
          most = Math.max(most, ++current);
          try {
            return await (value as (...a: unknown[]) => unknown).apply(
              target,
              args,
            );
          } finally {
            current--;
          }
        };
      },
    });
    const read = await readMessages({ ...ctx, db }, rows);
    return { inFlight: most, read: read.length };
  });
  expect(read).toBe(1_024);
  expect(inFlight).toBeLessThan(1_000);
});

test("a clone batch of messages with heavy files stays within the read limit", async () => {
  const t = convexTest({ schema, modules, transactionLimits: true });
  const threadId = await createThread(t);
  const count = 20;
  for (let i = 0; i < count; i += 4) {
    const fileIds = await t.run((ctx) =>
      Promise.all(
        Array.from({ length: 4 }, (_, j) =>
          ctx.db.insert("files", {
            storageId: `s${i + j}`,
            hash: `h${i + j}`,
            filename: "f".repeat(900_000),
            refcount: 0,
            lastTouchedAt: 0,
          }),
        ),
      ),
    );
    await t.mutation(api.messages.addMessages, {
      threadId,
      messages: fileIds.map((fileId) => ({
        message: { role: "user" as const, content: "see file" },
        fileIds: [fileId],
      })),
    });
  }
  const targetThreadId = await createThread(t);
  const copied = await t.action(api.messages.cloneThread, {
    sourceThreadId: threadId,
    targetThreadId,
    batchSize: count,
  });
  expect(copied).toBe(count);
});

test("a clone reads each file once to count its refs", async () => {
  const t = convexTest({ schema, modules, transactionLimits: true });
  const threadId = await createThread(t);
  const fileIds = await t.run((ctx) =>
    Promise.all(
      Array.from({ length: 2_500 }, (_, i) =>
        ctx.db.insert("files", {
          storageId: `s${i}`,
          hash: `h${i}`,
          refcount: 0,
          lastTouchedAt: 0,
        }),
      ),
    ),
  );
  for (const message of [
    { role: "user" as const, content: "see files" },
    { role: "assistant" as const, content: "seen" },
  ]) {
    await t.mutation(api.messages.addMessages, {
      threadId,
      messages: [{ message, fileIds }],
    });
  }
  const targetThreadId = await createThread(t);
  const copied = await t.action(api.messages.cloneThread, {
    sourceThreadId: threadId,
    targetThreadId,
    batchSize: 2,
  });
  expect(copied).toBe(2);
  const files = await t.run((ctx) => ctx.db.query("files").collect());
  expect(files.every((file) => file.refcount === 4)).toBe(true);
});

async function insertFiles(t: T, count: number, prefix: string) {
  return t.run((ctx) =>
    Promise.all(
      Array.from({ length: count }, (_, i) =>
        ctx.db.insert("files", {
          storageId: `${prefix}${i}`,
          hash: `${prefix}${i}`,
          refcount: 0,
          lastTouchedAt: 0,
        }),
      ),
    ),
  );
}

test("a clone batch stops before the message that would pass its read budget", async () => {
  const t = convexTest({ schema, modules, transactionLimits: true });
  const threadId = await createThread(t);
  // Clones copy the newest first, so the heaviest comes last.
  for (const [prefix, files] of [
    ["a", 2_500],
    ["b", 1_000],
    ["c", 1_000],
  ] as const) {
    await t.mutation(api.messages.addMessages, {
      threadId,
      messages: [
        {
          message: { role: "user", content: prefix },
          fileIds: await insertFiles(t, files, prefix),
        },
      ],
    });
  }
  const targetThreadId = await createThread(t);
  const copied = await t.action(api.messages.cloneThread, {
    sourceThreadId: threadId,
    targetThreadId,
    batchSize: 3,
  });
  expect(copied).toBe(3);
});

test("a clone of a tool call and its result sharing files reads them once", async () => {
  const t = convexTest({ schema, modules, transactionLimits: true });
  const threadId = await createThread(t);
  const fileIds = await insertFiles(t, 2_500, "f");
  for (const step of [heavyToolCall, heavyToolResult]) {
    await t.mutation(api.messages.addMessages, {
      threadId,
      messages: [{ ...step, fileIds }],
    });
  }
  const targetThreadId = await createThread(t);
  const copied = await t.action(api.messages.cloneThread, {
    sourceThreadId: threadId,
    targetThreadId,
    batchSize: 2,
  });
  expect(copied).toBe(2);
  const files = await t.run((ctx) => ctx.db.query("files").collect());
  expect(files.every((file) => file.refcount === 4)).toBe(true);
});

test("a clone counts a file listed twice in a message as one ref", async () => {
  const t = initConvexTest();
  const threadId = await createThread(t);
  const [fileId] = await insertFiles(t, 1, "f");
  await t.mutation(api.messages.addMessages, {
    threadId,
    messages: [{ ...user, fileIds: [fileId, fileId] }],
  });
  const targetThreadId = await createThread(t);
  await t.action(api.messages.cloneThread, {
    sourceThreadId: threadId,
    targetThreadId,
  });
  await t.action(api.threads.deleteAllForThreadIdSync, {
    threadId: targetThreadId,
  });
  const file = await t.run((ctx) => ctx.db.get("files", fileId));
  expect(file?.refcount).toBe(1);
});

const READ_LIMIT = /at most 1024 messages/;

test("a read by ids joins at most 1,024 messages, and says so", async () => {
  const t = convexTest({ schema, modules, transactionLimits: true });
  const threadId = await createThread(t);
  const { messages } = await t.mutation(api.messages.addMessages, {
    threadId,
    messages: [heavyAssistant],
  });
  const id = messages[0]._id as Id<"messages">;
  const read = await t.query(api.messages.getMessagesByIds, {
    messageIds: Array.from({ length: 1_024 }, () => id),
  });
  expect(read).toHaveLength(1_024);
  await expect(
    t.query(api.messages.getMessagesByIds, {
      messageIds: Array.from({ length: 1_025 }, () => id),
    }),
  ).rejects.toThrow(READ_LIMIT);
});

test("a search joins at most 1,024 messages, and says so", async () => {
  const t = convexTest({ schema, modules, transactionLimits: true });
  const threadId = await createThread(t, "u");
  const vector = Array.from({ length: 128 }, (_, i) => (i === 0 ? 1 : 0));
  for (let i = 0; i < 1_100; i += 100) {
    await t.mutation(api.messages.addMessages, {
      threadId,
      messages: Array.from({ length: 100 }, () => user),
      embeddings: {
        model: "m",
        dimension: 128,
        vectors: Array.from({ length: 100 }, () => vector),
      },
    });
  }
  const search = (around: number) =>
    t.action(api.messages.searchMessages, {
      threadId,
      embedding: vector,
      embeddingModel: "m",
      vectorSearch: true,
      limit: 1,
      messageRange: { before: around, after: around },
    });
  expect((await search(500)).length).toBeLessThanOrEqual(1_001);
  await expect(search(1_100)).rejects.toThrow(READ_LIMIT);
});

const WORDS = ["convex", "agent", "stream", "tool", "query", "schema", "delta"];
const words = (seed: number, count: number) =>
  Array.from(
    { length: count },
    (_, i) => WORDS[(seed * 7 + i * 13) % WORDS.length],
  ).join(" ");

// The shapes of the cloud bench fixture's threads, as the AI SDK providers
// save them.
const fixtureThreads: Record<string, Message[]> = {
  "plain chat": [
    { role: "user", content: words(1, 25) },
    {
      role: "assistant",
      content: [
        {
          type: "text",
          text: words(2, 100),
          providerOptions: { openai: { itemId: "msg_1" } },
        },
      ],
    },
  ],
  "encrypted reasoning": [
    { role: "user", content: words(3, 25) },
    {
      role: "assistant",
      content: [
        {
          type: "reasoning",
          text: words(4, 60),
          providerOptions: {
            openai: {
              itemId: "rs_1",
              reasoningEncryptedContent: "e".repeat(500),
            },
          },
        },
        {
          type: "text",
          text: words(5, 100),
          providerOptions: { openai: { itemId: "msg_2" } },
        },
      ],
    },
  ],
  "Claude thinking": [
    { role: "user", content: words(6, 25) },
    {
      role: "assistant",
      content: [
        {
          type: "reasoning",
          text: words(7, 150),
          providerOptions: { anthropic: { signature: "s".repeat(300) } },
        },
        { type: "text", text: words(8, 100) },
      ],
    },
  ],
  "Claude tools": [
    { role: "user", content: words(9, 25) },
    {
      role: "assistant",
      content: [
        {
          type: "reasoning",
          text: words(10, 40),
          providerOptions: { anthropic: { signature: "s".repeat(300) } },
        },
        {
          type: "tool-call",
          toolCallId: "toolu_1",
          toolName: "search",
          input: { query: words(11, 6) },
          args: { query: words(11, 6) },
        },
      ],
    },
    {
      role: "tool",
      content: [
        {
          type: "tool-result",
          toolCallId: "toolu_1",
          toolName: "search",
          output: { type: "json", value: { results: [words(12, 50)] } },
        },
      ],
    },
    {
      role: "assistant",
      content: [
        {
          type: "reasoning",
          text: words(13, 40),
          providerOptions: { anthropic: { signature: "s".repeat(300) } },
        },
        { type: "text", text: words(14, 40) },
      ],
    },
  ],
  "Gemini grounding": [
    { role: "user", content: words(15, 25) },
    {
      role: "assistant",
      content: [
        {
          type: "reasoning",
          text: words(16, 60),
          providerOptions: { google: { thoughtSignature: "t".repeat(200) } },
        },
        {
          type: "text",
          text: words(17, 100),
          providerOptions: { google: { thoughtSignature: "t".repeat(200) } },
        },
      ],
    },
  ],
  "image + PDF": [
    {
      role: "user",
      content: [
        { type: "text", text: "What is in these?" },
        {
          type: "image",
          image: new Uint8Array([137, 80, 78, 71]).buffer,
          mediaType: "image/png",
        },
        {
          type: "file",
          data: "https://example.convex.cloud/api/storage/a",
          filename: "a.pdf",
          mediaType: "application/pdf",
        },
      ],
    },
    {
      role: "assistant",
      content: [
        {
          type: "text",
          text: words(18, 100),
          providerOptions: { openai: { itemId: "msg_3" } },
        },
      ],
    },
  ],
};

// Messages whose text is easy to split wrongly: several text parts, empty
// ones, spaces at either end or doubled, text between other parts, and text
// parts with provider options.
function generatedMessages(seed: number, count: number): Message[] {
  let state = seed;
  const next = () => {
    state = (state * 1_103_515_245 + 12_345) % 2 ** 31;
    return state / 2 ** 31;
  };
  const choose = <V>(values: readonly V[]) =>
    values[Math.floor(next() * values.length)];
  const PIECES = ["", " ", "  ", "a", "one two", " lead", "trail ", "a  b"];
  const MORE = ["line\nbreak", "é ü 🧪", "x".repeat(300)];
  const text = () =>
    Array.from({ length: Math.floor(next() * 4) }, () =>
      choose([...PIECES, ...MORE]),
    ).join(choose(["", " "]));
  const options = () =>
    next() < 0.4 ? { providerOptions: { p: { id: text() } } } : {};
  const textPart = () => ({ type: "text", text: text(), ...options() });
  const userParts = [
    textPart,
    textPart,
    () => ({
      type: "image",
      image: new Uint8Array([1, 2, 3]).buffer,
      mediaType: "image/png",
    }),
    () => ({ type: "file", data: text(), mediaType: "text/plain" }),
  ];
  const assistantParts = [
    textPart,
    textPart,
    () => ({ type: "reasoning", text: text(), ...options() }),
    () => ({
      type: "tool-call",
      toolCallId: "c",
      toolName: "t",
      input: { q: text() },
    }),
    () => ({ type: "file", data: text(), mediaType: "text/plain" }),
  ];
  const parts = (kinds: (() => object)[]) =>
    Array.from({ length: Math.floor(next() * 6) }, () => choose(kinds)());
  return Array.from({ length: count }, () => {
    const extra = next() < 0.2 ? { providerOptions: { p: { m: 1 } } } : {};
    const kind = choose([
      "user",
      "user parts",
      "assistant",
      "assistant parts",
      "system",
      "tool",
    ]);
    const message =
      kind === "user"
        ? { role: "user", content: text() }
        : kind === "user parts"
          ? { role: "user", content: parts(userParts) }
          : kind === "assistant"
            ? { role: "assistant", content: text() }
            : kind === "assistant parts"
              ? { role: "assistant", content: parts(assistantParts) }
              : kind === "system"
                ? { role: "system", content: text() }
                : {
                    role: "tool",
                    content: [
                      {
                        type: "tool-result",
                        toolCallId: "c",
                        toolName: "t",
                        output: { type: "text", value: text() },
                      },
                    ],
                  };
    return { ...message, ...extra } as Message;
  });
}

const exact = (value: unknown) => convexToJson(value as Value);

// The messages read back from rows holding them whole: on the row, as saved
// before messagePayloads, and apart, as a message without row text is saved.
async function readAsStoredWhole(t: T, docs: MessageDoc[]) {
  const threadId = await createThread(t);
  const ids = await t.run((ctx) =>
    Promise.all(
      docs.flatMap((doc) => {
        const row = {
          ...omit(doc, ["_id", "_creationTime", "threadId", "embeddingId"]),
          threadId,
          fileIds: doc.fileIds as Id<"files">[] | undefined,
        } as Omit<Doc<"messages">, "_id" | "_creationTime">;
        return [
          ctx.db.insert("messages", row),
          ctx.db
            .insert("messagePayloads", { message: doc.message! })
            .then((payloadId) =>
              ctx.db.insert("messages", {
                ...row,
                message: undefined,
                payloadId,
                payloadSize: getDocumentSize({ message: doc.message! }),
                role: doc.message!.role,
              }),
            ),
        ];
      }),
    ),
  );
  const read = await t.query(api.messages.getMessagesByIds, {
    messageIds: ids,
  });
  return read.map((doc) => doc!.message);
}

describe("each message's text is stored once", () => {
  test("a payload leaves out the text its row holds", async () => {
    const t = initConvexTest();
    for (const messages of Object.values(fixtureThreads)) {
      await t.mutation(api.messages.addMessages, {
        threadId: await createThread(t),
        messages: messages.map((message) => ({ message })),
      });
    }
    const texts = await t.run(async (ctx) => {
      const rows = await ctx.db.query("messages").collect();
      return Promise.all(
        rows.flatMap((row) =>
          row.text
            ? [
                ctx.db
                  .get("messagePayloads", row.payloadId!)
                  .then((payload) => extractText(payload!.message) ?? ""),
              ]
            : [],
        ),
      );
    });
    expect(texts.length).toBeGreaterThan(10);
    expect(texts.filter(Boolean)).toEqual([]);
  });

  test("a model read reads each message's text once", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    const text = "word ".repeat(2_000);
    await t.mutation(api.messages.addMessages, {
      threadId,
      messages: Array.from({ length: 5 }, () => [
        { message: { role: "user" as const, content: text } },
        {
          message: {
            role: "assistant" as const,
            content: [{ type: "text" as const, text }],
          },
        },
      ]).flat(),
    });
    const { bytesRead, result } = await measure(t, (ctx) =>
      ctx.runQuery(api.messages.listMessagesByThreadId, {
        threadId,
        order: "desc",
        view: "model",
        paginationOpts: { cursor: null, numItems: 10 },
      }),
    );
    expect(result.page.map((m) => m.text)).toEqual(Array(10).fill(text));
    expect(bytesRead).toBeLessThan(10 * text.length * 1.2);
  });

  test("a row whose text was changed apart from its message fails to read, not misreads", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    const { messages } = await t.mutation(api.messages.addMessages, {
      threadId,
      messages: [user, heavyAssistant],
    });
    for (const doc of messages) {
      await t.run((ctx) =>
        ctx.db.patch("messages", doc._id as Id<"messages">, {
          text: "edited",
        }),
      );
      await expect(
        t.query(api.messages.getMessagesByIds, {
          messageIds: [doc._id as Id<"messages">],
        }),
      ).rejects.toThrow(/text no longer matches/);
    }
  });

  test("a row whose text an older version extracted otherwise reads and copies as its message", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    const message = {
      role: "assistant" as const,
      content: [
        { type: "text" as const, text: "one" },
        { type: "reasoning" as const, text: "thinking" },
        { type: "text" as const, text: "two" },
      ],
    };
    const id = await t.run((ctx) =>
      ctx.db.insert("messages", {
        message,
        threadId,
        order: 0,
        stepOrder: 0,
        status: "success",
        tool: false,
        text: "onetwo",
      }),
    );
    const textView = async (threadId: Id<"threads">) =>
      (await list(t, threadId, { view: "text" })).page[0].message;
    expect(await textView(threadId)).toEqual({
      role: "assistant",
      content: "one two",
    });

    const targetThreadId = await createThread(t);
    await t.action(api.messages.cloneThread, {
      sourceThreadId: threadId,
      targetThreadId,
    });
    const [copy] = (await list(t, targetThreadId)).page;
    expect(copy).toMatchObject({ message, text: "one two" });
    expect(await textView(targetThreadId)).toEqual({
      role: "assistant",
      content: "one two",
    });

    const updated = await t.mutation(api.messages.updateMessage, {
      messageId: id,
      patch: { providerOptions: { p: { updated: true } } },
    });
    expect(updated).toMatchObject({ message, text: "one two" });
    expect(await textView(threadId)).toEqual({
      role: "assistant",
      content: "one two",
    });
  });

  test("every message reads back exactly, on every path", async () => {
    const t = initConvexTest();
    const sets = [
      ...Object.values(fixtureThreads),
      generatedMessages(1, 100),
      generatedMessages(2, 100),
      generatedMessages(3, 100),
    ];
    for (const sent of sets) {
      for (const hideFromUserIdSearch of [false, true]) {
        const threadId = await createThread(t, "u");
        const { messages: saved } = await t.mutation(api.messages.addMessages, {
          threadId,
          hideFromUserIdSearch,
          messages: sent.map((message) => ({ message })),
        });
        const ids = saved.map((m) => m._id as Id<"messages">);
        const expectExact = (docs: (MessageDoc | null)[], expected = sent) =>
          expect(docs.map((m) => exact(m?.message))).toStrictEqual(
            expected.map(exact),
          );
        expectExact(saved);
        expect(saved.map((m) => m.text)).toEqual(
          sent.map((m) => (hideFromUserIdSearch ? undefined : extractText(m))),
        );
        expectExact((await list(t, threadId)).page);
        expectExact((await list(t, threadId, { view: "model" })).page);
        expectExact(
          await t.query(api.messages.getMessagesByIds, { messageIds: ids }),
        );
        const text = await list(t, threadId, { view: "text" });
        expect(text.page.map((m) => m.message)).toEqual(
          sent
            .filter((_, i) => !saved[i].tool)
            .map((m) => ({ role: m.role, content: extractText(m) ?? "" })),
        );

        const targetThreadId = await createThread(t);
        await t.action(api.messages.cloneThread, {
          sourceThreadId: threadId,
          targetThreadId,
        });
        expectExact((await list(t, targetThreadId)).page);

        expect((await readAsStoredWhole(t, saved)).map(exact)).toStrictEqual(
          saved.flatMap((doc) => [exact(doc.message), exact(doc.message)]),
        );

        await t.run(async (ctx) => {
          for (const messageId of ids) {
            await ctx.runMutation(api.messages.updateMessage, {
              messageId,
              patch: { providerOptions: { p: { updated: true } } },
            });
          }
        });
        expectExact((await list(t, threadId)).page);
        const replaced = [...sent.slice(1), sent[0]];
        await t.run(async (ctx) => {
          for (const [i, messageId] of ids.entries()) {
            await ctx.runMutation(api.messages.updateMessage, {
              messageId,
              patch: { message: replaced[i] },
            });
          }
        });
        expectExact((await list(t, threadId)).page, replaced);
        expectExact(
          (await list(t, threadId, { view: "model" })).page,
          replaced,
        );
      }
    }
    await expectConsistentStorage(t);
  }, 60_000);
});
