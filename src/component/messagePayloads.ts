import { omit, pick } from "convex-helpers";
import {
  QueryStream,
  type IndexBounds,
  type IndexKey,
} from "convex-helpers/server/stream";
import type { WithoutSystemFields } from "convex/server";
import { getDocumentSize, v, type Infer, type Value } from "convex/values";
import { extractText } from "../shared.js";
import type { Message } from "../validators.js";
import type { Doc, Id } from "./_generated/dataModel.js";
import type { MutationCtx, QueryCtx } from "./_generated/server.js";

type MessageFields = WithoutSystemFields<Doc<"messages">>;
type Tier = "messagePayloads" | "messageRecords";

const BODY_FIELDS = ["message"] as const satisfies (keyof MessageFields)[];
const RECORD_FIELDS = [
  "providerOptions",
  "usage",
  "providerMetadata",
  "sources",
  "warnings",
  "reasoning",
  "reasoningDetails",
] as const satisfies (keyof MessageFields)[];
const TIER_FIELDS = [...BODY_FIELDS, ...RECORD_FIELDS];
const STORAGE_FIELDS = [
  "payloadId",
  "payloadSize",
  "recordId",
  "recordSize",
  "role",
  "approvalIds",
] as const satisfies (keyof MessageFields)[];

export type Payload = Pick<MessageFields, (typeof TIER_FIELDS)[number]>;

/**
 * Each view holds the one before it: text is a message's role and text, model
 * adds what the model reads, and full adds provider metadata, usage, sources
 * and warnings.
 */
export const vMessageView = v.union(
  v.literal("text"),
  v.literal("model"),
  v.literal("full"),
);
export type MessageView = Infer<typeof vMessageView>;

// A row's text is unset when it's hidden from search, so its text view needs
// the body. Rows saved whole have no tiers to join.
function joins(row: Doc<"messages">, view: MessageView) {
  return {
    body: !!row.payloadId && (view !== "text" || row.text === undefined),
    record: !!row.recordId && view === "full",
  };
}

// Convex allows a transaction 4,096 `db.get` and `db.query` calls, and a
// message read costs up to three: its row, body and record. A quarter is left
// for the caller.
export const MAX_MESSAGES_PER_READ = 1_024;

export function checkReadSize(count: number) {
  if (count > MAX_MESSAGES_PER_READ) {
    throw new Error(
      `A transaction reads at most ${MAX_MESSAGES_PER_READ} messages, and this one asked for ${count}. Read fewer at once.`,
    );
  }
}

function joinedSize(row: Doc<"messages">, view: MessageView) {
  const { body, record } = joins(row, view);
  return (
    (body ? (row.payloadSize ?? 0) : 0) + (record ? (row.recordSize ?? 0) : 0)
  );
}

export function touchesPayload(patch: object) {
  return TIER_FIELDS.some((field) => field in patch);
}

async function storeTier<T extends Tier>(
  ctx: MutationCtx,
  table: T,
  fields: Record<string, unknown>,
  existingId: Id<T> | undefined,
): Promise<{ id?: Id<T>; size?: number }> {
  if (Object.values(fields).every((value) => value === undefined)) {
    if (existingId) {
      await ctx.db.delete(table, existingId);
    }
    return {};
  }
  const size = getDocumentSize(fields as Record<string, Value>);
  if (existingId) {
    await ctx.db.replace(table, existingId, fields as never);
    return { id: existingId, size };
  }
  return { id: await ctx.db.insert(table, fields as never), size };
}

/**
 * Writes the message's body and record rows, replacing the existing ones, and
 * returns the fields for the message row itself. A tier row exists exactly
 * when it has a field.
 */
export async function storeMessageFields(
  ctx: MutationCtx,
  fields: MessageFields,
  existing?: Pick<Doc<"messages">, "payloadId" | "recordId">,
): Promise<{ row: MessageFields; payload: Payload }> {
  // A row's text is its message's text, or unset to hide it from search, so
  // a text view reads it instead of the message. Rows an older extractText
  // wrote are brought in line when rewritten.
  const text =
    fields.message && fields.text !== undefined
      ? extractText(fields.message)
      : fields.text;
  const body = await storeTier(
    ctx,
    "messagePayloads",
    fields.message ? leaveOutText(fields.message, text) : {},
    existing?.payloadId,
  );
  const record = await storeTier(
    ctx,
    "messageRecords",
    pick(fields, RECORD_FIELDS),
    existing?.recordId,
  );
  return {
    row: {
      ...(omit(fields, TIER_FIELDS) as MessageFields),
      text,
      payloadId: body.id,
      payloadSize: body.size,
      recordId: record.id,
      recordSize: record.size,
      role: fields.message?.role,
      approvalIds: approvalIds(fields.message),
    },
    payload: pick(fields, TIER_FIELDS) as Payload,
  };
}

/**
 * The message without the text its row holds, and the length of each text
 * part's text, so that the row's text, cut at those lengths and the single
 * spaces `extractText` joins them with, gives each part's text back. A message
 * whose row holds no text is stored whole.
 */
function leaveOutText(message: Message, text: string | undefined) {
  if (!text) {
    return { message };
  }
  if (typeof message.content === "string") {
    return {
      message: { ...message, content: "" } as Message,
      textLengths: [text.length],
    };
  }
  const textLengths: number[] = [];
  const content = message.content.map((part) => {
    if (part.type !== "text") {
      return part;
    }
    textLengths.push(part.text.length);
    return { ...part, text: "" };
  });
  return { message: { ...message, content } as Message, textLengths };
}

function putBackText(
  message: Message,
  textLengths: number[] | undefined,
  row: Doc<"messages">,
): Message {
  if (!textLengths) {
    return message;
  }
  const text = row.text ?? "";
  const changed = new Error(
    `Message ${row._id}'s text no longer matches its stored message`,
  );
  if (typeof message.content === "string") {
    if (textLengths[0] !== text.length) {
      throw changed;
    }
    return { ...message, content: text } as Message;
  }
  let start = 0;
  let next = 0;
  const content = message.content.map((part) => {
    if (part.type !== "text") {
      return part;
    }
    const length = textLengths[next++];
    if (!length) {
      return part;
    }
    const partText = text.slice(start, start + length);
    start += length + 1;
    return { ...part, text: partText };
  });
  if (next !== textLengths.length || start !== text.length + 1) {
    throw changed;
  }
  return { ...message, content } as Message;
}

export async function deletePayload(
  ctx: MutationCtx,
  row: Pick<Doc<"messages">, "payloadId" | "recordId">,
) {
  if (row.payloadId) {
    await ctx.db.delete("messagePayloads", row.payloadId);
  }
  if (row.recordId) {
    await ctx.db.delete("messageRecords", row.recordId);
  }
}

function approvalIds(message: Message | undefined) {
  const ids =
    Array.isArray(message?.content) &&
    message.content.flatMap((part) =>
      part.type === "tool-approval-request" ||
      part.type === "tool-approval-response"
        ? [part.approvalId]
        : [],
    );
  return ids && ids.length > 0 ? ids : undefined;
}

/**
 * Whether the row can hold a request or response for one of the approvals
 * without reading its body: a row saved whole holds its message itself.
 */
export function mayHoldApproval(
  row: Doc<"messages">,
  approvalIds: Set<string>,
) {
  return !row.payloadId || !!row.approvalIds?.some((id) => approvalIds.has(id));
}

/**
 * The storage fields and text of a copy of the row, its tiers written as a
 * save writes them, so a copy of a row saved whole is split too. Undefined
 * drops the fields such a row holds itself.
 */
export async function copyPayload(ctx: MutationCtx, row: Doc<"messages">) {
  const { row: copy } = await storeMessageFields(
    ctx,
    omit(await readMessage(ctx, row), ["_id", "_creationTime"]),
  );
  return {
    ...Object.fromEntries(TIER_FIELDS.map((field) => [field, undefined])),
    ...pick(copy, [...STORAGE_FIELDS, "text"]),
  } as Partial<MessageFields>;
}

/** The message a write stored, from its row and the payload it split off. */
export function joinPayload(
  row: Doc<"messages">,
  payload: Payload,
): Doc<"messages"> {
  return { ...omit(row, STORAGE_FIELDS), ...payload };
}

async function readTier<T extends Tier>(
  ctx: QueryCtx,
  table: T,
  id: Id<T>,
  messageId: Id<"messages">,
) {
  const tier = await ctx.db.get(table, id);
  if (!tier) {
    throw new Error(`${table} ${id} of message ${messageId} not found`);
  }
  return omit(tier, ["_id", "_creationTime"]);
}

export async function readMessage(
  ctx: QueryCtx,
  row: Doc<"messages">,
  view: MessageView = "full",
): Promise<Doc<"messages">> {
  const { body, record } = joins(row, view);
  const [payload, records] = await Promise.all([
    body
      ? readTier(ctx, "messagePayloads", row.payloadId!, row._id).then(
          ({ message, textLengths }) => ({
            message: putBackText(message, textLengths, row),
          }),
        )
      : {},
    record ? readTier(ctx, "messageRecords", row.recordId!, row._id) : {},
  ]);
  const doc = { ...row, ...payload, ...records };
  return view === "text" ? textOnlyMessage(doc) : omit(doc, STORAGE_FIELDS);
}

export function readMessages(
  ctx: QueryCtx,
  rows: Doc<"messages">[],
  view: MessageView = "full",
) {
  checkReadSize(rows.length);
  return inBatches(rows, (row) => readMessage(ctx, row, view));
}

// Convex allows a function 1,000 concurrent operations, and each message here
// makes at most two at once.
const CONCURRENT_MESSAGES = 256;

export async function inBatches<T, R>(
  items: T[],
  fn: (item: T) => Promise<R>,
  size = CONCURRENT_MESSAGES,
): Promise<R[]> {
  const results: R[] = [];
  for (let i = 0; i < items.length; i += size) {
    results.push(...(await Promise.all(items.slice(i, i + size).map(fn))));
  }
  return results;
}

/**
 * The message as its role and text. The text is what a full read's
 * `extractText` gives, or "" without any.
 */
function textOnlyMessage(doc: Doc<"messages">): Doc<"messages"> {
  const role = doc.role ?? doc.message?.role;
  const text = doc.message ? extractText(doc.message) : doc.text;
  return {
    ...omit(doc, [...TIER_FIELDS, ...STORAGE_FIELDS]),
    ...(role && { message: { role, content: text ?? "" } as Message }),
  } as Doc<"messages">;
}

/**
 * The rows, each charging a byte budget for the tiers its view joins, so a
 * page of them ends where a page of rows saved whole would. Wrap each stream
 * before merging, so rows a merge reads ahead are charged too. Paginating
 * reads no tiers; the page's are read together after.
 */
export class ChargingPayloads extends QueryStream<Doc<"messages">> {
  #rows: QueryStream<Doc<"messages">>;
  #view: MessageView;
  constructor(rows: QueryStream<Doc<"messages">>, view: MessageView) {
    super();
    this.#rows = rows;
    this.#view = view;
  }
  iterWithKeys(trackBandwidth = false) {
    const rows = this.#rows.iterWithKeys(trackBandwidth);
    const view = this.#view;
    return {
      async *[Symbol.asyncIterator](): AsyncGenerator<
        [Doc<"messages"> | null, IndexKey, number],
        undefined
      > {
        for await (const [row, indexKey, bandwidth] of rows) {
          const joined = trackBandwidth && row ? joinedSize(row, view) : 0;
          yield [row, indexKey, bandwidth + joined];
        }
      },
    };
  }
  narrow(indexBounds: IndexBounds) {
    return new ChargingPayloads(this.#rows.narrow(indexBounds), this.#view);
  }
  getOrder() {
    return this.#rows.getOrder();
  }
  getIndexFields() {
    return this.#rows.getIndexFields();
  }
  getEqualityIndexFilter() {
    return this.#rows.getEqualityIndexFilter();
  }
}
