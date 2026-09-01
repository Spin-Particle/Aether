import { describe, expect, test } from "bun:test"
import { Instance } from "../../src/project/instance"
import { ModelID, ProviderID } from "../../src/provider/schema"
import type { Provider } from "../../src/provider/provider"
import { Session } from "../../src/session"
import { MessageV2 } from "../../src/session/message-v2"
import { SessionPrompt } from "../../src/session/prompt"
import { MessageID, PartID, type SessionID } from "../../src/session/schema"
import { Log } from "../../src/util/log"
import { tmpdir } from "../fixture/fixture"

Log.init({ print: false })

const model: Provider.Model = {
  id: ModelID.make("test-model"),
  providerID: ProviderID.make("test"),
  api: {
    id: "test-model",
    url: "https://example.com",
    npm: "@ai-sdk/openai",
  },
  name: "Test Model",
  capabilities: {
    temperature: true,
    reasoning: false,
    attachment: false,
    toolcall: true,
    input: { text: true, audio: false, image: false, video: false, pdf: false },
    output: { text: true, audio: false, image: false, video: false, pdf: false },
    interleaved: false,
  },
  cost: { input: 0, output: 0, cache: { read: 0, write: 0 } },
  limit: { context: 0, input: 0, output: 0 },
  status: "active",
  options: {},
  headers: {},
  release_date: "2026-01-01",
}

function assistant(sessionID: SessionID, parentID: string): MessageV2.Assistant {
  return {
    id: MessageID.ascending(),
    sessionID,
    role: "assistant",
    time: { created: Date.now(), completed: Date.now() },
    parentID: MessageID.make(parentID),
    modelID: model.id,
    providerID: model.providerID,
    mode: "",
    agent: "build",
    path: { cwd: "/", root: "/" },
    cost: 0,
    tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
  }
}

type Draft<T> = T extends unknown ? Omit<T, "id" | "sessionID" | "messageID"> : never
type DraftPart = Draft<MessageV2.Part>

async function reply(sessionID: SessionID, parentID: string, parts: DraftPart[]) {
  const info = assistant(sessionID, parentID)
  await Session.updateMessage(info)
  for (const part of parts) {
    await Session.updatePart({
      ...part,
      id: PartID.ascending(),
      sessionID,
      messageID: info.id,
    } as MessageV2.Part)
  }
  return info
}

describe("session.prompt session reference", () => {
  test("injects referenced session user/assistant text as a session-ref part", async () => {
    await using tmp = await tmpdir({
      git: true,
      config: { agent: { build: { model: "openai/gpt-5.2" } } },
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const b = await Session.create({})
        const ask = await SessionPrompt.prompt({
          sessionID: b.id,
          agent: "build",
          noReply: true,
          parts: [{ type: "text", text: "question one" }],
        })
        await reply(b.id, ask.info.id, [
          { type: "reasoning", text: "secret thinking", time: { start: 0 } },
          { type: "text", text: "answer one" },
        ])

        const a = await Session.create({})
        const msg = await SessionPrompt.prompt({
          sessionID: a.id,
          agent: "build",
          noReply: true,
          parts: [{ type: "text", text: `@${b.id} please continue` }],
        })

        const ref = msg.parts.find((part) => part.type === "session-ref")
        if (!ref || ref.type !== "session-ref") throw new Error("expected session-ref part")
        expect(ref.refID).toBe(b.id)
        expect(ref.total).toBe(2)
        expect(ref.shown).toBe(2)
        expect(ref.text).toContain(`<referenced_session id="${b.id}"`)
        expect(ref.text).toContain("[User]: question one")
        expect(ref.text).toContain("[Assistant]: answer one")
        expect(ref.text).not.toContain("secret thinking")

        await Session.remove(a.id)
        await Session.remove(b.id)
      },
    })
  })

  test("renders session-ref part into model messages", async () => {
    await using tmp = await tmpdir({
      git: true,
      config: { agent: { build: { model: "openai/gpt-5.2" } } },
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const b = await Session.create({})
        await SessionPrompt.prompt({
          sessionID: b.id,
          agent: "build",
          noReply: true,
          parts: [{ type: "text", text: "hello from b" }],
        })

        const a = await Session.create({})
        const msg = await SessionPrompt.prompt({
          sessionID: a.id,
          agent: "build",
          noReply: true,
          parts: [{ type: "text", text: `@${b.id}` }],
        })

        const rendered = MessageV2.toModelMessages([{ info: msg.info, parts: msg.parts }], model)
        const text = JSON.stringify(rendered)
        expect(text).toContain("<referenced_session")
        expect(text).toContain("[User]: hello from b")

        await Session.remove(a.id)
        await Session.remove(b.id)
      },
    })
  })

  test("truncates long referenced sessions", async () => {
    await using tmp = await tmpdir({
      git: true,
      config: { agent: { build: { model: "openai/gpt-5.2" } } },
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const b = await Session.create({})
        const ask = await SessionPrompt.prompt({
          sessionID: b.id,
          agent: "build",
          noReply: true,
          parts: [{ type: "text", text: "x".repeat(50_000) }],
        })
        await reply(b.id, ask.info.id, [{ type: "text", text: "recent answer" }])

        const a = await Session.create({})
        const msg = await SessionPrompt.prompt({
          sessionID: a.id,
          agent: "build",
          noReply: true,
          parts: [{ type: "text", text: `@${b.id}` }],
        })

        const ref = msg.parts.find((part) => part.type === "session-ref")
        if (!ref || ref.type !== "session-ref") throw new Error("expected session-ref part")
        expect(ref.total).toBe(2)
        expect(ref.shown).toBe(1)
        expect(ref.text).toContain("[Assistant]: recent answer")
        expect(ref.text).toContain("most recent 1 of 2")
        expect(ref.text.length).toBeLessThan(45_000)

        await Session.remove(a.id)
        await Session.remove(b.id)
      },
    })
  })

  test("skips self reference with a note", async () => {
    await using tmp = await tmpdir({
      git: true,
      config: { agent: { build: { model: "openai/gpt-5.2" } } },
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const a = await Session.create({})
        const msg = await SessionPrompt.prompt({
          sessionID: a.id,
          agent: "build",
          noReply: true,
          parts: [{ type: "text", text: `@${a.id} self` }],
        })

        expect(msg.parts.some((part) => part.type === "session-ref")).toBe(false)
        const note = msg.parts.find((part) => part.type === "text" && part.synthetic)
        if (!note || note.type !== "text") throw new Error("expected synthetic note")
        expect(note.text).toContain("current session")

        await Session.remove(a.id)
      },
    })
  })

  test("skips unknown session with a note", async () => {
    await using tmp = await tmpdir({
      git: true,
      config: { agent: { build: { model: "openai/gpt-5.2" } } },
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const a = await Session.create({})
        const msg = await SessionPrompt.prompt({
          sessionID: a.id,
          agent: "build",
          noReply: true,
          parts: [{ type: "text", text: "@ses_000000000000zzzzzzzzzzzz missing" }],
        })

        expect(msg.parts.some((part) => part.type === "session-ref")).toBe(false)
        const note = msg.parts.find((part) => part.type === "text" && part.synthetic)
        if (!note || note.type !== "text") throw new Error("expected synthetic note")
        expect(note.text).toContain("could not be found")

        await Session.remove(a.id)
      },
    })
  })

  test("dedupes repeated references to the same session", async () => {
    await using tmp = await tmpdir({
      git: true,
      config: { agent: { build: { model: "openai/gpt-5.2" } } },
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const b = await Session.create({})
        await SessionPrompt.prompt({
          sessionID: b.id,
          agent: "build",
          noReply: true,
          parts: [{ type: "text", text: "dup target" }],
        })

        const a = await Session.create({})
        const msg = await SessionPrompt.prompt({
          sessionID: a.id,
          agent: "build",
          noReply: true,
          parts: [{ type: "text", text: `@${b.id} and again @${b.id}` }],
        })

        expect(msg.parts.filter((part) => part.type === "session-ref").length).toBe(1)

        await Session.remove(a.id)
        await Session.remove(b.id)
      },
    })
  })
})
