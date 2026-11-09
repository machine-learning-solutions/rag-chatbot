import express from "express";
import multer from "multer";

const upload = multer({ storage: multer.memoryStorage() });

const PORT = process.env.PORT || 3001;
const AVATAR_URL = process.env.AVATAR_URL || "http://localhost:8010";
// RAG backend (Chatbot/backend) — already has Jehad's resume ingested and
// indexed (pgvector + hybrid search + reranker), so the avatar answers from
// it instead of an unaugmented Ollama chat.
const RAG_BACKEND_URL = process.env.RAG_BACKEND_URL || "http://localhost:8000";

const app = express();
app.use(express.json());
// Permissive CORS: lets the portfolio's local dev server (a different port,
// e.g. localhost:3000) call this API directly, bypassing Next.js dev's own
// rewrite proxy, whose request timeout is too short for avatar generation
// (TTS + Wav2Lip + GFPGAN can take well past it). No cookies/auth involved,
// and production never hits this directly (it goes through the ngrok
// gateway), so a wildcard origin is fine here.
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") {
    res.sendStatus(204);
    return;
  }
  next();
});
app.use(express.static("public"));
app.use("/avatar", express.static("avatar"));

app.post("/api/chat", async (req, res) => {
  const { messages } = req.body;
  const lastUserMessage = [...messages].reverse().find((m) => m.role === "user");
  const question = lastUserMessage?.content?.trim();

  res.setHeader("Content-Type", "application/x-ndjson");

  if (!question) {
    res.end();
    return;
  }

  // Each turn is answered independently from the resume knowledge base — the
  // backend's retrieval only takes a single question, no conversation history.
  let answer;
  try {
    const ragRes = await fetch(`${RAG_BACKEND_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, portfolio_fast: true }),
    });

    if (!ragRes.ok) {
      throw new Error(`RAG backend returned ${ragRes.status}`);
    }

    ({ answer } = await ragRes.json());
  } catch (err) {
    console.error("RAG backend request failed:", err.message);
    answer = "Sorry, I can't reach my knowledge base right now. Please try again in a moment.";
  }

  // Mimic Ollama's own streaming chunk shape so the existing frontend parser
  // (which expects { message: { content } } ndjson lines) needs no changes.
  res.write(JSON.stringify({ message: { content: answer || "" } }) + "\n");
  res.end();
});

app.post("/api/speak", async (req, res) => {
  const { text } = req.body;

  if (!text || !text.trim()) {
    res.status(400).json({ error: "text is required" });
    return;
  }

  try {
    const avatarRes = await fetch(`${AVATAR_URL}/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!avatarRes.ok) {
      const detail = await avatarRes.text();
      console.error(`Avatar service /speak failed (${avatarRes.status}):`, detail);
      res.status(502).json({ error: "Avatar service failed", detail });
      return;
    }

    const { video_url } = await avatarRes.json();
    // Return a same-origin path (proxied below), not an absolute localhost:8010
    // URL — the latter breaks for any client not on this exact machine (e.g.
    // through the ngrok tunnel) and can get mixed-content blocked over HTTPS.
    res.json({ video_url });
  } catch (err) {
    res.status(502).json({ error: "Avatar service unreachable", detail: err.message });
  }
});

app.post("/api/speak/stream", async (req, res) => {
  const { text } = req.body;

  if (!text || !text.trim()) {
    res.status(400).json({ error: "text is required" });
    return;
  }

  try {
    const avatarRes = await fetch(`${AVATAR_URL}/speak/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!avatarRes.ok || !avatarRes.body) {
      const detail = await avatarRes.text().catch(() => "");
      console.error(`Avatar service /speak/stream failed (${avatarRes.status}):`, detail);
      res.status(502).json({ error: "Avatar service failed", detail });
      return;
    }

    // Straight byte passthrough of the upstream SSE stream -- same shape as
    // the /avatar-video proxy below, just for an event-stream body instead
    // of a video file. No reframing needed: the client parses SSE directly
    // against this same-origin response.
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    for await (const chunk of avatarRes.body) {
      res.write(chunk);
    }
    res.end();
  } catch (err) {
    res.status(502).json({ error: "Avatar service unreachable", detail: err.message });
  }
});

app.get("/avatar-video/:filename", async (req, res) => {
  try {
    const avatarRes = await fetch(`${AVATAR_URL}/avatar-video/${req.params.filename}`);
    if (!avatarRes.ok || !avatarRes.body) {
      res.status(avatarRes.status || 502).end();
      return;
    }
    res.setHeader("Content-Type", avatarRes.headers.get("content-type") || "video/mp4");
    for await (const chunk of avatarRes.body) {
      res.write(chunk);
    }
    res.end();
  } catch (err) {
    res.status(502).json({ error: "Avatar service unreachable", detail: err.message });
  }
});

app.post("/api/transcribe", upload.single("audio"), async (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: "audio file is required" });
    return;
  }

  try {
    const form = new FormData();
    form.append(
      "audio",
      new Blob([req.file.buffer], { type: req.file.mimetype || "application/octet-stream" }),
      req.file.originalname || "audio.webm",
    );

    const avatarRes = await fetch(`${AVATAR_URL}/transcribe`, {
      method: "POST",
      body: form,
    });

    if (!avatarRes.ok) {
      const detail = await avatarRes.text();
      res.status(502).json({ error: "Transcription failed", detail });
      return;
    }

    const data = await avatarRes.json();
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: "Avatar service unreachable", detail: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`AI Agent running at http://localhost:${PORT}`);
  console.log(`Using RAG backend at ${RAG_BACKEND_URL}`);
});
