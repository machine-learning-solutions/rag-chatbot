const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const avatarIdle = document.getElementById("avatarIdle");
const avatarVideo = document.getElementById("avatarVideo");
const avatarStatus = document.getElementById("avatarStatus");
const avatarWrap = document.getElementById("avatarWrap");
const micBtn = document.getElementById("micBtn");
const micStatus = document.getElementById("micStatus");

function updateInputPlaceholder() {
  inputEl.placeholder = window.innerWidth < 400
    ? "Ask me anything… / اسألني"
    : "How can I help you today? / كيف يمكنني مساعدتك اليوم؟";
}
updateInputPlaceholder();
window.addEventListener("resize", updateInputPlaceholder);

let history = [];

// --- Continuous "keep listening" mode (client-side voice activity detection) ---
let continuousMode = false;
let micStream = null;
let audioContext = null;
let analyser = null;
let mediaRecorder = null;
let recordedChunks = [];
let vadRafId = null;
let bargeInRafId = null;
let segmentActive = false;
let busy = false; // true while transcribing, waiting on LFM3, or avatar is speaking
let turnController = null; // AbortController for the in-flight transcribe/chat/speak calls

const SILENCE_MS = 1200; // pause length that ends a spoken segment
const MIN_SPEECH_MS = 300; // ignore blips shorter than this
const MAX_SEGMENT_MS = 20000; // hard cap per segment
const VOLUME_THRESHOLD = 0.02; // RMS level considered "speech" — tune if too sensitive/insensitive for your mic
const BARGE_IN_MS = 250; // sustained loud audio required to interrupt the avatar mid-reply

function setMicStatus(text, isError = false) {
  micStatus.textContent = text;
  micStatus.classList.toggle("error", isError);
}

function pickMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];
  return candidates.find((t) => window.MediaRecorder?.isTypeSupported?.(t)) || "";
}

function currentVolume() {
  const data = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(data);
  let sumSquares = 0;
  for (let i = 0; i < data.length; i++) {
    const normalized = (data[i] - 128) / 128;
    sumSquares += normalized * normalized;
  }
  return Math.sqrt(sumSquares / data.length);
}

async function enableContinuousMode() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    setMicStatus(`Microphone access denied or unavailable: ${err.message}`, true);
    continuousMode = false;
    return;
  }

  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioContext.createMediaStreamSource(micStream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 512;
  source.connect(analyser);

  micBtn.classList.add("recording");
  micBtn.title = "Click to stop continuous listening";
  setMicStatus("Listening… (use headphones so the avatar's own voice isn't picked up)");
  startSegment();
}

function disableContinuousMode() {
  continuousMode = false;
  if (vadRafId) cancelAnimationFrame(vadRafId);
  vadRafId = null;
  segmentActive = false;
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.onstop = null;
    mediaRecorder.stop();
  }
  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  micBtn.classList.remove("recording");
  micBtn.title = "Click to start continuous listening";
  setMicStatus("");
}

function startSegment() {
  if (!continuousMode || busy || !micStream) return;
  if (mediaRecorder && mediaRecorder.state === "recording") return; // already listening

  const mimeType = pickMimeType();
  try {
    mediaRecorder = mimeType
      ? new MediaRecorder(micStream, { mimeType })
      : new MediaRecorder(micStream);
  } catch (err) {
    setMicStatus(`Could not start recorder: ${err.message}`, true);
    disableContinuousMode();
    return;
  }

  recordedChunks = [];
  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.onstop = async () => {
    segmentActive = false;
    const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" });
    await transcribeAndSend(blob);
    if (continuousMode && !busy) startSegment();
  };

  mediaRecorder.start();
  segmentActive = true;

  let hasSpeech = false;
  let lastLoudAt = Date.now();
  const segmentStartedAt = Date.now();

  const loop = () => {
    if (!segmentActive) return;
    const vol = currentVolume();
    const now = Date.now();

    if (vol > VOLUME_THRESHOLD) {
      lastLoudAt = now;
      if (!hasSpeech && now - segmentStartedAt > MIN_SPEECH_MS) hasSpeech = true;
    }

    const silentFor = now - lastLoudAt;
    const segmentAge = now - segmentStartedAt;

    if ((hasSpeech && silentFor > SILENCE_MS) || segmentAge > MAX_SEGMENT_MS) {
      if (!hasSpeech) {
        // nothing but silence/background noise — restart listening without sending
        mediaRecorder.onstop = () => {
          segmentActive = false;
          if (continuousMode && !busy) startSegment();
        };
      }
      mediaRecorder.stop();
      return;
    }

    vadRafId = requestAnimationFrame(loop);
  };
  vadRafId = requestAnimationFrame(loop);
}

// While busy (transcribing / waiting on LFM3 / avatar speaking), keep watching
// the mic. If the user starts talking, interrupt everything immediately so
// they can barge in instead of waiting for the avatar to finish.
function startBargeInWatcher() {
  let loudSince = null;
  const loop = () => {
    if (!continuousMode || !busy) {
      bargeInRafId = null;
      return;
    }
    const vol = currentVolume();
    const now = Date.now();

    if (vol > VOLUME_THRESHOLD) {
      if (loudSince === null) loudSince = now;
      else if (now - loudSince > BARGE_IN_MS) {
        interruptAndListen();
        return;
      }
    } else {
      loudSince = null;
    }

    bargeInRafId = requestAnimationFrame(loop);
  };
  bargeInRafId = requestAnimationFrame(loop);
}

function interruptAndListen() {
  if (turnController) {
    turnController.abort();
    turnController = null;
  }
  avatarVideo.pause(); // resolves the "wait for playback to end" promise in speakReply()
  avatarVideo.currentTime = 0;
  avatarVideo.style.display = "none";
  avatarIdle.style.display = "block";
  avatarStatus.textContent = "";
  busy = false;
  setMicStatus("Heard you — go ahead…");
  startSegment();
}

async function transcribeAndSend(blob) {
  if (blob.size < 1000) return; // too short to contain real speech

  busy = true;
  const controller = new AbortController();
  turnController = controller;
  if (continuousMode) startBargeInWatcher();
  setMicStatus("Transcribing…");

  try {
    const formData = new FormData();
    const ext = blob.type.includes("mp4") ? "mp4" : blob.type.includes("ogg") ? "ogg" : "webm";
    formData.append("audio", blob, `recording.${ext}`);

    const res = await fetch("/api/transcribe", { method: "POST", body: formData, signal: controller.signal });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || err.detail || `HTTP ${res.status}`);
    }

    const { text, language } = await res.json();

    if (!text || !text.trim()) {
      setMicStatus(continuousMode ? "Listening…" : "Didn't catch any speech.");
      return;
    }

    setMicStatus(`Heard (${language || "?"}): "${text.trim()}"`);
    inputEl.value = text.trim();
    await send(controller.signal); // awaits the full reply AND the avatar speaking it, so the
    // mic only resumes listening once the avatar has finished — unless interrupted (see above).
    if (continuousMode) setMicStatus("Listening…");
  } catch (err) {
    if (err.name !== "AbortError") {
      setMicStatus(`Transcription error: ${err.message}`, true);
    }
  } finally {
    busy = false;
    if (turnController === controller) turnController = null;
  }
}

micBtn.addEventListener("click", () => {
  if (continuousMode) {
    disableContinuousMode();
  } else {
    continuousMode = true;
    enableContinuousMode();
  }
});

async function speakReply(text, signal) {
  avatarStatus.textContent = "generating avatar…";
  avatarStatus.classList.add("active");

  try {
    const res = await fetch("/api/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }

    const { video_url } = await res.json();

    avatarVideo.src = video_url;
    avatarIdle.style.display = "none";
    avatarVideo.style.display = "block";
    avatarStatus.textContent = "";

    // Wait for playback to actually finish (or be cut short by an interruption's
    // .pause() call) — not just for it to start — so `busy` stays true, and the
    // mic stays paused, for the full duration the avatar is actually talking.
    avatarWrap.classList.add("speaking");
    await new Promise((resolve) => {
      avatarVideo.onended = () => {
        avatarVideo.style.display = "none";
        avatarIdle.style.display = "block";
        resolve();
      };
      avatarVideo.onpause = () => resolve();
      avatarVideo.play().catch(() => resolve());
    });
  } catch (err) {
    if (err.name !== "AbortError") {
      avatarStatus.textContent = `avatar error: ${err.message}`;
    }
  } finally {
    avatarStatus.classList.remove("active");
    avatarWrap.classList.remove("speaking");
  }
}

function stripMarkdown(text) {
  return text
    .replace(/```[\s\S]*?```/g, (block) => block.replace(/```/g, ""))
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/__(.*?)__/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/(?<!\w)_(.*?)_(?!\w)/g, "$1")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*\d+\.\s+/gm, "")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/\n{3,}/g, "\n\n");
}

function addBubble(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.dir = "auto";
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

async function send(signal) {
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = "";
  sendBtn.disabled = true;

  history.push({ role: "user", content: text });
  addBubble("user", text);

  const assistantBubble = addBubble("assistant", "");
  assistantBubble.classList.add("thinking");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
      signal,
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let full = "";
    let thinking = "";
    let startedContent = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.trim()) continue;
        const json = JSON.parse(line);

        if (json.message?.content) {
          if (!startedContent) {
            startedContent = true;
            assistantBubble.classList.remove("thinking");
          }
          full += json.message.content;
          assistantBubble.textContent = stripMarkdown(full);
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (json.message?.thinking && !startedContent) {
          thinking += json.message.thinking;
          assistantBubble.textContent = `thinking… ${thinking.slice(-120)}`;
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
      }
    }

    history.push({ role: "assistant", content: full });

    const cleanReply = stripMarkdown(full).trim();
    if (cleanReply) {
      await speakReply(cleanReply, signal);
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      assistantBubble.textContent = `Error: ${err.message}`;
    }
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

sendBtn.addEventListener("click", () => send());
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
