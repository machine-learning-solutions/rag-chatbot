import os

# Must be set before `import torch` initializes the CUDA allocator. This GPU
# runs right at the edge of its 8GB (avatar models ~4.7-5.2GB resident +
# Ollama ~2GB when warm leave only a few hundred MB slack -- confirmed via a
# live CUDA OOM: "58.45 MiB is reserved by PyTorch but unallocated"), so
# fragmentation alone can tip an allocation over even when enough memory is
# technically free in aggregate. expandable_segments lets the caching
# allocator grow/shrink segments instead of hunting for a contiguous
# fixed-size block -- PyTorch's own OOM message suggests this exact setting.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import re
import sys
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path

import requests
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
from faster_whisper import WhisperModel
from piper import PiperVoice
from piper.config import SynthesisConfig

# cuDNN benchmark mode autotunes and caches the fastest conv algorithm for the
# (always-identical, per TIMING logs the dominant cost by far) GFPGAN 512x512
# call and Wav2Lip's batched forward -- pure algorithm selection, no precision
# change. TF32 matmul trims the remaining fp32 GEMMs on this RTX 4060's Ada
# tensor cores; unlike the naive .half() cast face_restore.py's docstring
# already tried and rejected (garbled StyleGAN2 output), TF32's reduced
# mantissa stays well clear of the dynamic-range instability that caused that,
# so it doesn't carry the same quality risk. cudnn.allow_tf32 is already True
# by default in this torch version -- set explicitly so behavior doesn't
# depend on the installed version's default.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BASE_DIR = Path(__file__).resolve().parent
WAV2LIP_DIR = BASE_DIR / "Wav2Lip"
CHECKPOINT = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"
GFPGAN_CHECKPOINT = BASE_DIR / "gfpgan_weights" / "GFPGANv1.4.pth"
REFERENCE_FACE = BASE_DIR.parent / "avatar" / "reference.png"
VOICES_DIR = BASE_DIR / "piper_voices"
PIPER_VOICE_EN_PATH = VOICES_DIR / "en_US-hfc_male-medium.onnx"  # lessac turned out to sound female
PIPER_VOICE_AR_PATH = VOICES_DIR / "ar_JO-kareem-medium.onnx"
OUTPUT_DIR = BASE_DIR / "outputs"

# Ollama's own compose port, exposed on the host (see chatbot/docker-compose.yml).
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")
OUTPUT_DIR.mkdir(exist_ok=True)

# /speak/stream segment length. Render time scales ~1:1 with speech length
# (GFPGAN restoration is the fixed-cost bottleneck, already near its measured
# hardware floor), so for a long answer there's no way to make the full clip
# render fast -- but the client doesn't need the full clip to start playing,
# just the first few seconds of it. ~4s keeps time-to-first-chunk low while
# not fragmenting short answers (which still render as a single chunk).
STREAM_CHUNK_SECONDS = float(os.environ.get("AVATAR_STREAM_CHUNK_SECONDS", "4.0"))

# Piper length_scale: lower = faster speech. Arabic (kareem) defaults to 1.0
# and reads slow for portfolio answers; English stays at the model default.
AR_LENGTH_SCALE = float(os.environ.get("PIPER_AR_LENGTH_SCALE", "0.75"))

sys.path.insert(0, str(WAV2LIP_DIR))
import resident_inference  # noqa: E402  (Wav2Lip's own flat-module imports need WAV2LIP_DIR on sys.path first)

ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")


def detect_voice(text: str) -> PiperVoice:
    arabic_chars = len(ARABIC_RE.findall(text))
    if arabic_chars > max(3, len(text) * 0.15):
        return PIPER_VOICE_AR
    return PIPER_VOICE_EN


app = FastAPI()

# Dual Whisper: English large-v3-turbo stays on GPU for fast mic turns.
# Arabic multi-dialect CT2 fine-tune stays on CPU (int8) so its weights do
# not compete with Wav2Lip/GFPGAN on this 8GB card during /speak/stream --
# mic clips are short enough that CPU AR STT is acceptable.
WHISPER_EN_MODEL_ID = os.environ.get("WHISPER_EN_MODEL", "large-v3-turbo")
WHISPER_AR_MODEL_ID = os.environ.get(
    "WHISPER_AR_MODEL",
    "dev-ahmedhany/whisper-large-v3-turbo-arabic-ft-ct2-int8",
)


def _load_whisper(model_id: str, device: str, compute_type: str) -> WhisperModel:
    try:
        return WhisperModel(model_id, device=device, compute_type=compute_type)
    except Exception as exc:
        print(
            f"[WARN] Whisper load failed ({model_id} on {device}/{compute_type}): {exc}; "
            "falling back to CPU int8",
            flush=True,
        )
        return WhisperModel(model_id, device="cpu", compute_type="int8")


WHISPER_MODEL_EN = _load_whisper(WHISPER_EN_MODEL_ID, "cuda", "float16")
try:
    WHISPER_MODEL_AR = _load_whisper(WHISPER_AR_MODEL_ID, "cpu", "int8")
except Exception as exc:
    print(f"[WARN] Arabic Whisper unavailable ({exc}); Arabic falls back to EN model", flush=True)
    WHISPER_MODEL_AR = WHISPER_MODEL_EN

# Piper voices loaded once (CPU) instead of spawning `python -m piper` per request.
PIPER_VOICE_EN = PiperVoice.load(str(PIPER_VOICE_EN_PATH))
PIPER_VOICE_AR = PiperVoice.load(str(PIPER_VOICE_AR_PATH))

# Wav2Lip generator + S3FD face detector loaded once, and the (always-identical)
# reference face is face-detected once and cached, instead of reloading both
# checkpoints and re-running face detection on every request via a subprocess.
# A GFPGAN restoration pass (also resident, also landmark-cached) sharpens
# Wav2Lip's inherently blurry 96px-upscaled output.
resident_inference.init(str(CHECKPOINT), str(REFERENCE_FACE), str(GFPGAN_CHECKPOINT))

# Run one throwaway forward pass through each model now, at startup, so
# cudnn.benchmark's one-time algorithm search (set above) happens here
# instead of adding its cost to the first real user request.
resident_inference.warmup()

# --- GPU clock keep-alive -----------------------------------------------
# Profiling found the dominant source of LATENCY VARIANCE (not the ~2.2s
# steady-state floor itself, which is GFPGAN restoration cost) was this
# laptop GPU dropping into a deep idle power state (P5, ~435MHz) between
# requests. The first CUDA kernel launch after that idle gap pays a clock
# ramp-up penalty, making Wav2Lip's forward pass balloon from ~70ms to
# ~1.4s on an otherwise-identical call (confirmed via torch.cuda.synchronize
# timing around just that step, with nothing else changed).
#
# Properly fixing this needs a one-time, system-level `sudo nvidia-smi -lgc
# <min>,<max>` to lock clocks -- out of scope to run ourselves (modifying
# system GPU power settings). This lightweight background thread is an
# application-level mitigation: a trivial matmul every 50ms keeps the GPU
# busy enough that the driver doesn't downclock as deeply, which measured
# ~4/5 calls hitting the fast steady-state floor instead of randomly
# spiking to 3.5-5.3s. Tiny footprint (256x256 matmul, well under 1ms of
# actual compute) -- the cost is a small amount of idle power draw, not
# competing GPU memory or meaningfully competing compute.
def _gpu_keepalive():
    if not torch.cuda.is_available():
        return
    x = torch.zeros(256, 256, device="cuda")
    while True:
        x = x @ x
        torch.cuda.synchronize()
        time.sleep(0.05)


threading.Thread(target=_gpu_keepalive, daemon=True).start()

# --- Output cleanup -------------------------------------------------------
# Nothing was ever deleting generated files from OUTPUT_DIR (confirmed: 420
# files / 200MB had accumulated since Nov 8, well before chunked streaming
# existed). /speak/stream makes this acutely worse -- a single long answer
# now produces dozens of small .mp4s instead of one file. Clients fetch each
# chunk once, shortly after it's generated, so anything older than a
# generous safety margin (longest observed render for one answer: ~4.5min)
# is safe to remove. Sweeps immediately at startup (clearing any backlog)
# and on an interval after that.
OUTPUT_MAX_AGE_SECONDS = 20 * 60
OUTPUT_SWEEP_INTERVAL_SECONDS = 5 * 60


def _cleanup_old_outputs():
    now = time.time()
    removed = 0
    for path in OUTPUT_DIR.iterdir():
        try:
            if path.is_file() and (now - path.stat().st_mtime) > OUTPUT_MAX_AGE_SECONDS:
                path.unlink()
                removed += 1
        except OSError:
            pass  # file removed/replaced concurrently -- fine, skip it
    if removed:
        print(f"[CLEANUP] removed {removed} stale file(s) from {OUTPUT_DIR}", flush=True)


def _cleanup_loop():
    while True:
        _cleanup_old_outputs()
        time.sleep(OUTPUT_SWEEP_INTERVAL_SECONDS)


threading.Thread(target=_cleanup_loop, daemon=True).start()


class SpeakRequest(BaseModel):
    text: str


def _unload_ollama() -> None:
    """Best-effort: free Ollama's VRAM before the GPU-heavy Wav2Lip/GFPGAN
    step below. By the time /speak runs, Ollama has already produced the
    chat answer this speech is for and has no more work to do this turn --
    from here its only effect is competing for the same ~8GB the avatar
    models need (this is what the CUDA OOM in PYTORCH_CUDA_ALLOC_CONF's
    comment above traced back to: avatar models + a still-warm Ollama left
    only ~100MB free). keep_alive=0 on a prompt-less generate call is
    Ollama's documented way to unload immediately rather than waiting out
    its normal keep-alive window. A small cold-reload cost on Ollama's
    *next* chat request is a fair trade for not crashing this one.
    """
    try:
        requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "keep_alive": 0},
            timeout=5,
        )
    except requests.RequestException:
        pass  # Ollama being unreachable/slow should never block avatar generation.


def synthesize_speech(text: str, wav_path: Path, voice: PiperVoice) -> None:
    syn_config = None
    if voice is PIPER_VOICE_AR:
        syn_config = SynthesisConfig(length_scale=AR_LENGTH_SCALE)
    with wave.open(str(wav_path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=syn_config)
    if not wav_path.exists():
        raise RuntimeError("Piper TTS failed: no output file written")


def generate_lipsync_video(wav_path: Path, video_path: Path) -> None:
    resident_inference.generate(str(wav_path), str(video_path))
    if not video_path.exists():
        raise RuntimeError("Wav2Lip inference failed: no output file written")


@app.post("/speak")
def speak(req: SpeakRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    job_id = uuid.uuid4().hex
    wav_path = OUTPUT_DIR / f"{job_id}.wav"
    video_path = OUTPUT_DIR / f"{job_id}.mp4"
    voice = detect_voice(text)

    try:
        _t0 = time.perf_counter()
        synthesize_speech(text, wav_path, voice)
        _t_tts = time.perf_counter()

        # Ollama has already done its job for this turn (producing the chat
        # answer we're now speaking) -- give its VRAM back before the
        # GPU-heavy step below needs it. See _unload_ollama()'s docstring.
        _unload_ollama()

        try:
            generate_lipsync_video(wav_path, video_path)
        except torch.cuda.OutOfMemoryError:
            # Fallback in case the unload above didn't free enough in time
            # (e.g. Ollama was mid-generation for a *different* request).
            # empty_cache() only frees blocks *this* process was holding but
            # not using, so a brief wait first gives Ollama's unload a moment
            # to actually land before retrying into the same contention.
            print("[WARN] /speak: CUDA OOM, retrying once in 2s after cache clear", flush=True)
            torch.cuda.empty_cache()
            time.sleep(2)
            generate_lipsync_video(wav_path, video_path)

        _t_video = time.perf_counter()
        print(
            f"[TIMING] /speak tts={_t_tts - _t0:.3f}s lipsync_endpoint={_t_video - _t_tts:.3f}s "
            f"total={_t_video - _t0:.3f}s text_len={len(text)}",
            flush=True,
        )
    except Exception as exc:
        # Broadened from (RuntimeError, ValueError): a failure here was
        # returning a bare 500 with no detail in the log (the except clause
        # only matched our own raised errors, so anything else -- e.g. a
        # Piper phonemizer error on unexpected characters -- vanished
        # silently). Log the actual text that triggered it so this is
        # diagnosable next time instead of just "500 Internal Server Error".
        print(
            f"[ERROR] /speak failed: {type(exc).__name__}: {exc} "
            f"text_len={len(text)} text={text[:300]!r}",
            flush=True,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if wav_path.exists():
            wav_path.unlink()

    return {"video_url": f"/avatar-video/{job_id}.mp4"}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.post("/speak/stream")
def speak_stream(req: SpeakRequest):
    """Like /speak, but emits each ~STREAM_CHUNK_SECONDS segment as an SSE
    event as soon as it's ready, instead of waiting for the whole clip.
    Lets the client start playback after the first segment (a few seconds)
    rather than after the full render, which for a long answer can take
    minutes (render time tracks speech length almost 1:1 -- see
    resident_inference.py).

    Event sequence:
      1. After TTS: {audio_url, chunk_seconds} -- full wav for continuous
         master-audio playback on the client (video chunks stay muted).
      2. Per segment: {chunk_index, chunk_start, chunk_duration,
         video_url, is_last} with exact frame-derived timings.

    On failure mid-stream, emits {"error": "..."} and ends -- the client
    plays whatever chunks already arrived and stops there, rather than this
    endpoint trying to patch a partial failure back together.

    The wav is intentionally NOT deleted when the stream ends: the client
    may still be playing it after the last video chunk arrives. The output
    sweeper removes it after OUTPUT_MAX_AGE_SECONDS.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    job_id = uuid.uuid4().hex
    wav_path = OUTPUT_DIR / f"{job_id}.wav"
    voice = detect_voice(text)

    def event_gen():
        try:
            _t0 = time.perf_counter()
            synthesize_speech(text, wav_path, voice)
            _t_tts = time.perf_counter()

            # Same reasoning as /speak: Ollama's done its job for this turn,
            # give its VRAM back before the GPU-heavy step needs it.
            _unload_ollama()

            # Full TTS wav first so the client can play one unbroken audio
            # track while lip-sync video chunks swap underneath.
            yield _sse(
                {
                    "audio_url": f"/avatar-video/{job_id}.wav",
                    "chunk_seconds": STREAM_CHUNK_SECONDS,
                }
            )

            def out_path_fn(i):
                return str(OUTPUT_DIR / f"{job_id}_{i}.mp4")

            for (
                seg_path,
                seg_index,
                is_last,
                chunk_start,
                chunk_duration,
            ) in resident_inference.generate_chunks(
                str(wav_path), out_path_fn, chunk_seconds=STREAM_CHUNK_SECONDS
            ):
                filename = Path(seg_path).name
                yield _sse(
                    {
                        "chunk_index": seg_index,
                        "chunk_start": chunk_start,
                        "chunk_duration": chunk_duration,
                        "video_url": f"/avatar-video/{filename}",
                        "is_last": is_last,
                    }
                )

            _t_video = time.perf_counter()
            print(
                f"[TIMING] /speak/stream tts={_t_tts - _t0:.3f}s lipsync_endpoint={_t_video - _t_tts:.3f}s "
                f"total={_t_video - _t0:.3f}s text_len={len(text)}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[ERROR] /speak/stream failed: {type(exc).__name__}: {exc} "
                f"text_len={len(text)} text={text[:300]!r}",
                flush=True,
            )
            yield _sse({"error": str(exc)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


DELETE_AFTER_SERVE_DELAY_SECONDS = 5


def _delete_served_output(path: Path) -> None:
    # <video>/<audio> elements fetch via HTTP Range (FileResponse supports
    # it), issuing several separate GETs for the same URL -- one for an
    # initial probe range, more for the rest of the bytes. Deleting right
    # after the *first* of those finishes (this used to run inline, with no
    # delay) left the file gone before the follow-up range requests arrived,
    # 404ing them and aborting playback mid-chunk (confirmed: browser logs
    # showed a 200 then a 404 for the identical chunk URL). A short delay
    # lets same-chunk follow-up requests land first; the periodic sweep in
    # _cleanup_loop is the backstop if this races with it or a client never
    # finishes reading the response.
    time.sleep(DELETE_AFTER_SERVE_DELAY_SECONDS)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


@app.get("/avatar-video/{filename}")
def get_video(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists() or ".." in filename:
        raise HTTPException(status_code=404, detail="not found")
    is_wav = path.suffix.lower() == ".wav"
    media_type = "audio/wav" if is_wav else "video/mp4"
    # The client fetches the master .wav early (as soon as it streams in) to
    # start audio playback while later chunks are still rendering, but those
    # later chunks' ffmpeg mux still needs to read this same .wav from disk
    # (see resident_inference._ffmpeg_cmd). Deleting it on first fetch, like
    # we do for .mp4 chunks, raced ffmpeg out from under itself and aborted
    # the stream ("No such file or directory"). The .wav is left for
    # _cleanup_loop's sweep instead, per speak_stream's own docstring.
    return FileResponse(
        path,
        media_type=media_type,
        background=None if is_wav else BackgroundTask(_delete_served_output, path),
    )


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("en"),
):
    suffix = Path(audio.filename or "").suffix or ".webm"
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio upload")

    lang = (language or "en").strip().lower()
    use_arabic = lang.startswith("ar")
    model = WHISPER_MODEL_AR if use_arabic else WHISPER_MODEL_EN
    forced_language = "ar" if use_arabic else "en"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        segments, info = model.transcribe(
            str(tmp_path),
            language=forced_language,
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )
        text = "".join(seg.text for seg in segments).strip()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"text": text, "language": info.language or forced_language}


@app.get("/health")
def health():
    return {
        "ok": True,
        "checkpoint_exists": CHECKPOINT.exists(),
        "reference_face_exists": REFERENCE_FACE.exists(),
        "piper_voice_en_exists": PIPER_VOICE_EN_PATH.exists(),
        "piper_voice_ar_exists": PIPER_VOICE_AR_PATH.exists(),
        "whisper_en_model": WHISPER_EN_MODEL_ID,
        "whisper_ar_model": WHISPER_AR_MODEL_ID,
    }
