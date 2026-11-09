import re
import sys
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
from piper import PiperVoice

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
OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(WAV2LIP_DIR))
import resident_inference  # noqa: E402  (Wav2Lip's own flat-module imports need WAV2LIP_DIR on sys.path first)

ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")


def detect_voice(text: str) -> PiperVoice:
    arabic_chars = len(ARABIC_RE.findall(text))
    if arabic_chars > max(3, len(text) * 0.15):
        return PIPER_VOICE_AR
    return PIPER_VOICE_EN


app = FastAPI()

# "large-v3" multilingual: noticeably more accurate than "medium" on both
# English and Arabic, especially for accents/noisy audio. Running on GPU
# (float16) — confirmed enough free VRAM alongside Ollama/LFM3 + Wav2Lip.
# Loaded once at startup so per-request latency is just inference time.
try:
    WHISPER_MODEL = WhisperModel("large-v3", device="cuda", compute_type="float16")
except Exception:
    WHISPER_MODEL = WhisperModel("medium", device="cpu", compute_type="int8")

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


class SpeakRequest(BaseModel):
    text: str


def synthesize_speech(text: str, wav_path: Path, voice: PiperVoice) -> None:
    with wave.open(str(wav_path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
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
        generate_lipsync_video(wav_path, video_path)
        _t_video = time.perf_counter()
        print(
            f"[TIMING] /speak tts={_t_tts - _t0:.3f}s lipsync_endpoint={_t_video - _t_tts:.3f}s "
            f"total={_t_video - _t0:.3f}s text_len={len(text)}",
            flush=True,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if wav_path.exists():
            wav_path.unlink()

    return {"video_url": f"/avatar-video/{job_id}.mp4"}


@app.get("/avatar-video/{filename}")
def get_video(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists() or ".." in filename:
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="video/mp4")


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    suffix = Path(audio.filename or "").suffix or ".webm"
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio upload")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        segments, info = WHISPER_MODEL.transcribe(
            str(tmp_path),
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

    return {"text": text, "language": info.language}


@app.get("/health")
def health():
    return {
        "ok": True,
        "checkpoint_exists": CHECKPOINT.exists(),
        "reference_face_exists": REFERENCE_FACE.exists(),
        "piper_voice_en_exists": PIPER_VOICE_EN_PATH.exists(),
        "piper_voice_ar_exists": PIPER_VOICE_AR_PATH.exists(),
    }
