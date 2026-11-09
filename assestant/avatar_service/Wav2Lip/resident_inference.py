"""
In-process, model-resident replacement for the inference.py CLI.

inference.py is an argparse script that reloads the Wav2Lip generator and the
S3FD face detector from disk on every invocation, and (being a CLI) pays full
Python/torch/CUDA startup cost per call. This module loads everything once
(call init() at process startup) and exposes generate() for cheap, repeated
calls from a long-lived server process.

Behavior mirrors the exact parameters the old subprocess call used:
  --pads 0 20 0 0, img_size=96, fps=25, face_det_batch_size=16,
  wav2lip_batch_size=128, static face image (single frame, face-detected once).
"""
import os
import subprocess as sp
import time

import cv2
import numpy as np
import torch

import audio
import face_detection
import face_restore
from models import Wav2Lip

IMG_SIZE = 96
MEL_STEP_SIZE = 16
PADS = (0, 20, 0, 0)  # top, bottom, left, right -- matches prior subprocess call
WAV2LIP_BATCH_SIZE = 128
FPS = 25.0
RESTORE_EVERY_N = 2  # restore every Nth frame with GFPGAN; reuse raw Wav2Lip output between

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_cached_face = None  # {"frame": ndarray, "coords": (y1,y2,x1,x2), "face_resized": ndarray}
_restore_enabled = False


def _load(checkpoint_path):
    if DEVICE == "cuda":
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    else:
        checkpoint = torch.load(
            checkpoint_path, weights_only=False, map_location=lambda storage, loc: storage
        )
    return checkpoint


def _load_wav2lip_model(checkpoint_path):
    model = Wav2Lip()
    checkpoint = _load(checkpoint_path)
    s = checkpoint["state_dict"]
    new_s = {k.replace("module.", ""): v for k, v in s.items()}
    model.load_state_dict(new_s)
    model = model.to(DEVICE)
    return model.eval()


def init(checkpoint_path: str, face_image_path: str, gfpgan_checkpoint_path: str = None) -> None:
    """Load the Wav2Lip generator + S3FD detector and cache the static face crop. Call once at startup."""
    global _model, _cached_face, _restore_enabled

    _model = _load_wav2lip_model(checkpoint_path)

    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False, device=DEVICE
    )

    frame = cv2.imread(face_image_path)
    if frame is None:
        raise ValueError(f"Could not read face image: {face_image_path}")

    predictions = detector.get_detections_for_batch(np.array([frame]))
    rect = predictions[0]
    if rect is None:
        raise ValueError("Face not detected in reference image")

    del detector  # only needed once; frees detector weights from memory/VRAM

    pady1, pady2, padx1, padx2 = PADS
    y1 = max(0, rect[1] - pady1)
    y2 = min(frame.shape[0], rect[3] + pady2)
    x1 = max(0, rect[0] - padx1)
    x2 = min(frame.shape[1], rect[2] + padx2)

    face_crop = frame[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))

    _cached_face = {
        "frame": frame,
        "coords": (y1, y2, x1, x2),
        "face_resized": face_resized,
    }

    if gfpgan_checkpoint_path:
        face_restore.init(gfpgan_checkpoint_path, frame)
        _restore_enabled = True


def _prepare_batch(img_batch, mel_batch, frame_batch, coords_batch):
    img_batch_np = np.asarray(img_batch)
    mel_batch_np = np.asarray(mel_batch)

    img_masked = img_batch_np.copy()
    img_masked[:, IMG_SIZE // 2 :] = 0

    img_batch_np = np.concatenate((img_masked, img_batch_np), axis=3) / 255.0
    mel_batch_np = np.reshape(
        mel_batch_np, [len(mel_batch_np), mel_batch_np.shape[1], mel_batch_np.shape[2], 1]
    )
    return img_batch_np, mel_batch_np, frame_batch, coords_batch


def _datagen(mel_chunks):
    face_resized = _cached_face["face_resized"]
    frame = _cached_face["frame"]
    coords = _cached_face["coords"]

    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    for m in mel_chunks:
        img_batch.append(face_resized)
        mel_batch.append(m)
        frame_batch.append(frame.copy())
        coords_batch.append(coords)

        if len(img_batch) >= WAV2LIP_BATCH_SIZE:
            yield _prepare_batch(img_batch, mel_batch, frame_batch, coords_batch)
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if img_batch:
        yield _prepare_batch(img_batch, mel_batch, frame_batch, coords_batch)


def generate(audio_path: str, out_path: str) -> None:
    """Run Wav2Lip inference against the cached static face, writing a lip-synced mp4 to out_path."""
    if _model is None or _cached_face is None:
        raise RuntimeError("resident_inference.init() must be called before generate()")

    _t0 = time.perf_counter()
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    _t_mel = time.perf_counter()

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            "Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again"
        )

    mel_chunks = []
    mel_idx_multiplier = 80.0 / FPS
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + MEL_STEP_SIZE > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - MEL_STEP_SIZE :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + MEL_STEP_SIZE])
        i += 1

    frame = _cached_face["frame"]
    frame_h, frame_w = frame.shape[:-1]

    tmp_avi = out_path + ".tmp.avi"
    out = cv2.VideoWriter(tmp_avi, cv2.VideoWriter_fourcc(*"DIVX"), FPS, (frame_w, frame_h))

    # GPU-batching GFPGAN restoration was tried and rejected (see face_restore.py
    # docstring: no win at small batch sizes, OOM risk at large ones on this 8GB
    # card). Instead, restore only every RESTORE_EVERY_N-th frame and reuse the
    # raw (unrestored) Wav2Lip frame for the rest -- a frame counter running
    # across the whole clip (not reset per Wav2Lip batch) keeps the pattern
    # consistent regardless of batch boundaries.
    frame_idx = 0
    _t_w2l_total = 0.0
    _t_restore_total = 0.0
    _t_write_total = 0.0
    _n_frames = 0
    _n_restored = 0

    try:
        for img_batch, mel_batch, frames, coords in _datagen(mel_chunks):
            img_batch_t = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(DEVICE)
            mel_batch_t = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(DEVICE)

            _t_a = time.perf_counter()
            with torch.no_grad():
                pred = _model(mel_batch_t, img_batch_t)
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0  # .cpu() already syncs
            _t_w2l_total += time.perf_counter() - _t_a

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                if _restore_enabled and frame_idx % RESTORE_EVERY_N == 0:
                    _t_c = time.perf_counter()
                    f = face_restore.restore(f)
                    _t_restore_total += time.perf_counter() - _t_c
                    _n_restored += 1
                _t_d = time.perf_counter()
                out.write(f)
                _t_write_total += time.perf_counter() - _t_d
                frame_idx += 1
                _n_frames += 1
    finally:
        out.release()
    _t_frames_done = time.perf_counter()

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-i",
        tmp_avi,
        "-strict",
        "-2",
        "-q:v",
        "1",
        out_path,
    ]
    result = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    _t_ffmpeg_done = time.perf_counter()
    try:
        os.remove(tmp_avi)
    except OSError:
        pass
    if result.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg mux failed: {result.stderr.decode(errors='ignore')[-1500:]}")

    print(
        f"[TIMING] frames={_n_frames} restored={_n_restored} "
        f"wav+mel={_t_mel - _t0:.3f}s "
        f"wav2lip_fwd_total={_t_w2l_total:.3f}s "
        f"restore_total={_t_restore_total:.3f}s "
        f"videowriter_total={_t_write_total:.3f}s "
        f"frame_loop_wallclock={_t_frames_done - _t_mel:.3f}s "
        f"ffmpeg_mux={_t_ffmpeg_done - _t_frames_done:.3f}s "
        f"generate_total={_t_ffmpeg_done - _t0:.3f}s",
        flush=True,
    )
