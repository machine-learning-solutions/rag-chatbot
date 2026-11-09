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


def warmup() -> None:
    """Run one dummy forward pass through each model at the exact shapes
    generate() uses (full Wav2Lip batch, single GFPGAN call) so cuDNN's
    benchmark-mode algorithm search pays its one-time cost now instead of
    during the first real request."""
    if _model is None or _cached_face is None:
        return

    img_t = torch.zeros(WAV2LIP_BATCH_SIZE, 6, IMG_SIZE, IMG_SIZE, device=DEVICE)
    mel_t = torch.zeros(WAV2LIP_BATCH_SIZE, 1, 80, MEL_STEP_SIZE, device=DEVICE)
    with torch.no_grad():
        _model(mel_t, img_t)

    if _restore_enabled:
        face_restore.restore(_cached_face["frame"])

    if DEVICE == "cuda":
        torch.cuda.synchronize()


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


def _ffmpeg_cmd(out_path, frame_w, frame_h, even_w, even_h, audio_path, audio_start, audio_duration):
    """Build the ffmpeg command that mixes a raw-BGR-frame stdin pipe with
    `audio_path` into out_path. audio_start/audio_duration (seconds, or None
    for "use the whole file") are applied as an input-level seek/trim on the
    audio input only, placed before its -i per ffmpeg's input-option rules
    so they never affect the rawvideo input before it -- used by
    generate_chunks() to carve each segment's matching audio slice out of
    the one full synthesized WAV without needing per-segment WAV files.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{frame_w}x{frame_h}",
        "-r",
        str(FPS),
        "-i",
        "-",
    ]
    if audio_start is not None:
        cmd += ["-ss", f"{audio_start:.6f}"]
    if audio_duration is not None:
        cmd += ["-t", f"{audio_duration:.6f}"]
    cmd += [
        "-i",
        audio_path,
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-vf",
        f"crop={even_w}:{even_h}:0:0",
        "-c:v",
        "libx264",
        "-q:v",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        out_path,
    ]
    return cmd


def _generate_segments(audio_path, out_path_fn, chunk_frames):
    """Core generator behind generate() and generate_chunks(). Runs Wav2Lip
    inference against the cached static face in groups of `chunk_frames`
    video frames (None = one segment covering the whole clip), writes each
    segment to its own single-streamed-encode mp4 (out_path_fn(index)), and
    yields (path, index, is_last) as soon as that segment's file is ready --
    before later segments have even started rendering.
    """
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
    # libx264's yuv420p output needs even dimensions; the reference frame's
    # raw size isn't guaranteed to be (e.g. 678x715). The old cv2.VideoWriter
    # (DIVX) path this replaced silently rounded this down before its own
    # encode, so cropping to even here matches that existing behavior.
    even_w = frame_w - (frame_w % 2)
    even_h = frame_h - (frame_h % 2)

    segment_size = chunk_frames if chunk_frames is not None else len(mel_chunks)
    segments = [mel_chunks[s : s + segment_size] for s in range(0, len(mel_chunks), segment_size)]

    # GPU-batching GFPGAN restoration was tried and rejected (see face_restore.py
    # docstring: no win at small batch sizes, OOM risk at large ones on this 8GB
    # card). Instead, restore only every RESTORE_EVERY_N-th frame and reuse the
    # raw (unrestored) Wav2Lip frame for the rest -- a frame counter running
    # across the WHOLE clip (not reset per Wav2Lip batch, and not reset per
    # segment either) keeps the alternating pattern consistent regardless of
    # batch or chunk boundaries.
    frame_idx = 0

    for seg_index, seg_mel_chunks in enumerate(segments):
        is_last = seg_index == len(segments) - 1
        seg_out_path = out_path_fn(seg_index)

        # Single-segment calls (generate()) pass the audio straight through
        # unmodified -- no -ss/-t -- so that path stays byte-for-byte what it
        # was before this function existed. Multi-segment calls trim each
        # segment's matching slice out of the one full synthesized WAV.
        if chunk_frames is None:
            cmd = _ffmpeg_cmd(seg_out_path, frame_w, frame_h, even_w, even_h, audio_path, None, None)
        else:
            seg_start_sec = (seg_index * segment_size) / FPS
            seg_duration_sec = len(seg_mel_chunks) / FPS
            cmd = _ffmpeg_cmd(
                seg_out_path, frame_w, frame_h, even_w, even_h, audio_path, seg_start_sec, seg_duration_sec
            )

        ffmpeg_stderr_path = seg_out_path + ".ffmpeg.log"
        ffmpeg_stderr_file = open(ffmpeg_stderr_path, "wb")
        # stderr -> a file, not a pipe: ffmpeg writes periodic progress lines
        # there for the segment's whole duration, and an unread PIPE would
        # fill its OS buffer and deadlock against this loop's stdin writes.
        proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=ffmpeg_stderr_file)

        _t_seg_start = time.perf_counter()
        _t_w2l_total = 0.0
        _t_restore_total = 0.0
        _t_write_total = 0.0
        _n_frames = 0
        _n_restored = 0

        try:
            for img_batch, mel_batch, frames, coords in _datagen(seg_mel_chunks):
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
                    try:
                        proc.stdin.write(f.tobytes())
                    except BrokenPipeError as exc:
                        proc.wait()
                        with open(ffmpeg_stderr_path, "rb") as fh:
                            err_tail = fh.read()[-1500:]
                        raise RuntimeError(
                            f"ffmpeg exited early: {err_tail.decode(errors='ignore')}"
                        ) from exc
                    _t_write_total += time.perf_counter() - _t_d
                    frame_idx += 1
                    _n_frames += 1
        finally:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            proc.wait()
            ffmpeg_stderr_file.close()
        # By this point ffmpeg has already consumed and encoded every frame
        # of this segment concurrently with the loop above (it was draining
        # the pipe as fast as frames arrived, far faster than GFPGAN's
        # per-frame GPU cost) -- wait() just lets it flush the trailer.
        _t_seg_done = time.perf_counter()

        if proc.returncode != 0 or not os.path.exists(seg_out_path):
            with open(ffmpeg_stderr_path, "rb") as fh:
                err_tail = fh.read()[-1500:]
            os.remove(ffmpeg_stderr_path)
            raise RuntimeError(f"ffmpeg mux failed: {err_tail.decode(errors='ignore')}")
        os.remove(ffmpeg_stderr_path)

        print(
            f"[TIMING] segment={seg_index}/{len(segments) - 1} frames={_n_frames} restored={_n_restored} "
            f"wav+mel={_t_mel - _t0:.3f}s "
            f"wav2lip_fwd_total={_t_w2l_total:.3f}s "
            f"restore_total={_t_restore_total:.3f}s "
            f"videowriter_total={_t_write_total:.3f}s "
            f"segment_wallclock={_t_seg_done - _t_seg_start:.3f}s "
            f"since_request_start={_t_seg_done - _t0:.3f}s",
            flush=True,
        )

        yield seg_out_path, seg_index, is_last


def generate(audio_path: str, out_path: str) -> None:
    """Run Wav2Lip inference against the cached static face, writing a
    lip-synced mp4 to out_path. Thin single-segment wrapper around
    _generate_segments() -- kept so existing (non-streaming) callers don't
    need to change."""
    for _ in _generate_segments(audio_path, lambda i: out_path, chunk_frames=None):
        pass


def generate_chunks(audio_path: str, out_path_fn, chunk_seconds: float = 4.0):
    """Like generate(), but splits the clip into ~chunk_seconds segments and
    yields (path, index, is_last) for each as soon as it's ready, instead of
    waiting for the whole clip -- lets a streaming caller start playback
    after the first segment instead of after the full render. out_path_fn
    is called with each 0-based segment index and must return a unique path
    for it."""
    chunk_frames = max(1, round(chunk_seconds * FPS))
    yield from _generate_segments(audio_path, out_path_fn, chunk_frames=chunk_frames)
