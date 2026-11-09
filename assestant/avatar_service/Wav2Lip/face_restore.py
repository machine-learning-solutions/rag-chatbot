"""
GFPGAN-based face restoration, applied per-frame to sharpen Wav2Lip's output.

Wav2Lip's generator only predicts a 96x96 face, upscaled with a plain
cv2.resize() to fit back into the source frame -- this is why its output
looks soft/blurry. GFPGAN restores detail in that region after the fact.

Tuned for a single always-identical static face:
  - Landmark detection + the affine matrix (landmark -> template) and its
    inverse are computed ONCE in init() via one real align_warp_face()/
    get_inverse_affine() call, then reused directly with cv2.warpAffine for
    every subsequent frame (no FaceRestoreHelper bookkeeping, no per-call
    clean_all()/read_image()).
  - The blend mask FaceRestoreHelper's use_parse=False branch computes
    (erode + erode + Gaussian blur, geometry-only, depends only on the
    inverse affine + face_size) is *also* constant for a static face --
    precomputed ONCE in init() instead of recomputed on every frame.

Batching multiple frames into one GFPGAN forward call was tried and
REJECTED: benchmarked on this RTX 4060 Laptop (8GB), small batches (2-8)
were actually *slower* per-frame than batch=1 (GFPGAN's StyleGAN2-clean
generator at 512x512 doesn't parallelize well at small batch sizes on this
card), and batches large enough to help (16+) use enough VRAM (~1.2-1.5GB
extra) to risk OOM once Ollama + Whisper + Wav2Lip are also resident
(confirmed: batch=96 silently OOM'd even in an isolated test with far more
free memory than the full production stack has). The ~80ms/frame GPU cost
is effectively a fixed floor for this model on this hardware -- speed
control is handled by frame-skipping in resident_inference.py instead,
not by batching here.

fp16 was also tried and rejected: naively `.half()`-ing GFPGAN's StyleGAN2
generator produces garbled, featureless output (a known instability with
naive half-precision on modulated-convolution architectures). Stay fp32.
"""
import sys
import types

import cv2
import numpy as np
import torch

# basicsr (a GFPGAN dependency) imports a torchvision internal module that
# was removed in torchvision >= 0.17. This repo has torchvision 0.20.1.
# Inject a compatibility shim before anything imports basicsr/gfpgan.
if "torchvision.transforms.functional_tensor" not in sys.modules:
    from torchvision.transforms import functional as _F

    _shim = types.ModuleType("torchvision.transforms.functional_tensor")
    _shim.rgb_to_grayscale = _F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _shim

from basicsr.utils import img2tensor, tensor2img
from gfpgan import GFPGANer
from torchvision.transforms.functional import normalize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_restorer = None
_affine_matrix = None
_inverse_affine = None
_face_size = None
_mask_restored = None  # precomputed: multiplies the warped-back restored face
_mask_background = None  # precomputed: multiplies the original background


def init(checkpoint_path: str, reference_frame) -> None:
    """Load GFPGAN and precompute the (frame-invariant) affine + blend-mask geometry."""
    global _restorer, _affine_matrix, _inverse_affine, _face_size
    global _mask_restored, _mask_background

    _restorer = GFPGANer(
        model_path=checkpoint_path,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        device=DEVICE,
    )
    helper = _restorer.face_helper
    helper.use_parse = False  # skip the parsing-network mask; see module docstring

    helper.clean_all()
    helper.read_image(reference_frame)
    num_faces = helper.get_face_landmarks_5(only_center_face=True, eye_dist_threshold=5)
    if num_faces == 0:
        raise ValueError("face_restore.init: no face detected in reference frame")

    helper.align_warp_face()
    helper.get_inverse_affine(None)
    _affine_matrix = helper.affine_matrices[0]
    _inverse_affine = helper.inverse_affine_matrices[0]
    _face_size = helper.face_size
    upscale_factor = helper.upscale_factor  # 1, per GFPGANer(upscale=1) above

    h, w = reference_frame.shape[:2]

    # Replicate FaceRestoreHelper.paste_faces_to_input_image's use_parse=False
    # mask math exactly, but ONCE: this geometry never changes for a static face.
    mask = np.ones(_face_size, dtype=np.float32)
    inv_mask = cv2.warpAffine(mask, _inverse_affine, (w, h))
    inv_mask_erosion = cv2.erode(
        inv_mask, np.ones((int(2 * upscale_factor), int(2 * upscale_factor)), np.uint8)
    )
    total_face_area = np.sum(inv_mask_erosion)
    w_edge = int(total_face_area**0.5) // 20
    erosion_radius = w_edge * 2
    inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
    blur_size = w_edge * 2
    inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)

    _mask_restored = (inv_mask_erosion * inv_soft_mask)[:, :, None].astype(np.float32)
    _mask_background = (1 - inv_soft_mask)[:, :, None].astype(np.float32)

    # The detector/helper's bookkeeping is only needed for that one setup call.
    del helper.face_det
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def restore(frame):
    """Restore the face region of `frame`. Falls back to the original frame on any failure."""
    if _restorer is None or _affine_matrix is None:
        raise RuntimeError("face_restore.init() must be called before restore()")

    try:
        cropped = cv2.warpAffine(
            frame, _affine_matrix, _face_size, borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132)
        )
        t = img2tensor(cropped / 255.0, bgr2rgb=True, float32=True)
        normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        t = t.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = _restorer.gfpgan(t, return_rgb=False)[0]
        restored = tensor2img(out.squeeze(0), rgb2bgr=True, min_max=(-1, 1)).astype(np.float32)

        h, w = frame.shape[:2]
        inv_restored = cv2.warpAffine(restored, _inverse_affine, (w, h))
        blended = _mask_restored * inv_restored + _mask_background * frame.astype(np.float32)
        return np.clip(blended, 0, 255).astype(np.uint8)
    except Exception:
        return frame
