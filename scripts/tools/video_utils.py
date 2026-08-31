"""Save RGB frames to PNG sequence and optionally encode MP4 via ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def rgb_to_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def write_png_sequence(frames: list[np.ndarray], out_dir: Path, *, prefix: str = "frame") -> list[Path]:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(rgb_to_uint8(frame))
        p = out_dir / f"{prefix}_{i:04d}.png"
        img.save(p)
        paths.append(p)
    return paths


def _find_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    conda = Path.home() / "miniconda3" / "bin" / "ffmpeg"
    return str(conda) if conda.is_file() else None


def encode_mp4(png_dir: Path, mp4_path: Path, *, fps: int = 30, pattern: str = "frame_%04d.png") -> bool:
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        return False
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(png_dir / pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return mp4_path.is_file()
    except subprocess.CalledProcessError:
        return False


def save_video(
    frames: list[np.ndarray],
    out_mp4: Path,
    *,
    fps: int = 30,
    keep_png: bool = False,
) -> dict:
    """Write frames to MP4 (via ffmpeg) or PNG fallback."""
    png_dir = out_mp4.parent / (out_mp4.stem + "_frames")
    write_png_sequence(frames, png_dir)
    meta = {
        "mp4": str(out_mp4),
        "num_frames": len(frames),
        "fps": fps,
        "png_dir": str(png_dir),
        "mp4_encoded": False,
    }
    if encode_mp4(png_dir, out_mp4, fps=fps):
        meta["mp4_encoded"] = True
        if not keep_png:
            shutil.rmtree(png_dir, ignore_errors=True)
            meta.pop("png_dir", None)
    else:
        meta["note"] = "ffmpeg not available; PNG sequence saved instead"
    manifest = out_mp4.with_suffix(".json")
    manifest.write_text(json.dumps(meta, indent=2))
    return meta
