from __future__ import annotations

import hashlib
import os
import re
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_grayscale(
    path: str | Path,
    cache_dir: str | Path | None = None,
    *,
    expected_sha256: str | None = None,
) -> np.ndarray:
    """Read one grayscale image with an optional content-bound cache.

    Historical caches were keyed only by the resolved path.  That can return
    stale pixels when a source file is replaced in place.  When a manifest
    checksum is available, use it both as the cache identity and as a source
    integrity check before creating a missing cache entry.
    """

    path = Path(path)
    expected = _validated_expected_sha256(expected_sha256)
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = _cache_path(path, cache_dir, source_sha256=expected)
        if cache_path.exists():
            return _ensure_grayscale_uint8(np.load(cache_path))
    if expected is not None and sha256_file(path) != expected:
        raise ValueError(f"source image SHA-256 mismatch: {path}")
    if path.suffix.lower() == ".npy":
        arr = _ensure_grayscale_uint8(np.load(path))
        if cache_path is not None:
            _write_cache_atomically(cache_path, arr)
        return arr
    try:
        arr = _read_with_pillow(path)
    except Exception:
        try:
            arr = _read_with_system_pillow(path)
        except Exception:
            if path.suffix.lower() == ".png":
                arr = _read_png_grayscale(path)
            else:
                raise
    arr = _ensure_grayscale_uint8(arr)
    if cache_path is not None:
        _write_cache_atomically(cache_path, arr)
    return arr


def _cache_path(
    path: Path,
    cache_dir: str | Path,
    *,
    source_sha256: str | None = None,
) -> Path:
    if source_sha256 is not None:
        return (
            Path(cache_dir)
            / "sha256"
            / source_sha256[:2]
            / f"{source_sha256}.npy"
        )
    resolved = str(path.resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.npy"


def _validated_expected_sha256(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    normalized = str(value).strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError("expected image SHA-256 must be 64 lowercase hex characters")
    return normalized


def _write_cache_atomically(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npy", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.save(temporary, arr)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_grayscale_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float32).mean(axis=2)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _read_with_pillow(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("L"))


def _read_with_system_pillow(path: Path) -> np.ndarray:
    helper = os.environ.get("CLINGEO_PIL_PYTHON", "/usr/bin/python3.12")
    code = (
        "import sys, struct\n"
        "from PIL import Image\n"
        "p=sys.argv[1]\n"
        "with Image.open(p) as im:\n"
        "    im=im.convert('L')\n"
        "    w,h=im.size\n"
        "    sys.stdout.buffer.write(struct.pack('<II', h, w))\n"
        "    sys.stdout.buffer.write(im.tobytes())\n"
    )
    proc = subprocess.run([helper, "-c", code, str(path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if len(proc.stdout) < 8:
        raise RuntimeError(f"System Pillow returned no bytes for {path}: {proc.stderr.decode('utf-8', 'ignore')}")
    h, w = struct.unpack("<II", proc.stdout[:8])
    raw = proc.stdout[8:]
    expected = h * w
    if len(raw) != expected:
        raise RuntimeError(f"System Pillow returned {len(raw)} bytes for {path}, expected {expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w).copy()


def _read_png_grayscale(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG file")
    pos = 8
    width = height = bit_depth = color_type = interlace = None
    idat = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filter, interlace = struct.unpack(">IIBBBBB", chunk)
        elif chunk_type == b"IDAT":
            idat.extend(chunk)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None:
        raise ValueError(f"{path} has no IHDR")
    if bit_depth != 8 or interlace != 0:
        raise ValueError(f"Unsupported PNG format in {path}: bit_depth={bit_depth}, interlace={interlace}")
    channels_by_type = {0: 1, 2: 3, 4: 2, 6: 4}
    if color_type not in channels_by_type:
        raise ValueError(f"Unsupported PNG color type {color_type} in {path}")
    channels = channels_by_type[color_type]
    stride = width * channels
    raw = zlib.decompress(bytes(idat))
    rows = []
    prev = np.zeros(stride, dtype=np.uint8)
    offset = 0
    for _ in range(height):
        filt = raw[offset]
        offset += 1
        row = np.frombuffer(raw[offset : offset + stride], dtype=np.uint8).copy()
        offset += stride
        recon = _png_unfilter(row, prev, filt, channels)
        rows.append(recon)
        prev = recon
    arr = np.stack(rows, axis=0).reshape(height, width, channels)
    if channels == 1:
        return arr[..., 0]
    return arr[..., :3].astype(np.float32).mean(axis=2).astype(np.uint8)


def _png_unfilter(row: np.ndarray, prev: np.ndarray, filt: int, bpp: int) -> np.ndarray:
    out = row.astype(np.int16)
    prev_i = prev.astype(np.int16)
    if filt == 0:
        pass
    elif filt == 1:
        for i in range(bpp, out.size):
            out[i] = (out[i] + out[i - bpp]) & 0xFF
    elif filt == 2:
        out = (out + prev_i) & 0xFF
    elif filt == 3:
        for i in range(out.size):
            left = out[i - bpp] if i >= bpp else 0
            up = prev_i[i]
            out[i] = (out[i] + ((left + up) // 2)) & 0xFF
    elif filt == 4:
        for i in range(out.size):
            left = out[i - bpp] if i >= bpp else 0
            up = prev_i[i]
            up_left = prev_i[i - bpp] if i >= bpp else 0
            out[i] = (out[i] + _paeth(left, up, up_left)) & 0xFF
    else:
        raise ValueError(f"Unsupported PNG filter {filt}")
    return out.astype(np.uint8)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c
