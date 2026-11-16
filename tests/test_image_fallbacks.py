import sys
import numpy as np
import pytest
from PIL import Image
from clingeo.data import image_io


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_pure_png_reader_preserves_grayscale_pixels(tmp_path, mode):
    raw = np.arange(20 * 24, dtype=np.uint8).reshape(20, 24)
    path = tmp_path / "synthetic.png"
    Image.fromarray(raw).convert(mode).save(path)
    assert np.array_equal(image_io._read_png_grayscale(path), raw)


def test_loader_fallback_and_cache_paths(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.png"
    raw = np.arange(256, dtype=np.uint8).reshape(16, 16)
    Image.fromarray(raw).save(path)
    cache = tmp_path / "cache"
    assert np.array_equal(image_io.read_grayscale(path, cache), raw)
    assert np.array_equal(image_io.read_grayscale(path, cache), raw)
    monkeypatch.setenv("CLINGEO_PIL_PYTHON", sys.executable)
    def fail(*args): raise RuntimeError("simulate an unavailable decoder")
    monkeypatch.setattr(image_io, "_read_with_pillow", fail)
    assert np.array_equal(image_io.read_grayscale(path), raw)
    monkeypatch.setattr(image_io, "_read_with_system_pillow", fail)
    assert np.array_equal(image_io.read_grayscale(path), raw)
    with pytest.raises(RuntimeError): image_io.read_grayscale(tmp_path / "bad.jpg")
    with pytest.raises(ValueError): image_io.read_grayscale(path, expected_sha256="bad")
    rgb = np.ones((2, 2, 3), dtype=np.float32) * 300
    assert image_io._ensure_grayscale_uint8(rgb).tolist() == [[255, 255], [255, 255]]


@pytest.mark.parametrize("filter_id,expected", [
    (0, [1, 2, 3]), (1, [1, 3, 6]), (2, [11, 22, 33]),
    (3, [6, 15, 25]), (4, [11, 22, 33]),
])
def test_png_filter_known_vectors(filter_id, expected):
    actual = image_io._png_unfilter(np.array([1, 2, 3], dtype=np.uint8),
        np.array([10, 20, 30], dtype=np.uint8), filter_id, 1)
    assert actual.tolist() == expected


def test_png_rejects_invalid_encoding(tmp_path):
    path = tmp_path / "bad.png"; path.write_bytes(b"not a png")
    with pytest.raises(ValueError): image_io._read_png_grayscale(path)
    with pytest.raises(ValueError): image_io._png_unfilter(np.ones(3), np.ones(3), 9, 1)
