import numpy as np

from clingeo.data.transforms import clahe, histogram_equalize, make_three_channel


def test_clahe_returns_float_image_in_unit_range():
    img = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    out = clahe(img, tiles=4, clip_limit=2.0)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_make_three_channel_keeps_legacy_global_he_selectable():
    img = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    chw = make_three_channel(img, ch2="global_he")
    raw = chw[0]
    expected = histogram_equalize(raw)
    np.testing.assert_allclose(chw[1], expected, atol=1e-6)


def test_make_three_channel_clahe_has_same_shape_as_legacy():
    img = np.random.default_rng(0).normal(size=(48, 40)).astype(np.float32)
    clahe_chw = make_three_channel(img, ch2="clahe", clahe_tiles=4, clahe_clip=2.0)
    legacy_chw = make_three_channel(img, ch2="global_he")
    assert clahe_chw.shape == legacy_chw.shape == (3, 48, 40)
    assert clahe_chw.dtype == np.float32
