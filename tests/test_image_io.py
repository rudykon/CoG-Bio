from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clingeo.data.dataset import BiometryDataset
from clingeo.data.image_io import read_grayscale


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checksum_bound_cache_does_not_reuse_legacy_path_entry(tmp_path: Path) -> None:
    source = tmp_path / "image.npy"
    expected = np.full((8, 9), 17, dtype=np.uint8)
    np.save(source, expected)
    cache = tmp_path / "cache"

    legacy = read_grayscale(source, cache_dir=cache)
    assert np.array_equal(legacy, expected)

    current = np.full((8, 9), 93, dtype=np.uint8)
    np.save(source, current)
    checksum = _sha(source)
    observed = read_grayscale(
        source,
        cache_dir=cache,
        expected_sha256=checksum,
    )

    assert np.array_equal(observed, current)
    assert (cache / "sha256" / checksum[:2] / f"{checksum}.npy").is_file()


def test_checksum_bound_cache_rejects_wrong_source_bytes(tmp_path: Path) -> None:
    source = tmp_path / "image.npy"
    np.save(source, np.zeros((4, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="source image SHA-256 mismatch"):
        read_grayscale(
            source,
            cache_dir=tmp_path / "cache",
            expected_sha256="0" * 64,
        )


def test_biometry_dataset_uses_manifest_checksum_cache_namespace(
    tmp_path: Path,
) -> None:
    source = tmp_path / "images" / "case.npy"
    source.parent.mkdir(parents=True)
    np.save(source, np.full((64, 80), 31, dtype=np.uint8))
    checksum = _sha(source)
    frame = pd.DataFrame(
        [
            {
                "split": "train_labeled",
                "domain": "FUGC",
                "case_id": "case",
                "image_path": "images/case.npy",
                "height": 64,
                "width": 80,
                "landmarks": "[[20.0,16.0],[30.0,24.0]]",
                "valid_mask": "[1,1]",
                "clinical_params": "{}",
                "checksum": checksum,
            }
        ]
    )
    cache = tmp_path / "cache"

    item = BiometryDataset(
        frame,
        data_root=tmp_path,
        input_size=64,
        training=False,
        cache_dir=cache,
    )[0]

    assert item["image"].shape == (3, 64, 64)
    assert (cache / "sha256" / checksum[:2] / f"{checksum}.npy").is_file()
