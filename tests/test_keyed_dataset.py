from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from clingeo.data.dataset import BiometryDataset, collate_biometry
from clingeo.training.keyed_augmentation import (
    CommittedExposureLedger,
    freeze_exposure_run_contract,
    issue_exposures,
)


def _dataset_and_exposures(tmp_path: object):
    from helpers import make_synthetic_manifest

    manifest = make_synthetic_manifest(tmp_path)
    frame = pd.read_csv(manifest)
    frame["manifest_row_id"] = [f"row-{index}" for index in range(len(frame))]
    dataset = BiometryDataset(
        frame,
        data_root=tmp_path,
        input_size=64,
        training=True,
        cache_dir=None,
        require_keyed_augmentation=True,
    )
    contract = freeze_exposure_run_contract(
        tuple(frame["manifest_row_id"]),
        training_seed=42,
        partition_seed=20260711,
        outer_fold=0,
        phase="inner_selection",
        frozen_exposures_per_epoch=len(frame),
    )
    ledger = CommittedExposureLedger.empty(contract)
    return dataset, issue_exposures(contract, ledger=ledger, count=len(frame))


def test_keyed_dataset_is_invariant_to_process_global_rng_state(tmp_path) -> None:
    dataset, exposures = _dataset_and_exposures(tmp_path)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    first = dataset[(0, exposures[0])]
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    second = dataset[(0, exposures[0])]

    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["landmarks"], second["landmarks"])
    assert first["augmentation_record_sha256"] == exposures[0].record_sha256
    assert first["augmentation_draw_sha256"] == exposures[0].draw.draw_sha256
    assert first["exposure_record"] == exposures[0]


def test_keyed_dataset_requires_record_and_matching_manifest_row(tmp_path) -> None:
    dataset, exposures = _dataset_and_exposures(tmp_path)
    with pytest.raises(ValueError, match="requires a keyed exposure"):
        dataset[0]
    with pytest.raises(ValueError, match="manifest_row_id mismatch"):
        dataset[(1, exposures[0])]


def test_keyed_dataset_rejects_tampered_exposure(tmp_path) -> None:
    dataset, exposures = _dataset_and_exposures(tmp_path)
    forged = replace(exposures[0], record_sha256="0" * 64)
    with pytest.raises(ValueError, match="record hash mismatch"):
        dataset[(0, forged)]


def test_collate_preserves_exposure_records_for_optimizer_commit(tmp_path) -> None:
    dataset, exposures = _dataset_and_exposures(tmp_path)
    batch = collate_biometry(
        [dataset[(0, exposures[0])], dataset[(1, exposures[1])]]
    )
    assert batch["exposure_record"] == [exposures[0], exposures[1]]
    assert batch["augmentation_exposure_index"] == [0, 1]


def test_dataloader_sampler_can_issue_row_and_exposure_pairs(tmp_path) -> None:
    dataset, exposures = _dataset_and_exposures(tmp_path)
    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=[(0, exposures[0]), (1, exposures[1])],
        collate_fn=collate_biometry,
        num_workers=0,
    )
    batch = next(iter(loader))
    assert batch["augmentation_exposure_index"] == [0, 1]
    assert batch["exposure_record"] == [exposures[0], exposures[1]]


def test_evaluation_dataset_rejects_exposure_tuple(tmp_path) -> None:
    from helpers import make_synthetic_manifest

    manifest = make_synthetic_manifest(tmp_path)
    frame = pd.read_csv(manifest)
    frame["manifest_row_id"] = [f"row-{index}" for index in range(len(frame))]
    contract = freeze_exposure_run_contract(
        tuple(frame["manifest_row_id"]),
        training_seed=42,
        partition_seed=20260711,
        outer_fold=0,
        phase="inner_selection",
        frozen_exposures_per_epoch=len(frame),
    )
    exposure = issue_exposures(
        contract, ledger=CommittedExposureLedger.empty(contract), count=1
    )[0]
    dataset = BiometryDataset(
        frame, data_root=tmp_path, input_size=64, training=False, cache_dir=None
    )
    with pytest.raises(ValueError, match="evaluation dataset"):
        dataset[(0, exposure)]
