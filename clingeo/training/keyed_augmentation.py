"""Stateless supervised augmentation keys and committed exposure ledgers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


KEYED_DRAW_PARAMETER_NAMES = (
    "hflip_gate",
    "rotation",
    "scale",
    "translate_x",
    "translate_y",
    "speckle_gate",
    "speckle_variance",
    "gamma_gate",
    "gamma",
    "gain_gate",
    "gain",
    "bias",
    "shadow_gate",
    "shadow_x",
    "shadow_width",
    "shadow_attenuation",
    "depth_gate",
    "depth_alpha",
)
KEYED_DRAW_SCHEMA = (
    ("format_version", 1),
    ("namespace", "hidden_mre_v2"),
    ("seed_encoding", "sha256_first_u64_big_endian"),
    ("normalized_parameters", KEYED_DRAW_PARAMETER_NAMES),
)
KEYED_DRAW_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        dict(KEYED_DRAW_SCHEMA),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INITIAL_ROLLING_SHA256 = hashlib.sha256(
    b"hidden_mre_v2|committed_exposure_ledger|v1"
).hexdigest()


@dataclass(frozen=True)
class AugmentationKey:
    training_seed: int
    partition_seed: int
    outer_fold: int
    phase: str
    manifest_row_id: str
    epoch: int
    exposure_index: int
    view_id: str
    seed_u64: int
    key_sha256: str


@dataclass(frozen=True)
class AugmentationDraw:
    schema_sha256: str
    key_sha256: str
    seed_u64: int
    normalized_parameters: tuple[tuple[str, float], ...]
    draw_sha256: str


@dataclass(frozen=True)
class _ExposureRunContractSeal:
    valid: bool
    error: str
    manifest_row_schedule: object
    scalar_snapshot: tuple[Any, ...]


@dataclass(frozen=True)
class ExposureRunContract:
    format_version: int
    training_seed: int
    partition_seed: int
    outer_fold: int
    phase: str
    frozen_exposures_per_epoch: int
    view_id: str
    world_size: int
    manifest_row_schedule: tuple[str, ...]
    manifest_row_schedule_sha256: str
    keyed_draw_schema_sha256: str
    sha256: str
    _seal: _ExposureRunContractSeal = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_seal", _make_run_contract_seal(self))


@dataclass(frozen=True)
class ExposureRecord:
    manifest_row_id: str
    epoch: int
    sampler_position: int
    exposure_index: int
    view_id: str
    run_contract_sha256: str
    augmentation_key: AugmentationKey
    augmentation_key_sha256: str
    draw: AugmentationDraw
    record_sha256: str


@dataclass(frozen=True)
class CommittedExposureLedger:
    format_version: int
    committed_exposure_cursor: int
    next_committed_exposure_index: int
    keyed_draw_schema_sha256: str
    run_contract_sha256: str
    rolling_committed_draw_ledger_sha256: str

    @classmethod
    def empty(
        cls, run_contract: ExposureRunContract
    ) -> "CommittedExposureLedger":
        contract = _validate_run_contract(run_contract)
        return cls(
            format_version=1,
            committed_exposure_cursor=0,
            next_committed_exposure_index=0,
            keyed_draw_schema_sha256=KEYED_DRAW_SCHEMA_SHA256,
            run_contract_sha256=contract.sha256,
            rolling_committed_draw_ledger_sha256=_INITIAL_ROLLING_SHA256,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format_version": int(self.format_version),
            "committed_exposure_cursor": int(self.committed_exposure_cursor),
            "next_committed_exposure_index": int(
                self.next_committed_exposure_index
            ),
            "keyed_draw_schema_sha256": self.keyed_draw_schema_sha256,
            "run_contract_sha256": self.run_contract_sha256,
            "rolling_committed_draw_ledger_sha256": self.rolling_committed_draw_ledger_sha256,
        }


def exposure_index(
    *, epoch: int, sampler_position: int, frozen_exposures_per_epoch: int
) -> int:
    epoch_value = _nonnegative_integer(epoch, "epoch")
    position = _nonnegative_integer(sampler_position, "sampler_position")
    stride = _positive_integer(
        frozen_exposures_per_epoch, "frozen_exposures_per_epoch"
    )
    if position >= stride:
        raise ValueError("sampler_position must be below frozen_exposures_per_epoch")
    return epoch_value * stride + position


def augmentation_key(
    manifest_row_id: str,
    *,
    training_seed: int,
    partition_seed: int,
    outer_fold: int,
    phase: str,
    epoch: int,
    exposure_index: int,
    view_id: str,
    worker_id: int | None = None,
) -> AugmentationKey:
    training = _nonnegative_integer(training_seed, "training_seed")
    partition = _nonnegative_integer(partition_seed, "partition_seed")
    fold = _nonnegative_integer(outer_fold, "outer_fold")
    if fold > 4:
        raise ValueError("outer_fold must be in range 0..4")
    phase_text = _nonblank(phase, "phase")
    row_id = _nonblank(manifest_row_id, "manifest_row_id")
    epoch_value = _nonnegative_integer(epoch, "epoch")
    exposure = _nonnegative_integer(exposure_index, "exposure_index")
    view = _nonblank(view_id, "view_id")
    if phase_text not in {"inner_selection", "outer_refit"}:
        raise ValueError("phase must be inner_selection or outer_refit")
    if any("|" in value for value in (phase_text, row_id, view)):
        raise ValueError("augmentation key fields cannot contain the '|' delimiter")
    if worker_id is not None:
        _nonnegative_integer(worker_id, "worker_id")
    payload = (
        f"hidden_mre_v2|{training}|{partition}|{fold}|{phase_text}|"
        f"{row_id}|{epoch_value}|{exposure}|{view}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return AugmentationKey(
        training_seed=training,
        partition_seed=partition,
        outer_fold=fold,
        phase=phase_text,
        manifest_row_id=row_id,
        epoch=epoch_value,
        exposure_index=exposure,
        view_id=view,
        seed_u64=int.from_bytes(digest[:8], "big", signed=False),
        key_sha256=digest.hex(),
    )


def augmentation_draw(key: AugmentationKey) -> AugmentationDraw:
    if not isinstance(key, AugmentationKey):
        raise ValueError("key must be an AugmentationKey")
    try:
        current_schema_sha256 = _keyed_draw_schema_sha256(KEYED_DRAW_SCHEMA)
    except (TypeError, ValueError) as exc:
        raise ValueError("keyed draw schema runtime drift detected") from exc
    if current_schema_sha256 != KEYED_DRAW_SCHEMA_SHA256:
        raise ValueError("keyed draw schema runtime drift detected")
    schema_parameter_names = dict(KEYED_DRAW_SCHEMA)["normalized_parameters"]
    if KEYED_DRAW_PARAMETER_NAMES != schema_parameter_names:
        raise ValueError("keyed draw schema runtime drift detected")
    try:
        rebuilt_key = augmentation_key(
            key.manifest_row_id,
            training_seed=key.training_seed,
            partition_seed=key.partition_seed,
            outer_fold=key.outer_fold,
            phase=key.phase,
            epoch=key.epoch,
            exposure_index=key.exposure_index,
            view_id=key.view_id,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("augmentation key dataclass tamper detected") from exc
    if key != rebuilt_key:
        raise ValueError("augmentation key dataclass tamper detected")
    parameters = tuple(
        (name, _normalized_hash_draw(key.key_sha256, name))
        for name in KEYED_DRAW_PARAMETER_NAMES
    )
    payload = _draw_payload(
        schema_sha256=KEYED_DRAW_SCHEMA_SHA256,
        key_sha256=key.key_sha256,
        seed_u64=key.seed_u64,
        normalized_parameters=parameters,
    )
    draw_sha = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return AugmentationDraw(
        schema_sha256=KEYED_DRAW_SCHEMA_SHA256,
        key_sha256=key.key_sha256,
        seed_u64=key.seed_u64,
        normalized_parameters=parameters,
        draw_sha256=draw_sha,
    )


def freeze_exposure_run_contract(
    manifest_row_schedule: Sequence[str],
    *,
    training_seed: int,
    partition_seed: int,
    outer_fold: int,
    phase: str,
    frozen_exposures_per_epoch: int,
    view_id: str = "original",
    world_size: int = 1,
) -> ExposureRunContract:
    if isinstance(manifest_row_schedule, (str, bytes)) or not isinstance(
        manifest_row_schedule, Sequence
    ):
        raise ValueError("manifest_row_schedule must be a sequence of row IDs")
    schedule = tuple(
        _nonblank(row_id, f"manifest_row_schedule[{index}]")
        for index, row_id in enumerate(manifest_row_schedule)
    )
    if not schedule:
        raise ValueError("manifest_row_schedule must not be empty")
    if any("|" in row_id for row_id in schedule):
        raise ValueError("manifest row IDs cannot contain the '|' delimiter")
    training = _nonnegative_integer(training_seed, "training_seed")
    partition = _nonnegative_integer(partition_seed, "partition_seed")
    fold = _nonnegative_integer(outer_fold, "outer_fold")
    if fold > 4:
        raise ValueError("outer_fold must be in range 0..4")
    phase_text = _supervised_phase(phase)
    stride = _positive_integer(
        frozen_exposures_per_epoch, "frozen_exposures_per_epoch"
    )
    view = _nonblank(view_id, "view_id")
    if "|" in view:
        raise ValueError("view_id cannot contain the '|' delimiter")
    replicas = _positive_integer(world_size, "world_size")
    schedule_sha = hashlib.sha256(_canonical_json(list(schedule))).hexdigest()
    payload = _run_contract_payload(
        training_seed=training,
        partition_seed=partition,
        outer_fold=fold,
        phase=phase_text,
        frozen_exposures_per_epoch=stride,
        view_id=view,
        world_size=replicas,
        manifest_row_schedule_length=len(schedule),
        manifest_row_schedule_sha256=schedule_sha,
        keyed_draw_schema_sha256=KEYED_DRAW_SCHEMA_SHA256,
    )
    contract = ExposureRunContract(
        format_version=1,
        training_seed=training,
        partition_seed=partition,
        outer_fold=fold,
        phase=phase_text,
        frozen_exposures_per_epoch=stride,
        view_id=view,
        world_size=replicas,
        manifest_row_schedule=schedule,
        manifest_row_schedule_sha256=schedule_sha,
        keyed_draw_schema_sha256=KEYED_DRAW_SCHEMA_SHA256,
        sha256=hashlib.sha256(_canonical_json(payload)).hexdigest(),
    )
    return _validate_run_contract(contract)


def issue_exposures(
    run_contract: ExposureRunContract,
    *,
    ledger: CommittedExposureLedger,
    count: int,
    rank: int | None = None,
) -> tuple[ExposureRecord, ...]:
    contract = _validate_run_contract(run_contract)
    _validate_ledger(ledger, expected_run_contract=contract)
    requested = _nonnegative_integer(count, "count")
    if rank is None:
        replica_rank = None
    else:
        replica_rank = _nonnegative_integer(rank, "rank")
        if replica_rank >= contract.world_size:
            raise ValueError("rank must be below run contract world_size")
    start = ledger.next_committed_exposure_index
    stop = start + requested
    if stop > len(contract.manifest_row_schedule):
        raise ValueError("manifest row schedule does not cover requested exposures")
    return tuple(
        _build_exposure_record(contract, global_index)
        for global_index in range(start, stop)
        if replica_rank is None
        or global_index % contract.world_size == replica_rank
    )


def exposure_record_sha256(record: ExposureRecord) -> str:
    if not isinstance(record, ExposureRecord):
        raise ValueError("record must be an ExposureRecord")
    if not isinstance(record.augmentation_key, AugmentationKey):
        raise ValueError("record augmentation_key must be an AugmentationKey")
    if not isinstance(record.draw, AugmentationDraw):
        raise ValueError("record draw must be an AugmentationDraw")
    return hashlib.sha256(_canonical_json(_record_payload(record))).hexdigest()


def commit_exposures(
    ledger: CommittedExposureLedger,
    records: Sequence[ExposureRecord],
    *,
    run_contract: ExposureRunContract,
) -> CommittedExposureLedger:
    contract = _validate_run_contract(run_contract)
    _validate_ledger(ledger, expected_run_contract=contract)
    expected_index = ledger.next_committed_exposure_index
    rolling = ledger.rolling_committed_draw_ledger_sha256
    for offset, record in enumerate(records):
        if not isinstance(record, ExposureRecord):
            raise ValueError("committed exposure records must be ExposureRecord values")
        if record.exposure_index != expected_index + offset:
            raise ValueError("committed exposures must form one contiguous interval")
        try:
            actual_record_hash = exposure_record_sha256(record)
        except (TypeError, ValueError) as exc:
            raise ValueError("committed exposure record hash mismatch") from exc
        if record.record_sha256 != actual_record_hash:
            raise ValueError("committed exposure record hash mismatch")
        try:
            expected_draw = augmentation_draw(record.augmentation_key)
        except (TypeError, ValueError) as exc:
            raise ValueError("committed exposure draw hash mismatch") from exc
        if record.draw != expected_draw:
            raise ValueError("committed exposure draw hash mismatch")
        expected_record = _build_exposure_record(
            contract, expected_index + offset
        )
        if record != expected_record:
            raise ValueError("committed exposure semantic mismatch with run contract")
        rolling = hashlib.sha256(
            bytes.fromhex(rolling) + bytes.fromhex(record.record_sha256)
        ).hexdigest()
    committed_count = len(records)
    return CommittedExposureLedger(
        format_version=1,
        committed_exposure_cursor=ledger.committed_exposure_cursor
        + committed_count,
        next_committed_exposure_index=expected_index + committed_count,
        keyed_draw_schema_sha256=KEYED_DRAW_SCHEMA_SHA256,
        run_contract_sha256=contract.sha256,
        rolling_committed_draw_ledger_sha256=rolling,
    )


def restore_committed_exposure_ledger(
    payload: Mapping[str, Any],
    *,
    expected_run_contract: ExposureRunContract,
) -> CommittedExposureLedger:
    contract = _validate_run_contract(expected_run_contract)
    if not isinstance(payload, Mapping):
        raise ValueError("committed exposure ledger checkpoint must be a mapping")
    required = {
        "format_version",
        "committed_exposure_cursor",
        "next_committed_exposure_index",
        "keyed_draw_schema_sha256",
        "run_contract_sha256",
        "rolling_committed_draw_ledger_sha256",
    }
    if set(payload) != required:
        raise ValueError("committed exposure ledger checkpoint keys are invalid")
    if _positive_integer(payload["format_version"], "format_version") != 1:
        raise ValueError("committed exposure ledger format_version must equal 1")
    cursor = _nonnegative_integer(
        payload["committed_exposure_cursor"], "committed_exposure_cursor"
    )
    next_index = _nonnegative_integer(
        payload["next_committed_exposure_index"],
        "next_committed_exposure_index",
    )
    if cursor != next_index:
        raise ValueError("committed exposure cursor/index mismatch")
    schema_sha = _sha256(
        payload["keyed_draw_schema_sha256"], "keyed draw schema"
    )
    if schema_sha != KEYED_DRAW_SCHEMA_SHA256:
        raise ValueError("keyed draw schema SHA-256 mismatch")
    contract_sha = _sha256(payload["run_contract_sha256"], "run contract")
    if contract_sha != contract.sha256:
        raise ValueError("run contract SHA-256 mismatch")
    rolling = _sha256(
        payload["rolling_committed_draw_ledger_sha256"],
        "rolling committed draw ledger",
    )
    ledger = CommittedExposureLedger(
        format_version=1,
        committed_exposure_cursor=cursor,
        next_committed_exposure_index=next_index,
        keyed_draw_schema_sha256=schema_sha,
        run_contract_sha256=contract_sha,
        rolling_committed_draw_ledger_sha256=rolling,
    )
    _validate_ledger(ledger, expected_run_contract=contract)
    return ledger


def _make_run_contract_seal(
    run_contract: ExposureRunContract,
) -> _ExposureRunContractSeal:
    valid = False
    error = "invalid exposure run contract"
    try:
        if _positive_integer(run_contract.format_version, "format_version") != 1:
            raise ValueError("format_version must equal 1")
        training = _nonnegative_integer(
            run_contract.training_seed, "training_seed"
        )
        partition = _nonnegative_integer(
            run_contract.partition_seed, "partition_seed"
        )
        fold = _nonnegative_integer(run_contract.outer_fold, "outer_fold")
        if fold > 4:
            raise ValueError("outer_fold must be in range 0..4")
        phase = _supervised_phase(run_contract.phase)
        stride = _positive_integer(
            run_contract.frozen_exposures_per_epoch,
            "frozen_exposures_per_epoch",
        )
        view = _nonblank(run_contract.view_id, "view_id")
        replicas = _positive_integer(run_contract.world_size, "world_size")
        if (
            training != run_contract.training_seed
            or partition != run_contract.partition_seed
            or fold != run_contract.outer_fold
            or phase != run_contract.phase
            or stride != run_contract.frozen_exposures_per_epoch
            or view != run_contract.view_id
            or replicas != run_contract.world_size
        ):
            raise ValueError("run contract fields are not canonical")
        schedule = run_contract.manifest_row_schedule
        if not isinstance(schedule, tuple) or not schedule:
            raise ValueError("manifest_row_schedule must be a non-empty tuple")
        for index, row_id in enumerate(schedule):
            canonical_row_id = _nonblank(
                row_id, f"manifest_row_schedule[{index}]"
            )
            if canonical_row_id != row_id or "|" in row_id:
                raise ValueError("manifest row ID is not canonical")
        if "|" in view:
            raise ValueError("view_id cannot contain the '|' delimiter")
        schedule_sha = hashlib.sha256(
            _canonical_json(list(schedule))
        ).hexdigest()
        if schedule_sha != _sha256(
            run_contract.manifest_row_schedule_sha256,
            "manifest row schedule",
        ):
            raise ValueError("manifest row schedule SHA-256 mismatch")
        if run_contract.keyed_draw_schema_sha256 != KEYED_DRAW_SCHEMA_SHA256:
            raise ValueError("keyed draw schema SHA-256 mismatch")
        runtime_schema_sha = _keyed_draw_schema_sha256(KEYED_DRAW_SCHEMA)
        if runtime_schema_sha != KEYED_DRAW_SCHEMA_SHA256:
            raise ValueError("keyed draw schema runtime drift detected")
        payload = _run_contract_payload(
            training_seed=training,
            partition_seed=partition,
            outer_fold=fold,
            phase=phase,
            frozen_exposures_per_epoch=stride,
            view_id=view,
            world_size=replicas,
            manifest_row_schedule_length=len(schedule),
            manifest_row_schedule_sha256=schedule_sha,
            keyed_draw_schema_sha256=KEYED_DRAW_SCHEMA_SHA256,
        )
        expected_sha = hashlib.sha256(_canonical_json(payload)).hexdigest()
        if expected_sha != _sha256(run_contract.sha256, "run contract"):
            raise ValueError("run contract SHA-256 mismatch")
        valid = True
        error = ""
    except (AttributeError, TypeError, ValueError) as exc:
        error = str(exc)
    return _ExposureRunContractSeal(
        valid=valid,
        error=error,
        manifest_row_schedule=run_contract.manifest_row_schedule,
        scalar_snapshot=_run_contract_scalar_snapshot(run_contract),
    )


def _run_contract_scalar_snapshot(
    run_contract: ExposureRunContract,
) -> tuple[Any, ...]:
    return (
        run_contract.format_version,
        run_contract.training_seed,
        run_contract.partition_seed,
        run_contract.outer_fold,
        run_contract.phase,
        run_contract.frozen_exposures_per_epoch,
        run_contract.view_id,
        run_contract.world_size,
        run_contract.manifest_row_schedule_sha256,
        run_contract.keyed_draw_schema_sha256,
        run_contract.sha256,
    )


def _validate_run_contract(
    run_contract: ExposureRunContract,
) -> ExposureRunContract:
    if not isinstance(run_contract, ExposureRunContract):
        raise ValueError("run contract must be an ExposureRunContract")
    try:
        seal = run_contract._seal
        runtime_schema_sha = _keyed_draw_schema_sha256(KEYED_DRAW_SCHEMA)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("exposure run contract tamper detected") from exc
    if (
        not isinstance(seal, _ExposureRunContractSeal)
        or not seal.valid
        or run_contract.manifest_row_schedule is not seal.manifest_row_schedule
        or _run_contract_scalar_snapshot(run_contract) != seal.scalar_snapshot
        or runtime_schema_sha != KEYED_DRAW_SCHEMA_SHA256
    ):
        raise ValueError("exposure run contract tamper detected")
    return run_contract


def _validate_ledger(
    ledger: CommittedExposureLedger,
    *,
    expected_run_contract: ExposureRunContract | None = None,
) -> None:
    if not isinstance(ledger, CommittedExposureLedger):
        raise ValueError("ledger must be a CommittedExposureLedger")
    if ledger.format_version != 1:
        raise ValueError("committed exposure ledger format_version must equal 1")
    cursor = _nonnegative_integer(
        ledger.committed_exposure_cursor, "committed_exposure_cursor"
    )
    next_index = _nonnegative_integer(
        ledger.next_committed_exposure_index,
        "next_committed_exposure_index",
    )
    if cursor != next_index:
        raise ValueError("committed exposure cursor/index mismatch")
    if ledger.keyed_draw_schema_sha256 != KEYED_DRAW_SCHEMA_SHA256:
        raise ValueError("keyed draw schema SHA-256 mismatch")
    contract_sha = _sha256(ledger.run_contract_sha256, "run contract")
    if expected_run_contract is not None:
        contract = _validate_run_contract(expected_run_contract)
        if contract_sha != contract.sha256:
            raise ValueError("run contract SHA-256 mismatch")
        if cursor > len(contract.manifest_row_schedule):
            raise ValueError("committed exposure cursor exceeds frozen schedule")
    rolling = _sha256(
        ledger.rolling_committed_draw_ledger_sha256,
        "rolling committed draw ledger",
    )
    if cursor == 0 and rolling != _INITIAL_ROLLING_SHA256:
        raise ValueError("zero-cursor ledger must use the genesis rolling hash")
    if cursor > 0 and rolling == _INITIAL_ROLLING_SHA256:
        raise ValueError("nonzero-cursor ledger cannot use the genesis rolling hash")


def _normalized_hash_draw(key_sha256: str, parameter: str) -> float:
    digest = hashlib.sha256(f"{key_sha256}|{parameter}".encode("utf-8")).digest()
    integer53 = int.from_bytes(digest[:8], "big", signed=False) >> 11
    return integer53 / float(1 << 53)


def _keyed_draw_schema_sha256(schema: Any) -> str:
    if not isinstance(schema, tuple):
        raise ValueError("keyed draw schema must be a tuple")
    if any(
        not isinstance(entry, tuple) or len(entry) != 2 or not isinstance(entry[0], str)
        for entry in schema
    ):
        raise ValueError("keyed draw schema entries are invalid")
    keys = tuple(entry[0] for entry in schema)
    if len(schema) != 4 or len(set(keys)) != len(keys):
        raise ValueError("keyed draw schema fields must be unique")
    payload = dict(schema)
    if tuple(payload) != (
        "format_version",
        "namespace",
        "seed_encoding",
        "normalized_parameters",
    ):
        raise ValueError("keyed draw schema fields are invalid")
    if not isinstance(payload["normalized_parameters"], tuple):
        raise ValueError("keyed draw parameter names must be a tuple")
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _draw_payload(
    *,
    schema_sha256: str,
    key_sha256: str,
    seed_u64: int,
    normalized_parameters: Sequence[tuple[str, float]],
) -> dict[str, Any]:
    return {
        "schema_sha256": schema_sha256,
        "key_sha256": key_sha256,
        "seed_u64": seed_u64,
        "normalized_parameters": [list(item) for item in normalized_parameters],
    }


def _build_exposure_record(
    run_contract: ExposureRunContract, global_index: int
) -> ExposureRecord:
    index = _nonnegative_integer(global_index, "global exposure index")
    if index >= len(run_contract.manifest_row_schedule):
        raise ValueError("manifest row schedule does not cover requested exposure")
    epoch_value, position = divmod(
        index, run_contract.frozen_exposures_per_epoch
    )
    row_id = run_contract.manifest_row_schedule[index]
    key = augmentation_key(
        row_id,
        training_seed=run_contract.training_seed,
        partition_seed=run_contract.partition_seed,
        outer_fold=run_contract.outer_fold,
        phase=run_contract.phase,
        epoch=epoch_value,
        exposure_index=index,
        view_id=run_contract.view_id,
    )
    draw = augmentation_draw(key)
    record = ExposureRecord(
        manifest_row_id=row_id,
        epoch=epoch_value,
        sampler_position=position,
        exposure_index=index,
        view_id=run_contract.view_id,
        run_contract_sha256=run_contract.sha256,
        augmentation_key=key,
        augmentation_key_sha256=key.key_sha256,
        draw=draw,
        record_sha256="",
    )
    return ExposureRecord(
        manifest_row_id=record.manifest_row_id,
        epoch=record.epoch,
        sampler_position=record.sampler_position,
        exposure_index=record.exposure_index,
        view_id=record.view_id,
        run_contract_sha256=record.run_contract_sha256,
        augmentation_key=record.augmentation_key,
        augmentation_key_sha256=record.augmentation_key_sha256,
        draw=record.draw,
        record_sha256=exposure_record_sha256(record),
    )


def _run_contract_payload(
    *,
    training_seed: int,
    partition_seed: int,
    outer_fold: int,
    phase: str,
    frozen_exposures_per_epoch: int,
    view_id: str,
    world_size: int,
    manifest_row_schedule_length: int,
    manifest_row_schedule_sha256: str,
    keyed_draw_schema_sha256: str,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "training_seed": training_seed,
        "partition_seed": partition_seed,
        "outer_fold": outer_fold,
        "phase": phase,
        "frozen_exposures_per_epoch": frozen_exposures_per_epoch,
        "view_id": view_id,
        "world_size": world_size,
        "manifest_row_schedule_length": manifest_row_schedule_length,
        "manifest_row_schedule_sha256": manifest_row_schedule_sha256,
        "keyed_draw_schema_sha256": keyed_draw_schema_sha256,
    }


def _record_payload(record: ExposureRecord) -> dict[str, Any]:
    key = record.augmentation_key
    return {
        "manifest_row_id": record.manifest_row_id,
        "epoch": record.epoch,
        "sampler_position": record.sampler_position,
        "exposure_index": record.exposure_index,
        "view_id": record.view_id,
        "run_contract_sha256": record.run_contract_sha256,
        "augmentation_key": {
            "training_seed": key.training_seed,
            "partition_seed": key.partition_seed,
            "outer_fold": key.outer_fold,
            "phase": key.phase,
            "manifest_row_id": key.manifest_row_id,
            "epoch": key.epoch,
            "exposure_index": key.exposure_index,
            "view_id": key.view_id,
            "seed_u64": key.seed_u64,
            "key_sha256": key.key_sha256,
        },
        "augmentation_key_sha256": record.augmentation_key_sha256,
        "draw_sha256": record.draw.draw_sha256,
    }


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256(value: Any, field: str) -> str:
    text = _nonblank(value, field)
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} SHA-256 is invalid")
    return text


def _positive_integer(value: Any, field: str) -> int:
    integer = _strict_integer(value, field)
    if integer <= 0:
        raise ValueError(f"{field} must be positive")
    return integer


def _nonnegative_integer(value: Any, field: str) -> int:
    integer = _strict_integer(value, field)
    if integer < 0:
        raise ValueError(f"{field} must be non-negative")
    return integer


def _strict_integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field} must be an integer")
    if str(value).strip() not in {str(integer), f"{integer}.0"}:
        raise ValueError(f"{field} must be an integer")
    return integer


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-blank")
    return value.strip()


def _supervised_phase(value: Any) -> str:
    phase = _nonblank(value, "phase")
    if phase not in {"inner_selection", "outer_refit"}:
        raise ValueError("phase must be inner_selection or outer_refit")
    if "|" in phase:
        raise ValueError("phase cannot contain the '|' delimiter")
    return phase


__all__ = [
    "AugmentationDraw",
    "AugmentationKey",
    "CommittedExposureLedger",
    "ExposureRecord",
    "ExposureRunContract",
    "KEYED_DRAW_SCHEMA",
    "KEYED_DRAW_SCHEMA_SHA256",
    "KEYED_DRAW_PARAMETER_NAMES",
    "augmentation_draw",
    "augmentation_key",
    "commit_exposures",
    "exposure_index",
    "exposure_record_sha256",
    "freeze_exposure_run_contract",
    "issue_exposures",
    "restore_committed_exposure_ledger",
]
