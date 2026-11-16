from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any


_FLAT_DOMAIN_GENERALIZATION_KEYS = {"enabled", "layers", "mixstyle_p", "dsu_p", "alpha"}
_TOP_LEVEL_KEYS = {"mixstyle", "dsu"}
_NOOP_DOMAIN_GENERALIZATION = {
    "mixstyle": {"enabled": False, "p": 0.0, "alpha": 0.1, "layers": []},
    "dsu": {"enabled": False, "p": 0.0, "layers": []},
}


def normalize_domain_generalization_config(
    cfg: Mapping[str, Any] | None,
    *,
    strict_clean: bool,
) -> dict[str, dict[str, Any]]:
    """Normalize canonical nested MixStyle/DSU config to internal layer indices."""

    if cfg is None:
        return _noop_domain_generalization()
    if not isinstance(cfg, Mapping):
        raise ValueError("domain_generalization config must be a mapping")
    cfg = dict(cfg)
    flat_keys = sorted(key for key in cfg if key in _FLAT_DOMAIN_GENERALIZATION_KEYS)
    if flat_keys:
        message = (
            "flat domain_generalization keys are legacy no-ops; use nested replacements "
            "mixstyle: {enabled, p, alpha, layers} and dsu: {enabled, p, layers}"
        )
        if strict_clean:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)
        return _noop_domain_generalization()
    unknown = sorted(set(cfg) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"unknown domain_generalization key(s): {', '.join(unknown)}")
    return {
        "mixstyle": _normalize_mixstyle(cfg.get("mixstyle", {}) or {}),
        "dsu": _normalize_dsu(cfg.get("dsu", {}) or {}),
    }


def _noop_domain_generalization() -> dict[str, dict[str, Any]]:
    return {
        "mixstyle": dict(_NOOP_DOMAIN_GENERALIZATION["mixstyle"]),
        "dsu": dict(_NOOP_DOMAIN_GENERALIZATION["dsu"]),
    }


def _normalize_mixstyle(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("mixstyle config must be a mapping")
    _reject_unknown_keys(raw, {"enabled", "p", "alpha", "layers"}, "mixstyle")
    enabled = bool(raw.get("enabled", False))
    p = _probability(raw.get("p", 0.0), "mixstyle.p")
    alpha = float(raw.get("alpha", 0.1))
    if alpha <= 0.0:
        raise ValueError("mixstyle.alpha must be positive")
    layers = _normalize_layers(raw.get("layers", []))
    if enabled and not layers:
        raise ValueError("mixstyle.layers must be non-empty when enabled")
    return {"enabled": enabled, "p": p, "alpha": alpha, "layers": layers}


def _normalize_dsu(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("dsu config must be a mapping")
    _reject_unknown_keys(raw, {"enabled", "p", "layers"}, "dsu")
    enabled = bool(raw.get("enabled", False))
    p = _probability(raw.get("p", 0.0), "dsu.p")
    layers = _normalize_layers(raw.get("layers", []))
    if enabled and not layers:
        raise ValueError("dsu.layers must be non-empty when enabled")
    return {"enabled": enabled, "p": p, "layers": layers}


def _reject_unknown_keys(raw: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} key(s): {', '.join(unknown)}")


def _probability(value: Any, name: str) -> float:
    p = float(value)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return p


def _normalize_layers(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("layers must be a non-empty sequence of 1-based stage numbers")
    layers: list[int] = []
    seen: set[int] = set()
    for item in raw:
        index = _stage_to_index(item)
        if index not in seen:
            layers.append(index)
            seen.add(index)
    return layers


def _stage_to_index(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("layers must contain 1-based stage numbers, not booleans")
    if isinstance(value, int):
        stage = value
    else:
        text = str(value).lower().replace("stage", "").strip()
        try:
            stage = int(text)
        except ValueError as exc:
            raise ValueError(f"invalid stage layer {value!r}") from exc
    if stage <= 0:
        raise ValueError("layers must be 1-based positive stage numbers")
    return stage - 1
