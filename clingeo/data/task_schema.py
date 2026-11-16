from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


TODO_DERIVATIONS = {"TODO", "TODO_OFFICIAL_FORMULA"}


@dataclass(frozen=True)
class ClinicalParamSchema:
    name: str
    derivation: str
    unit: str = "px"
    todo: str | None = None

    @property
    def is_todo(self) -> bool:
        return self.derivation in TODO_DERIVATIONS or self.derivation.startswith("TODO")


@dataclass(frozen=True)
class LandmarkTaskSchema:
    name: str
    canonical: str
    aliases: tuple[str, ...]
    num_landmarks: int
    allow_hflip: bool
    flip_pairs: tuple[tuple[int, int], ...]
    geometry_type: str
    clinical_params: tuple[ClinicalParamSchema, ...]
    heatmap_sigma: float
    refiner_roi_scale: float = 0.0
    use_pca_shape_prior: bool = False
    postprocess: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.clinical_params]


class LandmarkSchema:
    def __init__(self, tasks: dict[str, LandmarkTaskSchema], method_name: str = "ClinGeo-USBiometry"):
        self.tasks = tasks
        self.method_name = method_name
        self.alias_to_name: dict[str, str] = {}
        for name, task in tasks.items():
            self.alias_to_name[name] = name
            self.alias_to_name[name.lower()] = name
            self.alias_to_name[task.canonical] = task.canonical if task.canonical in tasks else name
            self.alias_to_name[task.canonical.lower()] = task.canonical if task.canonical in tasks else name
            for alias in task.aliases:
                self.alias_to_name[alias] = name
                self.alias_to_name[alias.lower()] = name

    @classmethod
    def from_file(cls, path: str | Path, strict_formulas: bool = False) -> "LandmarkSchema":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        if not isinstance(payload, dict) or "domains" not in payload:
            raise ValueError(f"{path} must contain a top-level 'domains' mapping")
        tasks: dict[str, LandmarkTaskSchema] = {}
        for name, raw in payload["domains"].items():
            if not isinstance(raw, dict):
                raise ValueError(f"Schema entry {name} must be a mapping")
            required = ["num_landmarks", "allow_hflip", "geometry_type", "heatmap_sigma"]
            missing = [k for k in required if k not in raw]
            if missing:
                raise ValueError(f"Schema entry {name} is missing fields: {missing}")
            num_landmarks = int(raw["num_landmarks"])
            if num_landmarks <= 0:
                raise ValueError(f"Schema entry {name} has invalid num_landmarks={num_landmarks}")
            flip_pairs = tuple(tuple(map(int, pair)) for pair in raw.get("flip_pairs", []) or [])
            for a, b in flip_pairs:
                if not (0 <= a < num_landmarks and 0 <= b < num_landmarks):
                    raise ValueError(f"Schema entry {name} has out-of-range flip pair {(a, b)}")
            params: list[ClinicalParamSchema] = []
            for param_name, param_raw in (raw.get("clinical_params") or {}).items():
                if not isinstance(param_raw, dict):
                    raise ValueError(f"Clinical param {name}.{param_name} must be a mapping")
                derivation = str(param_raw.get("derivation", "TODO_OFFICIAL_FORMULA"))
                param = ClinicalParamSchema(
                    name=str(param_name),
                    derivation=derivation,
                    unit=str(param_raw.get("unit", "px")),
                    todo=param_raw.get("todo"),
                )
                if strict_formulas and param.is_todo:
                    raise ValueError(f"Clinical formula for {name}.{param.name} is TODO; refusing strict mode")
                params.append(param)
            aliases = tuple(str(x) for x in raw.get("aliases", []) or [])
            canonical = str(raw.get("canonical", name))
            tasks[name] = LandmarkTaskSchema(
                name=name,
                canonical=canonical,
                aliases=aliases,
                num_landmarks=num_landmarks,
                allow_hflip=bool(raw["allow_hflip"]),
                flip_pairs=flip_pairs,
                geometry_type=str(raw["geometry_type"]),
                clinical_params=tuple(params),
                heatmap_sigma=float(raw["heatmap_sigma"]),
                refiner_roi_scale=float(raw.get("refiner_roi_scale", 0.0)),
                use_pca_shape_prior=bool(raw.get("use_pca_shape_prior", False)),
                postprocess=dict(raw.get("postprocess") or {}),
                raw=raw,
            )
        schema = cls(tasks, method_name=str(payload.get("method_name", "ClinGeo-USBiometry")))
        schema.validate_required_domains()
        return schema

    def validate_required_domains(self) -> None:
        required = {"HC", "AC", "Femur", "AOP", "FUGC", "A4C", "PLAX", "PSAX", "IVC"}
        missing = sorted(required.difference(self.tasks))
        if missing:
            raise ValueError(f"landmark_schema.yaml missing required domains: {missing}")

    def resolve_name(self, domain: str) -> str:
        if domain in self.alias_to_name:
            return self.alias_to_name[domain]
        lowered = domain.lower()
        if lowered in self.alias_to_name:
            return self.alias_to_name[lowered]
        raise KeyError(f"Unknown domain '{domain}'. Known domains: {sorted(self.tasks)}")

    def get(self, domain: str) -> LandmarkTaskSchema:
        return self.tasks[self.resolve_name(domain)]

    @property
    def domain_names(self) -> list[str]:
        return list(self.tasks.keys())

    @property
    def max_landmarks(self) -> int:
        return max(task.num_landmarks for task in self.tasks.values())

    @property
    def max_params(self) -> int:
        return max((len(task.clinical_params) for task in self.tasks.values()), default=0)

    def task_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.tasks)}


def load_landmark_schema(path: str | Path = "configs/landmark_schema.yaml", strict_formulas: bool = False) -> LandmarkSchema:
    return LandmarkSchema.from_file(path, strict_formulas=strict_formulas)
