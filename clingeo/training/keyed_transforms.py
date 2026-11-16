from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from clingeo.data.transforms import (
    GeometryAugmentConfig,
    affine_tensor_from_normalized,
    apply_hflip_tensor,
    ultrasound_physics_from_normalized,
)
from clingeo.training.keyed_augmentation import (
    KEYED_DRAW_PARAMETER_NAMES,
    ExposureRecord,
    augmentation_draw,
    exposure_record_sha256,
)


@dataclass(frozen=True)
class KeyedTransformAudit:
    exposure_index: int
    record_sha256: str
    draw_sha256: str
    hflip_applied: bool


def apply_keyed_supervised_augmentation(
    image: torch.Tensor,
    coords: np.ndarray,
    *,
    geometry: GeometryAugmentConfig,
    allow_hflip: bool,
    flip_pairs: tuple[tuple[int, int], ...],
    ultrasound_strength: str,
    exposure: ExposureRecord,
) -> tuple[torch.Tensor, np.ndarray, KeyedTransformAudit]:
    if not isinstance(exposure, ExposureRecord):
        raise ValueError("keyed augmentation requires an ExposureRecord")
    if exposure.record_sha256 != exposure_record_sha256(exposure):
        raise ValueError("keyed augmentation exposure record hash mismatch")
    expected_draw = augmentation_draw(exposure.augmentation_key)
    if exposure.draw != expected_draw:
        raise ValueError("keyed augmentation draw hash mismatch")
    if exposure.augmentation_key_sha256 != exposure.augmentation_key.key_sha256:
        raise ValueError("keyed augmentation key hash mismatch")
    raw_parameters = exposure.draw.normalized_parameters
    names = tuple(name for name, _value in raw_parameters)
    if names != KEYED_DRAW_PARAMETER_NAMES or len(set(names)) != len(names):
        raise ValueError("keyed augmentation parameter schema mismatch")
    parameters = {name: value for name, value in raw_parameters}

    augmented_image = image
    augmented_coords = np.asarray(coords, dtype=np.float32).copy()
    flipped = bool(
        allow_hflip
        and geometry.hflip
        and parameters["hflip_gate"] < 0.5
    )
    if flipped:
        augmented_image, augmented_coords = apply_hflip_tensor(
            augmented_image, augmented_coords, flip_pairs
        )
    augmented_image, augmented_coords = affine_tensor_from_normalized(
        augmented_image,
        augmented_coords,
        geometry,
        rotation=parameters["rotation"],
        scale=parameters["scale"],
        translate_x=parameters["translate_x"],
        translate_y=parameters["translate_y"],
    )
    augmented_image = ultrasound_physics_from_normalized(
        augmented_image,
        strength=ultrasound_strength,
        speckle_gate=parameters["speckle_gate"],
        speckle_variance=parameters["speckle_variance"],
        gamma_gate=parameters["gamma_gate"],
        gamma=parameters["gamma"],
        gain_gate=parameters["gain_gate"],
        gain=parameters["gain"],
        bias=parameters["bias"],
        shadow_gate=parameters["shadow_gate"],
        shadow_x=parameters["shadow_x"],
        shadow_width=parameters["shadow_width"],
        shadow_attenuation=parameters["shadow_attenuation"],
        depth_gate=parameters["depth_gate"],
        depth_alpha=parameters["depth_alpha"],
        noise_seed=exposure.augmentation_key.seed_u64,
    )
    return (
        augmented_image,
        augmented_coords,
        KeyedTransformAudit(
            exposure_index=exposure.exposure_index,
            record_sha256=exposure.record_sha256,
            draw_sha256=exposure.draw.draw_sha256,
            hflip_applied=flipped,
        ),
    )


__all__ = ["KeyedTransformAudit", "apply_keyed_supervised_augmentation"]
