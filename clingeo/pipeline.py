"""The paper's fixed four-view fusion followed by shape repair and projection."""
from .fusion import fuse
from .shape import apply_prior
from .projection import apply_projection
from .io import validate_predictions


def combine_views(views, rows, schema, config, prior):
    if len(views) != 4:
        raise ValueError("MRP flip/crop and IAP flip/crop views are required, in that order")
    for records in views:
        validate_predictions(records, rows, schema)
    base = fuse(views, config["fusion"], config["disagreement_floor_pixels"], "image_path")
    metadata = {r["sample_id"]: {"width": int(r["width"]), "height": int(r["height"])} for r in rows}
    repaired, changed = apply_prior(base, metadata, prior, config["shape_variant"], "image_path")
    final = apply_projection(repaired)
    validate_predictions(final, rows, schema)
    return final, {"cases": len(final), "shape_repaired_cases": changed}
