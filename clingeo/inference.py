"""Label-free image inference with externally supplied pretrained/adaptation weights."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data.task_schema import load_landmark_schema
from .images import OfficialImages, original_coordinates
from .io import image_rows, read_json, write_json
from .model import sha256_file
from .precision import PreciseFoundationBiometry
from .views import FixedViewMean
from .pipeline import combine_views


def load_predictor(checkpoint, encoder, schema, declaration, device, verify_checksum=False):
    if verify_checksum and sha256_file(checkpoint) != declaration["sha256"]:
        raise ValueError("checkpoint differs from the recorded paper artifact")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = dict(payload["model_config"])
    if config["lora_rank"] != declaration["rank"] or config["input_size"] != 512:
        raise ValueError("checkpoint rank/input resolution differs from the paper recipe")
    if config["candidate"] != "dino_lora":
        raise ValueError("DINOv3 LoRA checkpoint required")
    config["weight_dir"] = str(encoder)
    model = PreciseFoundationBiometry(schema, config).to(device).eval()
    model.load_research_state_dict(payload["state_dict"])
    return model


def predict(manifest, image_root, encoder, weights, output, schema_path, config_path,
            prior_path, device="cuda:0", workers=None, batch_size=None, verify_checksum=False):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu for a CPU check")
    torch.set_num_threads(4)
    schema = load_landmark_schema(schema_path)
    config, prior = read_json(config_path), read_json(prior_path)
    if [m["name"] for m in config["models"]] != ["MRP", "IAP"] or config["views"] != ["flip_mean", "crop_mean"]:
        raise ValueError("recipe must preserve MRP/IAP and flip/crop ordering")
    rows = image_rows(manifest, schema, image_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    workers = config["workers"] if workers is None else workers
    batch_size = config["batch_size"] if batch_size is None else batch_size
    views = []
    for declaration in config["models"]:
        model = load_predictor(Path(weights) / declaration["checkpoint"], encoder, schema,
                               declaration, device, verify_checksum)
        for kind in config["views"]:
            wrapped = FixedViewMean(model, schema, kind, **config["crop"])
            kwargs = {"multiprocessing_context": "spawn"} if workers else {}
            loader = DataLoader(OfficialImages(image_root, rows, schema, model.input_size),
                                batch_size=batch_size, num_workers=workers,
                                pin_memory=device.type == "cuda", **kwargs)
            records = []
            with torch.no_grad():
                for batch in loader:
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        coords = wrapped(batch["image"].to(device), batch["task_id"].to(device))["coords"].float().cpu().numpy()
                    for i, index in enumerate(batch["index"].tolist()):
                        row = rows[index]
                        if (int(batch["width"][i]), int(batch["height"][i])) != (int(row["width"]), int(row["height"])):
                            raise ValueError("manifest image dimensions differ from image pixels")
                        records.append({"image_path": row["sample_id"], "task_id": row["task_id"],
                            "predicted_points_pixels": original_coordinates(coords[i], batch["inverse"][i].numpy(),
                                int(batch["width"][i]), int(batch["height"][i]), int(row["num_landmarks"]))})
            write_json(output / f'{declaration["name"].lower()}_{kind}.json', records)
            views.append(records)
        del wrapped, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    predictions, receipt = combine_views(views, rows, schema, config, prior)
    write_json(output / "predictions.json", predictions)
    write_json(output / "inference.json", {**receipt, "labels_read": False, "device": str(device),
                                           "paper_checkpoint_checksums_verified": verify_checksum})
    return predictions
