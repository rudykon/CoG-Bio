import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from clingeo.data.task_schema import load_landmark_schema
from clingeo.evaluation import PAPER_TASKS

torch.set_num_threads(2)


@pytest.fixture
def schema():
    return load_landmark_schema(Path(__file__).resolve().parents[1] / "configs/landmark_schema.yaml")


@pytest.fixture
def nine_images(tmp_path, schema):
    images, labels, training = [], [], []
    for i, task in enumerate(sorted(PAPER_TASKS)):
        n = schema.tasks[task].num_landmarks
        name = f"synthetic_{i}.png"
        Image.fromarray(np.arange(64 * 80, dtype=np.uint8).reshape(64, 80)).save(tmp_path / name)
        coordinates = [[20. + j % 4 * 8, 16. + j // 4 * 8] for j in range(n)]
        image = dict(sample_id=name, task_id=task, image_relpath=name, width=80, height=64, num_landmarks=n)
        label = dict(**image, landmarks=json.dumps(coordinates), valid_mask=json.dumps([1] * n))
        images.append(image); labels.append(label)
        training.append(dict(domain=task, case_id=name, image_path=name, width=80, height=64,
            manifest_row_id=name, hidden_identity_group=f"synthetic-group-{i}", split="train_labeled",
            landmarks=label["landmarks"], valid_mask=label["valid_mask"]))
    pd.DataFrame(images).to_csv(tmp_path / "images.csv", index=False)
    pd.DataFrame(labels).to_csv(tmp_path / "labels.csv", index=False)
    pd.DataFrame(training).to_csv(tmp_path / "train.csv", index=False)
    return tmp_path, images, labels, pd.DataFrame(training)
