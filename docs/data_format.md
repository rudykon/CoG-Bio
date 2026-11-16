# Data interfaces

Use UTF-8 CSV. JSON arrays inside CSV cells must follow standard CSV quoting.
Coordinates are `(x, y)` in original pixels and follow the task's endpoint order.

## Label-free inference manifest

| Column | Meaning |
|---|---|
| `sample_id` | Unique non-identifying key, retained in output as `image_path` |
| `task_id` | `A4C`, `AOP`, `FA`, `FUGC`, `HC`, `IVC`, `PLAX`, `PSAX`, `fetal_femur` |
| `image_relpath` | Relative path inside the supplied image root |
| `width`, `height` | Actual original image dimensions |
| `num_landmarks` | Endpoint count from the task schema |
| `checksum` | Optional SHA-256 of image bytes |

Additional columns are ignored by inference. No landmark or measurement label
is needed. Output is a JSON list, with each record containing `image_path`,
`task_id` and flattened `predicted_points_pixels: [x1, y1, x2, y2, ...]`.
Every input image must have exactly one output record.

## Evaluation labels

Required columns are `sample_id`, `task_id`, `landmarks` and `valid_mask`.
`landmarks` is a JSON array of `K` pairs; `valid_mask` is a JSON array of `K`
zero/one values. Each image needs at least one valid annotation. The default
metric requires all nine tasks. `--allow-subset` is for subset diagnostics only.

## Training manifest

Required columns are `domain`, `case_id`, `image_path`, `manifest_row_id`,
`hidden_identity_group`, `width`, `height`, `landmarks` and `valid_mask`.
`domain` uses the dataset task names above. `case_id` and `manifest_row_id` must
be deidentified. `image_path` should be relative to the supplied image root.
`hidden_identity_group` is the inherited field name for a resolved patient or
duplicate-identity component; it does not denote hidden-test labels.

Recommended columns are `split=train_labeled`, `source_scope=labeled` and
`checksum`. Only released training labels enter final fitting. Resolve patient
and duplicate relationships before splitting; the trainer checks the supplied
identity groups but cannot infer missing patient relationships from filenames.
Keep split manifests ordered to preserve exposure sampling.

The paper's final training partition has 6,629 images and 6,246 identity groups.
The development evaluation partition has 2,384 images and the released benchmark
has 1,281. Individual inventories are external assets, not part of this source
distribution. Tests construct synthetic images and labels in temporary folders.

The dataset and pretrained encoder remain subject to their original access and
license terms; see `THIRD_PARTY.md`. The source package does not grant data access.
