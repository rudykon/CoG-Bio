# Reproduction details

The final inference recipe is in `configs/inference.json`. MRP has LoRA rank 40;
IAP has rank 32. Both receive a 512 × 512 letterboxed image. The original schema
order, endpoint semantics and task embeddings must be retained when loading
checkpoints. Image features use the normalized grayscale channel replicated
three times, matching the paper artifacts.

MRP is built from a rank-8 parent and a grown rank-32 parent. Their effective
residuals are averaged by concatenating the A factors and multiplying the
concatenated B factors by 0.5. Each original adapter uses alpha/rank = 2;
this scale remains 2 after merging. Shared floating-point head parameters are
averaged. Additional B columns are zero when growing an adapter.

| Stage | Seed | Optimizer steps | LR schedule length | Selected state |
|---|---:|---:|---:|---|
| MRP parent warmup | 42 | 313 | constant | raw |
| MRP parent continuation | 42 | 8,980 | 9,370 | EMA |
| MRP spatial-layout refinement | 44 | 1,953 | 4,685 | raw |
| MRP MRE refinement | 45 | 1,406 | 2,343 | EMA |
| IAP independent training | 46 | 8,199 | 9,370 | EMA |

The optimizer-step count and cosine-schedule length are intentionally distinct.
The final lengths transfer the selected development lengths to the larger
training partition. Do not extend each stage to the schedule endpoint.
The lightweight runner supports complete fixed-stage runs; it does not implement
the original research workspace's interrupted-run resume system.

The two view policies independently evaluate the native image. Horizontal flips
are enabled only for tasks permitted by the schema, with endpoint permutations
undone before averaging. The crop has minimum side fraction 0.8 and context 1.25,
and uses predicted points only. Preserve `align_corners=True` and the original
inverse coordinate transform. Original-coordinate view predictions and model
means are rounded to four decimal places at the same boundaries as the paper
implementation.

Consistency weighting uses a fixed 2.437224555989938 px floor. Weights are bounded
to [0.25, 0.75], with one weight per image shared across all endpoints. The final
recipe does not recalibrate this floor on evaluation images. Cardiac shape
repair precedes HC/FA projection. Auxiliary alternatives retained in the modules
are not selected by the final configuration.

The included shape prior contains only aggregate training statistics. To train
on a different partition, fit a new prior using that partition's training labels.
Development checkpoint selection must use development-only training and held-out
identities. Final training does not perform validation-based selection.

The cleaned numerical components were compared with the original implementation,
and saved four-view predictions reproduced all 1,281 final endpoint records
exactly. This verifies code extraction and postprocessing. It does not substitute
for rerunning the complete GPU training process from external data and weights.
