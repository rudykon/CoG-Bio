# Verification of the cleaned source

The release was checked without modifying the original research implementation,
checkpoints, annotations or saved predictions.

| Check | Result |
|---|---|
| Standalone synthetic/regression suite | 96 passed; 1 CUDA-only test skipped |
| Utilities, coordinate transforms, metrics, view fusion and geometry | 98.58% combined line coverage; 90% gate passed |
| Dataset, image loading and transform composition | 91.77% combined line coverage; 80% gate passed |
| Synthetic training | Two optimizer steps, finite loss, learned head updated, frozen base unchanged, checkpoint saved |
| Label-free inference wrapper | Synthetic nine-task images produce four view files and valid final JSON |
| Recorded MRP and IAP checkpoints | CPU forward at 64 × 64; every output head exactly matches the original implementation |
| Saved four-view prediction replay | All 1,281 final endpoint records match; maximum coordinate difference 0 px |
| Recomputed nine-task MRE | 23.6658696619782 px, rounded to 23.67 px |

The full GPU training and 512 × 512 GPU inference runs were not repeated during
source preparation. The CPU model check uses a small synthetic image to test
architecture, weights and forward semantics; the full-size saved prediction
replay separately checks the final fusion and correction path.

Run `make test` or `make coverage` to repeat the portable checks. Synthetic
fixtures need no patient data or external model weights. Coverage gates apply
to the listed numerical/data modules; model behavior and the training/inference
wrappers additionally receive smoke tests. Large external encoder internals and
CUDA-only execution are outside the CPU coverage gate.
