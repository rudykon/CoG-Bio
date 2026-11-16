# External components

The encoder is DINOv3, loaded through the pinned `timm` conversion. DINOv3 weights
are external assets governed by their model license; they are not redistributed.
The model identifier and immutable revision are recorded in `clingeo/model.py`.

The implementation depends on PyTorch, torchvision, timm, safetensors, NumPy,
SciPy, pandas, Pillow and PyYAML. Install these packages under their own licenses.
No third-party package source or binary is bundled here.

The ultrasound data originate from the MICCAI 2026 Foundation Model for Ultrasound
Biometry challenge. The source package does not redistribute the dataset or
change the organizers' data-use terms. Obtain the data through the organizers.

The conditional Gaussian shape prior was independently implemented, with the
Kev-Bio postprocessing description as a methodological reference. The publication
reference is retained in the aggregate prior metadata. No Kev-Bio code or neural
weights are included.

No separate open-source license is granted by this repository. Third-party
components retain their own licenses.
