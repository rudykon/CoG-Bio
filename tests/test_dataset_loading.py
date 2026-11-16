from clingeo.data.dataset import BiometryDataset
import pytest


def test_dataset_loading_synthetic_smoke(tmp_path):
    from helpers import make_synthetic_manifest

    manifest = make_synthetic_manifest(tmp_path)
    ds = BiometryDataset(manifest, data_root=tmp_path, input_size=64, training=True, ultrasound_aug="none", cache_dir=None)
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["landmarks"].shape[-1] == 2
    assert item["heatmap"].ndim == 3


def test_domain_geometry_override_replaces_default_scale_range():
    from clingeo.data.transforms import geometry_config_for_domain

    cfg = geometry_config_for_domain(
        domain="FUGC",
        geometry_type="long_bone",
        allow_hflip=False,
        overrides={"FUGC": {"scale_min": 0.5, "scale_max": 2.5, "rotation_deg": 18.0, "translate": 0.12}},
    )
    assert cfg.scale_min == 0.5
    assert cfg.scale_max == 2.5
    assert cfg.rotation_deg == 18.0
    assert cfg.translate == 0.12
    assert cfg.hflip is False


def test_dataset_loads_pseudo_sample_weight(tmp_path):
    import pandas as pd

    from helpers import make_synthetic_manifest

    manifest = make_synthetic_manifest(tmp_path)
    df = pd.read_csv(manifest)
    df.loc[0, "split"] = "train_pseudo"
    df.loc[0, "pseudo_weight"] = 0.25
    df.to_csv(manifest, index=False)

    ds = BiometryDataset(manifest, data_root=tmp_path, input_size=64, training=False, cache_dir=None)

    assert float(ds[0]["sample_weight"]) == 0.25
    assert float(ds[1]["sample_weight"]) == 1.0


def test_dataset_passes_identity_fields_through(tmp_path):
    import pandas as pd
    from helpers import make_synthetic_manifest

    manifest = make_synthetic_manifest(tmp_path)
    df = pd.read_csv(manifest)
    df["manifest_row_id"] = [f"row-{i}" for i in range(len(df))]
    df["hidden_identity_group"] = [f"hid-{i}" for i in range(len(df))]
    df["oof_ssl_resolution_state"] = "resolved_anchor"
    df["identity_resolved"] = True
    df["partition_eligible"] = True
    df["ssl_eligible"] = True
    df["source_scope"] = "labeled"
    ds = BiometryDataset(df, data_root=tmp_path, input_size=64, training=False, cache_dir=None)
    item = ds[0]
    assert item["manifest_row_id"] == "row-0"
    assert item["hidden_identity_group"] == "hid-0"
    assert item["oof_ssl_resolution_state"] == "resolved_anchor"
    assert item["identity_resolved"] is True


@pytest.mark.parametrize("column,value", [
    ("source_scope", "public_validation_exclusion"),
    ("oof_ssl_resolution_state", "public_overlap_excluded"),
])
def test_dataset_rejects_public_validation_identity_rows(tmp_path, column, value):
    import pandas as pd
    from helpers import make_synthetic_manifest

    manifest = make_synthetic_manifest(tmp_path)
    df = pd.read_csv(manifest)
    df[column] = value
    with pytest.raises(ValueError, match="public-validation identity row"):
        BiometryDataset(df, data_root=tmp_path, input_size=64, training=False, cache_dir=None)
