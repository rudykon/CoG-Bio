import torch
import pytest
from clingeo.training.merge import merge_lora_states,merged_model_config


def test_merge_has_exact_mean_effective_lora_update():
    generator=torch.Generator().manual_seed(12)
    first={"qkv.lora_a.weight":torch.randn(2,7,generator=generator),
           "qkv.lora_b.weight":torch.randn(5,2,generator=generator),"head":torch.ones(3)}
    second={"qkv.lora_a.weight":torch.randn(4,7,generator=generator),
            "qkv.lora_b.weight":torch.randn(5,4,generator=generator),"head":torch.ones(3)*3}
    merged=merge_lora_states(first,second)
    expected=.5*(first["qkv.lora_b.weight"]@first["qkv.lora_a.weight"])+.5*(second["qkv.lora_b.weight"]@second["qkv.lora_a.weight"])
    assert torch.allclose(merged["qkv.lora_b.weight"]@merged["qkv.lora_a.weight"],expected,atol=1e-6)
    assert torch.equal(merged["head"],torch.ones(3)*2)
    assert merged_model_config({"input_size":512,"lora_rank":2},{"input_size":512,"lora_rank":4})["lora_rank"]==6
    with pytest.raises(ValueError,match="architecture"):
        merged_model_config({"input_size":512,"lora_rank":2},{"input_size":768,"lora_rank":4})
