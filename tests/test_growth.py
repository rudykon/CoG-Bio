import pytest
import torch
from clingeo.training.growth import grow_lora_state


def test_rank_growth_preserves_effective_update_and_learns_new_columns():
    generator=torch.Generator().manual_seed(7)
    source={"attn.qkv.lora_a.weight":torch.randn(2,5,generator=generator),
            "attn.qkv.lora_b.weight":torch.randn(7,2,generator=generator),"head.weight":torch.ones(3,3)}
    template={"attn.qkv.lora_a.weight":torch.randn(4,5,generator=generator),
              "attn.qkv.lora_b.weight":torch.randn(7,4,generator=generator),"head.weight":torch.zeros(3,3)}
    result=grow_lora_state(source,template)
    a=result["attn.qkv.lora_a.weight"].requires_grad_();b=result["attn.qkv.lora_b.weight"].requires_grad_()
    expected=source["attn.qkv.lora_b.weight"]@source["attn.qkv.lora_a.weight"]
    assert torch.allclose(b@a,expected,atol=1e-6)
    assert torch.equal(a[2:],template["attn.qkv.lora_a.weight"][2:])
    assert torch.equal(result["head.weight"],source["head.weight"])
    (b@a).square().sum().backward()
    assert b.grad[:,2:].abs().sum()>0


def test_growth_rejects_unrelated_architecture_changes():
    with pytest.raises(ValueError,match="shape"):
        grow_lora_state({"head":torch.ones(2,3)},{"head":torch.ones(4,3)})
    with pytest.raises(ValueError,match="keys"):
        grow_lora_state({"head":torch.ones(2,3)}, {})
