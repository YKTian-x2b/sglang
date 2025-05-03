
import logging
from typing import Any, Callable, Dict, List, Optional
import torch
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.FastSeek.fs_srt.layers.moe.fused_moe_w4a8 import fused_experts_w4a8

class W4A8Config(QuantizationConfig):
    def __init__(self,
        activation_scheme: str = "dynamic",
        weight_block_size: List[int] = None,):
        self.weight_block_size = weight_block_size

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "W4A8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        return cls(
            activation_scheme=activation_scheme,
            weight_block_size=weight_block_size,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        # FIXME
        # assert isinstance(layer, EPMoE_W4A8)
        # return W4A8MoEMethod(self)
        raise NotImplementedError


    def get_scaled_act_names(self) -> List[str]:
        return []


# class W4A8MoEMethod():
#     def apply():
#         from sglang.srt.layers.moe.topk import select_experts
#         # Expert selection
#         topk_weights, topk_ids = select_experts(
#             hidden_states=x,
#             router_logits=router_logits,
#             use_grouped_topk=use_grouped_topk,
#             top_k=top_k,
#             renormalize=renormalize,
#             topk_group=topk_group,
#             num_expert_group=num_expert_group,
#             custom_routing_function=custom_routing_function,
#             correction_bias=correction_bias,
#         )
        
#         fused_experts_w4a8()