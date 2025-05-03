import torch
import logging
from typing import Callable, List, Optional, Tuple

from sglang.srt.utils import set_weight_attrs
from sglang.FastSeek.fs_srt.layers.quantization.w4a8 import W4A8Config

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.custom_op import scaled_fp8_quant as sgl_scaled_fp8_quant

logger = logging.getLogger(__name__)


class EPMoE_W4A8(torch.nn.Module):
    def __init__(self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        activation: str = "silu",):
        super().__init__()
        
        # FIXME: params_dtype
        assert params_dtype == torch.int8
        self.tp_size = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = get_tensor_model_parallel_rank()

        self.quant_method: Optional[QuantizeMethodBase] = W4A8EPMoEMethod(
                quant_config
            )
        self.use_w4a8 = True

    
    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            use_grouped_topk=self.use_grouped_topk,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            correction_bias=self.correction_bias,
            custom_routing_function=self.custom_routing_function,
        )
        

        # gateup_output = grouped_gemm_runner()
        # silu_and_mul()
        # down_output = grouped_gemm_runner() 
        # return down_output
        
        # FIXME
        return hidden_states
    
        
    @classmethod
    def make_expert_params_mapping(cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,):
        mapping_res = []
        for shard_id, weight_name in [("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),]:
            for expert_id in range(num_experts):
                mapping_res.append(((
                    "experts.w13_"
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                    else "experts.w2_"
                ), f"experts.{expert_id}.{weight_name}.", expert_id, shard_id))
    
    def weight_loader(self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,):

        if "scale" in weight_name:
            print(f"scale in weight_name:{weight_name}")
        
        if shard_id == "w2":
            param.data[expert_id] = loaded_weight
        elif shard_id == "w1":
            param.data[expert_id][: self.intermediate_size, :] = loaded_weight
        elif shard_id == "w3":
            param.data[expert_id][self.intermediate_size :, :] = loaded_weight
        else:
            raise ValueError(f"Expected shard_id w1,w2 or w3 but got {shard_id}")
        
    

class W4A8EPMoEMethod():
    def __init__(self, quant_config: W4A8Config):
        pass
    
    def create_weights(self,
        layer: torch.nn.Module,
        num_experts_per_partition: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,):
        # FIXME: should be int4
        params_dtype = torch.int8 

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                2 * intermediate_size,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                hidden_size,
                intermediate_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        
        # WEIGHT_SCALES
        
        # INPUT_SCALES
        
    
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    # 这里可能用不到，直接走forward
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:
        raise NotImplementedError