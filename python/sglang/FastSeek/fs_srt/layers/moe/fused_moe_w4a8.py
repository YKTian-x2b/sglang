
import logging
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)



def fused_experts_w4a8(hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    use_w4a8: bool = True,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,):

    assert use_w4a8 == True, "use_w4a8 must be True"
    torch.ops.sglang.fused_experts_w4a8(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
    )
    return hidden_states