import json
import math 
import torch
from typing import List, Optional, Set, Union
from sglang.srt.hf_transformers_utils import get_context_length


class ModelConfig:
    def __init__(self, model_path, context_length):
        self.model_path = model_path
        self.ffn_quant = "W8A8"
        
        self.model_config = None
        with open("/home/private_xgpt_repo/FastSeek/own/dsv3_config_tiny.json", "r") as rf:
            self.model_config = json.load(rf)
        
        self.is_generation = True
        
        self.dtype = torch.bfloat16 # self.model_config.get("dtype", "bfloat16")
        derived_context_len = get_context_length(self.model_config)
        self.context_len = derived_context_len if context_length is None else context_length
        
        self.kv_lora_rank = self.model_config.get("kv_lora_rank", 512)
        self.qk_nope_head_dim = self.model_config.get("qk_nope_head_dim", 128)
        self.qk_rope_head_dim = self.model_config.get("qk_rope_head_dim", 64)
        self.v_head_dim = self.model_config.get("v_head_dim", 128)
        
        self.scaling = 1 / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)
        mscale_all_dim = self.model_config["rope_scaling"]["mscale_all_dim"]
        scaling_factor = self.model_config["rope_scaling"]["factor"]
        mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
        self.scaling = self.scaling * mscale * mscale
        
        self.num_attention_heads = self.model_config["num_heads"]
        self.num_key_value_heads = self.model_config["num_kv_heads"]
        
        self.hf_eos_token_id = None
        
        # useless
        self.is_multimodal = False
        self.is_multimodal_gen = False
        self.is_image_gen = False
        self.is_audio_model = False
        self.is_encoder_decoder = False     

    def __str__(self) -> str:
        return str(self.__dict__)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0     
        