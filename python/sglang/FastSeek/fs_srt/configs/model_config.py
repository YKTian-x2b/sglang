import json
import math 
import torch
from enum import IntEnum, auto
from typing import List, Optional, Set, Union
from sglang.srt.hf_transformers_utils import get_context_length
from sglang.FastSeek.fs_configs.personal.model_config.config_dsv3 import DeepseekV3Config

class AttentionArch(IntEnum):
    MLA = auto()
    MHA = auto()

class ModelConfig:
    def __init__(self, model_path, model_config_path, context_length):
        self.model_path = model_path
        self.revision = None
        self.quantization = None 
        
        self.hf_config = DeepseekV3Config.from_pretrained(model_config_path)
        self.hf_text_config = self.hf_config
        
        self.is_generation = True
        
        self.dtype = torch.bfloat16 # self.hf_config.get("dtype", "bfloat16")
        derived_context_len = get_context_length(self.hf_config)
        self.context_len = derived_context_len if context_length is None else context_length
        
        self.head_dim = 256
        self.attention_arch = AttentionArch.MLA
        self.kv_lora_rank = self.hf_config.kv_lora_rank
        self.qk_nope_head_dim = self.hf_config.qk_nope_head_dim
        self.qk_rope_head_dim = self.hf_config.qk_rope_head_dim
        self.v_head_dim = self.hf_config.v_head_dim
        
        self.scaling = 1 / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)
        mscale_all_dim = self.hf_config.rope_scaling["mscale_all_dim"]
        scaling_factor = self.hf_config.rope_scaling["factor"]
        mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
        self.scaling = self.scaling * mscale * mscale
        
        self.num_attention_heads = self.hf_config.num_attention_heads
        self.num_key_value_heads = self.hf_config.num_key_value_heads
        self.hidden_size = self.hf_config.hidden_size
        self.num_hidden_layers = self.hf_config.num_hidden_layers
        self.vocab_size = self.hf_config.vocab_size
        
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
        