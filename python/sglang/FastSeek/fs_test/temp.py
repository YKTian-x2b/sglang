import torch 
import importlib
import pkgutil
from transformers import AutoTokenizer

# package_name = "sglang.srt.models"
# package = importlib.import_module(package_name)
# for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
#     print(name)


# module = importlib.import_module("sglang.FastSeek.deepseek_v3")
# entry = module.EntryClass
# print(entry)

# from sglang.FastSeek.fs_configs.personal.model_config.config_dsv3 import DeepseekV3Config
# hf_config = DeepseekV3Config.from_pretrained("/home/private_xgpt_repo/sglang/python/sglang/FastSeek/fs_configs/personal/model_config/dsv3_config_tiny.json")
# print(hf_config)


# tokenizer = AutoTokenizer.from_pretrained(
#         "/home/private_xgpt_repo/sglang/python/sglang/FastSeek/fs_configs/offical/tokenizer",
#     )
# print(len(tokenizer.encode("小炒肉怎么做")))

import numpy as np
batch_size = 1
input_len = 5
input_ids = np.ones((batch_size, input_len), dtype=np.int32)
print(input_ids)
print(list(input_ids))