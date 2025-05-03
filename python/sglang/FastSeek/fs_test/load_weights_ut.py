from typing import Callable, List, Optional, Tuple


def make_expert_params_mapping(
    ckpt_gate_proj_name: str,
    ckpt_down_proj_name: str,
    ckpt_up_proj_name: str,
    num_experts: int,
) -> List[Tuple[str, str, int, str]]:
    return [
        # (param_name, weight_name, expert_id, shard_id)
        (
            (
                "experts.w13_"
                if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                else "experts.w2_"
            ),
            f"experts.{expert_id}.{weight_name}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id, weight_name in [
            ("w1", ckpt_gate_proj_name),
            ("w2", ckpt_down_proj_name),
            ("w3", ckpt_up_proj_name),
        ]
    ]


stacked_params_mapping = [
    # (param_name, shard_name, shard_id)
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]

# Params for weights, fp8 weight scales, fp8 activation scales
# (param_name, weight_name, expert_id, shard_id)
expert_params_mapping = make_expert_params_mapping(
    ckpt_gate_proj_name="gate_proj",
    ckpt_down_proj_name="down_proj",
    ckpt_up_proj_name="up_proj",
    num_experts=61,
)

params_str = open("/home/private_xgpt_repo/sglang/python/sglang/FastSeek/fs_models/dsv3_fp8_params.txt", "r").read()
params_dict = dict(params_str)

weight_keys_str = open("/home/private_xgpt_repo/sglang/python/sglang/FastSeek/fs_models/dsv3_fp8_keys.txt", "r").read()
weight_keys = list()

loaded_w_names = []
for name in weight_keys:
    if "rotary_emb.inv_freq" in name:
        continue
    for param_name, weight_name, shard_id in stacked_params_mapping:
        # Skip non-stacked layers and experts (experts handled below).
        if weight_name not in name:
            continue
        if ("mlp.experts." in name) and name not in params_dict:
            continue
        # gate_proj -> gate_up_proj && up_proj -> gate_up_proj
        name = name.replace(weight_name, param_name)
        if name.endswith(".bias") and name not in params_dict:
            continue
        param = params_dict[name]
        loaded_w_names.append(name)
        break
    else:
        for mapping in expert_params_mapping:
            param_name, weight_name, expert_id, shard_id = mapping
            if weight_name not in name:
                continue
            # gate_up -> w13
            # down -> w2
            name = name.replace(weight_name, param_name)
            param = params_dict[name]
            loaded_w_names.append(name)
            break
        else:
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            loaded_w_names.append(name)

print(f"loaded_w_names: {loaded_w_names}")
