import torch
import os
from safetensors import safe_open


def load_keys(in_pth, out_pth):
    wf = open(out_pth, "a")
    model = torch.load(in_pth, weights_only=True)
    with open(out_pth, "w") as wf:
        wf.write('[')
        for key, value in model.items():
            wf.write(f"'{key}', ")
        wf.write(']')


def load_pt(in_pth, out_pth):
    wf = open(out_pth, "a")
    model = torch.load(in_pth, weights_only=True)
    for key, value in model.items():
        print(f"Key: {key}\nShape: {value.shape}\n", file=wf)


def load_safetensor(in_pth, out_pth):
    rf = open(out_pth, "a")
    with safe_open(in_pth, framework="pt", device="cpu") as f:
        for key in f.keys():
            print(f"key: {key},   ======shape: {f.get_tensor(key).shape}", file=rf)
            

if __name__ == "__main__":
    in_pth = "/home/private_xgpt_repo/sglang/python/sglang/FastSeek/fs_models/r1_w4a8_4layers.pt"
    out_pth = "/home/private_xgpt_repo/sglang/python/sglang/FastSeek/fs_models/dsv3_w4a8_state_dict.txt"
    # out_pth = "/home/private_xgpt_repo/sglang/python/sglang/FastSeek/fs_models/dsv3_fp8_keys.txt"
    load_pt(in_pth, out_pth)
    # load_keys(in_pth, out_pth)