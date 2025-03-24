import torch.nn.functional as F
import torch

from sglang.srt.layers.activation import SiluAndMul



def torch_run(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]

def sglang_run(x: torch.Tensor) -> torch.Tensor:
    return SiluAndMul()(x)


if __name__ == "__main__":
    bs = 8
    mid_dim = 128
    x = torch.randn(bs, mid_dim * 2, dtype=torch.bfloat16).cuda()
    torch_res = torch_run(x)
    sgl_res = sglang_run(x)
    torch.allclose(torch_res, sgl_res, 1e-3, 1e-5)
    print(torch_res.shape)
    print(sgl_res.shape)