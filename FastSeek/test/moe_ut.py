import unittest

import torch

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe


class TestFusedMOE(unittest.TestCase):

    @staticmethod
    def create_random_cuda_tensor(shape, dtype, mean=0, std=0.01):
        """Create a random CUDA tensor

        Args:
            shape: Tensor shape
            dtype: Data type
            mean: Mean value
            std: Standard deviation

        Returns:
            torch.Tensor: Randomly initialized CUDA tensor
        """
        return torch.empty(shape, dtype=dtype, device="cuda").normal_(mean, std)

    def get_tolerance(self, dtype):
        """Get tolerance values for different data types

        Args:
            dtype: Data type

        Returns:
            tuple: (relative tolerance, absolute tolerance)
        """
        if dtype == torch.float32:
            return 1e-3, 1e-5
        elif dtype in [torch.float16, torch.bfloat16]:
            return 1e-1, 1e-2
        else:
            return 1e-2, 1e-2  # Default values for other types

    def torch_naive_moe(self, a, w1, w2, score, topk):
        B, D = a.shape
        a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
        out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        topk_weight = topk_weight.view(-1)
        topk_ids = topk_ids.view(-1)
        for i in range(w1.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = SiluAndMul()(a[mask] @ w1[i].transpose(0, 1)) @ w2[
                    i
                ].transpose(0, 1)
        return (
            out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
        ).sum(dim=1)

    def _test_case(self, m, n, k, e, topk, dtype, use_fp8_w8a8=False):
        rtol, atol = self.get_tolerance(dtype)
        
        a = self.create_random_cuda_tensor((m, k), dtype)
        w1 = self.create_random_cuda_tensor((e, 2 * n, k), dtype)
        w2 = self.create_random_cuda_tensor((e, k, n), dtype)
        score = self.create_random_cuda_tensor((m, e), dtype)

        triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False)
        torch_output = self.torch_naive_moe(a, w1, w2, score, topk)
        torch.testing.assert_close(
            triton_output, torch_output, rtol=rtol, atol=atol
        )

    def test_various_configurations(self):
        m_values = 64
        n_values = 128
        k_values = 128
        dtype = torch.bfloat16
        use_fp8_w8a8 = False
        e = 8
        topk = 2

        self._test_case(
            m_values,
            n_values,
            k_values,
            e,
            topk,
            dtype,
            use_fp8_w8a8=use_fp8_w8a8,
        )


if __name__ == "__main__":
    unittest.main()
