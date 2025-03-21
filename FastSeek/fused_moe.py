import torch



class FusedMoE(torch.nn.Module):
    def __init__():
        super().__init__()
    
    
    def forward():
        
        Fp8MoEMethod.apply(
            
        )
        
        if self.reduce_results and self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states