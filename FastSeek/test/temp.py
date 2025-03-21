import torch 


hidden_states = torch.randn(2, 3)
print(len(list(hidden_states.shape)) == 2)