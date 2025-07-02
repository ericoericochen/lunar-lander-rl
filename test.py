import torch

a = torch.tensor([1, 2])
b = torch.tensor([0.5, 100])

# print(torch.min(a, b))

print(torch.clamp(a, 1.5, 3))
