import torch

tensor1 = torch.rand(2, 1, 3, 3)
tensor2 = torch.rand(2, 1, 3, 3)


print(tensor1)
print(tensor2)
print(tensor1 - tensor2)