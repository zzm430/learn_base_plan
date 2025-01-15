import torch
torch.ones(2,3)
print(torch.ones(2,3))
print(torch.zeros(3,3))
print(torch.rand(3,4))
print(torch.randint(0,10,(2,3)))
print(torch.tensor([1,2],
                   [3,4],
                   [5,6]))
a = torch.tensor([1,2],
                   [3,4],
                   [5,6])
print(torch.rand_like(a,dtype=float))