# pyTorch as numpy2

import torch
from IPython import embed

# Tensors are torch-arrays
x = torch.Tensor([5, 3])
y = torch.Tensor([2, 1])

print(x*y)                                                                      # element-wise multiplication


# 'reshape' is called 'view' in pyTorch

z = torch.rand([2, 5])
print(z)
z = z.view(1, 10)

# 'squeeze()' still work

embed()
