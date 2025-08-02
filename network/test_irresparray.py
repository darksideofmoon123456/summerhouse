from e3nn import IrrepsArray
from e3nn.o3 import Irreps

import torch

from e3nn.o3 import Irreps
from e3nn import IrrepsArray
import torch

x = torch.randn(10, 3)
arr = IrrepsArray("1e", x)
print(arr)
