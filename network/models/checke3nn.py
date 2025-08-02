import torch, e3nn
from e3nn.o3 import Irreps

print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("e3nn:", e3nn.__version__)

# simple irreps round-trip
ir = Irreps("1e + 1o")
print("Irreps OK:", ir)

import sys
print("Python executable:", sys.executable)
