import torch
import time

x = torch.randn(4096, 4096, device="cuda")
for i in range(500):
    x = torch.matmul(x, x)
    torch.cuda.synchronize()  # Force GPU to complete ops before next loop