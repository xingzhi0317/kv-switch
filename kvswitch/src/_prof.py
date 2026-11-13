import time, torch
from ks import *
B = Model("Qwen3-1.7B")
c = B.new_cache(4200)
ids = torch.randint(100, 5000, (1, 4096), device="cuda")
B.forward(ids, c)
x = torch.tensor([[100]], device="cuda")
for _ in range(5): B.forward(x, c, n0=4096)
torch.cuda.synchronize(); t = time.time()
for _ in range(20): B.forward(x, c, n0=4096)
torch.cuda.synchronize(); print("decode ms", (time.time() - t) / 20 * 1000)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for _ in range(3): B.forward(x, c, n0=4096)
    torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cpu_time_total", row_limit=25))
