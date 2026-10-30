"""Verify the Python/GPU stack before running EI-Bench. Exit code 0 = ready."""
import sys
import time

ok = True
print(f"python      {sys.version.split()[0]}")
try:
    import numpy, scipy, matplotlib, torch
    print(f"numpy       {numpy.__version__}\nscipy       {scipy.__version__}\nmatplotlib  {matplotlib.__version__}\ntorch       {torch.__version__}")
except ImportError as e:
    print("MISSING PACKAGE:", e); sys.exit(1)

if not torch.cuda.is_available():
    print("CUDA: NOT AVAILABLE -> runs will fall back to CPU (works, ~3-5x slower)")
    sys.exit(2)

i = torch.cuda.current_device()
name = torch.cuda.get_device_name(i)
cap = torch.cuda.get_device_capability(i)
archs = torch.cuda.get_arch_list()
print(f"GPU         {name}  (compute capability {cap[0]}.{cap[1]})")
print(f"torch CUDA  {torch.version.cuda}; compiled archs: {' '.join(archs)}")
need = f"sm_{cap[0]}{cap[1]}"
if need not in archs and f"compute_{cap[0]}{cap[1]}" not in archs:
    print(f"WARNING: this torch build has no {need} kernels; reinstall torch (pip install -U torch)")
    ok = False

x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize(); t = time.time()
for _ in range(20):
    y = x @ x
torch.cuda.synchronize()
tflops = 20 * 2 * 4096 ** 3 / (time.time() - t) / 1e12
print(f"matmul      {tflops:.1f} TFLOP/s (fp32/tf32 default)")
print("READY" if ok else "NOT READY")
sys.exit(0 if ok else 3)
