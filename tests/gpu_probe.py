"""Assert the accelerator is actually reachable from inside the image.

Run on the target host -- this is the one check that cannot be done from a
laptop. `torch.cuda.is_available()` is the correct call for BOTH backends:
the ROCm build of torch keeps the `cuda` namespace as its HIP entry point.
"""

import sys

import torch

available = torch.cuda.is_available()
build = "rocm" if getattr(torch.version, "hip", None) else "cuda"

print(f"torch      {torch.__version__}")
print(f"build      {build}")
print(f"available  {available}")

if not available:
    print("\nNo accelerator visible. Check device wiring:", file=sys.stderr)
    print("  nvidia -> --device nvidia.com/gpu=all (CDI configured on host)", file=sys.stderr)
    print("  amd    -> --device /dev/kfd --device /dev/dri --group-add keep-groups", file=sys.stderr)
    sys.exit(1)

print(f"device     {torch.cuda.get_device_name(0)}")
print(f"capability {torch.cuda.get_device_capability(0)}")
if build == "rocm":
    print(f"gfx        {torch.cuda.get_device_properties(0).gcnArchName}")

# A real allocation and a real matmul -- `is_available()` has lied before.
x = torch.randn(512, 512, device="cuda")
y = (x @ x).sum().item()
assert y == y, "matmul produced NaN"
print(f"matmul     ok ({y:.4f})")
