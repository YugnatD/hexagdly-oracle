"""Session-wide test setup.

Disables TF32 for the Keras torch backend.

Why: this suite asserts *numerical equivalence* between the Keras hex layers and
the indexed / PyTorch references, with thresholds around 1e-3..1e-4. The Keras
torch backend places tensors on CUDA when a GPU is visible, and PyTorch ships
with ``cudnn.allow_tf32 = True``, so cuDNN runs convolutions in TF32 -- a 10-bit
mantissa, i.e. ~1e-3 relative precision. That is hardware fast-math, not a
difference in what the layers compute, and it makes the equivalence tests
measure the GPU rather than the code.

Measured on an RTX 500 Ada, ``keras.ops.conv`` against a float64 reference
(input (1,28,27,3), dense random kernel (4,7,3,4), |output| ~ 360):

    cudnn.allow_tf32=True   ->  abs err 2.74e-01
    cudnn.allow_tf32=False  ->  abs err 3.21e-04    (CPU float32: 3.21e-04)

The error only appears once the kernel is at least 7 wide, which is where cuDNN
switches to the TF32 path -- and hexagdly kernels are wide by construction,
since dilation is realised by zero insertion. Before this file existed, that
produced 61 failures on the torch backend (45 Conv3d, 16 Conv2d) and none on
tensorflow, which simply had no GPU available and therefore ran in CPU float32.

This is scoped to the test session on purpose: the library itself must not
mutate global torch flags on import.
"""

try:
    import torch
except ImportError:  # torch backend not installed -- nothing to pin
    pass
else:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
