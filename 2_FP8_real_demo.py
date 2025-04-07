import torch

import triton
import triton.language as tl

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

@triton.jit
def fp8_mul_real_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.fma(x, y, 0.0)
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def vecmul(a, b):
    # Check constraints.
    assert a.shape == b.shape, "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    c = torch.empty(a.shape, device=a.device, dtype=torch.float16)
    n_elements = a.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    fp8_mul_real_kernel[grid](a, b, c, n_elements, 256)
    return c

# test fp8_mul_sim_kernel accuracy

import sys
N = int(sys.argv[1])
TORCH_HAS_FP8 = hasattr(torch, "float8_e4m3fn")
if TORCH_HAS_FP8 and is_cuda():
    torch.manual_seed(0)
    a = torch.randn((N), device='cuda', dtype=torch.float16)
    b = torch.randn((N), device='cuda', dtype=torch.float16)
    a = a.to(torch.float8_e4m3fn)
    b = b.to(torch.float8_e4m3fn)
    # print(f"a={a}")
    # print(f"b={b}")
    # print(triton.testing.do_bench(lambda: vecmul(a, b), warmup=10, rep=100))
    triton_output = vecmul(a, b)
    triton_output = triton_output.to(torch.float16)
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn
    torch_output = torch.mul(a.to(torch.float16), b.to(torch.float16))
    print(f"triton_output_with_fp8_inputs={triton_output}")
    print(f"torch_output={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
else:
    print("Skipping the test since torch.float8_e4m3fn is not available.")
