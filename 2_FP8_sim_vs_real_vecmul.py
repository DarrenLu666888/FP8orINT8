import triton
import triton.language as tl
import torch
import pdb

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

@triton.jit
def fp8_mul_sim_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # pdb.set_trace()
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load fp8 数据 (假设是 uint8 存储的 FP8)
    a_fp8 = tl.load(a_ptr + offsets, mask=mask)
    b_fp8 = tl.load(b_ptr + offsets, mask=mask)

    # ==== FP8 E4M3 decode ====
    sign_a = a_fp8 >> 7
    sign_b = b_fp8 >> 7
    exp_a = (a_fp8 >> 3) & 0xF
    exp_b = (b_fp8 >> 3) & 0xF
    mant_a = a_fp8 & 0x7 | 0x8  # 恢复隐含的1位 => 1.xxx
    mant_b = b_fp8 & 0x7 | 0x8

    # ==== integer mantissa multiply ====
    mant_mul = mant_a * mant_b  # 8-bit * 8-bit => 16-bit
    mant_mul = mant_mul.to(tl.uint16)
    # ==== exp add ====
    exp_sum = exp_a + exp_b + 1  # bias = 15 for E5M10(float16) 1=-7-7+15

    # ==== normalize mantissa ====
    # pdb.set_trace()
    tmp_bit = ((mant_mul&0x80)>>7)
    exp_sum = exp_sum + tmp_bit      # 确保 mantissa 是 1.xxxx
    mant_norm = mant_mul << (4 - tmp_bit)  # 将小数位移动到fp16的fraction的位置

    # ==== pack to FP16 ====
    # clip exp 和 mant
    mant_norm = mant_norm & 0x3FF  # 保留 10 bit mantissa
    exp_sum = tl.maximum(0, tl.minimum(exp_sum, 31))  # exp clip 到 5bit

    sign = sign_a ^ sign_b

    sign = sign.to(tl.uint16)
    exp_sum = exp_sum.to(tl.uint16)

    fp16_out = (sign << 15) | (exp_sum << 10) | mant_norm

    # store
    tl.store(c_ptr + offsets, fp16_out, mask=mask)

def vecmul_sim(a, b):
    # Check constraints.
    assert a.shape == b.shape, "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    a = a.view(torch.uint8)
    b = b.view(torch.uint8)
    # print(f"a_viewasINT8={a}")
    # print(f"b_viewasINT8={b}")
    c = torch.empty(a.shape, device=a.device, dtype=torch.uint16)
    n_elements = a.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    fp8_mul_sim_kernel[grid](a, b, c, n_elements, 256)
    # print(f"c_viewasINT16={c}")
    c = c.view(torch.float16)
    return c

def vecmul_real(a, b):
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
# TORCH_HAS_FP8 = hasattr(torch, "float8_e4m3fn")
# if TORCH_HAS_FP8 and is_cuda():
#     N = int(sys.argv[1])
#     torch.manual_seed(0)
#     a = torch.randn((N), device='cuda', dtype=torch.float16)
#     b = torch.randn((N), device='cuda', dtype=torch.float16)
#     a = a.to(torch.float8_e4m3fn)
#     b = b.to(torch.float8_e4m3fn)
#     # print(f"a={a}")
#     # print(f"b={b}")
#     print('vecmul_real: ',triton.testing.do_bench(lambda: vecmul_real(a, b), warmup=10, rep=100))
#     print('vecmul_sim: ',triton.testing.do_bench(lambda: vecmul_sim(a, b), warmup=10, rep=100))
#     # triton_output = vecmul(a, b)
#     # triton_output = triton_output.to(torch.float16)
#     # assert a.dtype == torch.float8_e4m3fn
#     # assert b.dtype == torch.float8_e4m3fn
#     # torch_output = torch.mul(a.to(torch.float16), b.to(torch.float16))
#     # print(f"triton_output_with_fp8_inputs={triton_output}")
#     # print(f"torch_output={torch_output}")
#     # if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
#     #     print("✅ Triton and Torch match")
#     # else:
#     #     print("❌ Triton and Torch differ")
# else:
#     print("Skipping the test since torch.float8_e4m3fn is not available.")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 30, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['sim', 'real'],  # Possible values for `line_arg`.
        line_names=['sim', 'real'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-mul-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    torch.manual_seed(0)
    a = torch.randn((size), device='cuda', dtype=torch.float16)
    b = torch.randn((size), device='cuda', dtype=torch.float16)
    a = a.to(torch.float8_e4m3fn)
    b = b.to(torch.float8_e4m3fn)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'sim':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: vecmul_sim(a, b), warmup=10, rep=100, quantiles=quantiles)
    if provider == 'real':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: vecmul_real(a, b), warmup=10, rep=100, quantiles=quantiles)
    gbps = lambda ms: 1 * a.numel() * a.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True)