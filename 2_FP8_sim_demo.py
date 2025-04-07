import triton
import triton.language as tl
import torch
import pdb

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

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

def vecmul(a, b):
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
    print(triton.testing.do_bench(lambda: vecmul(a, b), warmup=10, rep=100))
    triton_output = vecmul(a, b)
    # triton_output = triton_output.to(torch.float16)
    # assert a.dtype == torch.float8_e4m3fn
    # assert b.dtype == torch.float8_e4m3fn
    # torch_output = torch.mul(a.to(torch.float16), b.to(torch.float16))
    # print(f"triton_output_with_fp8_inputs={triton_output}")
    # print(f"torch_output={torch_output}")
    # if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
    #     print("✅ Triton and Torch match")
    # else:
    #     print("❌ Triton and Torch differ")
else:
    print("Skipping the test since torch.float8_e4m3fn is not available.")
