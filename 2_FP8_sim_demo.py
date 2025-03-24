import triton
import triton.language as tl
import torch
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

@triton.jit
def fp8_mul_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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

    # ==== exp add ====
    exp_sum = exp_a + exp_b - 7  # bias = 7 for E4M3

    # ==== normalize mantissa ====
    mant_norm = mant_mul >> 4  # 简化，正常要看最高位 normalize
    exp_sum = exp_sum + 1      # 确保 mantissa 是 1.xxxx

    # ==== pack back to FP8 ====
    # clip exp 和 mant
    mant_norm = mant_norm & 0x7  # 保留 3 bit mantissa
    exp_sum = tl.maximum(0, tl.minimum(exp_sum, 15))  # exp clip 到 4bit

    sign = sign_a ^ sign_b
    fp8_out = (sign << 7) | (exp_sum << 3) | mant_norm

    # store
    tl.store(c_ptr + offsets, fp8_out, mask=mask)

def vecmul(a, b):
    # Check constraints.
    assert a.shape == b.shape, "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    c = torch.empty(a.shape, device=a.device, dtype=torch.uint8)
    n_elements = a.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    fp8_mul_kernel[grid](a, b, c, n_elements, 256)
    c = c.view(torch.float8_e4m3fn)
    return c

# test fp8_mul_kernel accuracy

TORCH_HAS_FP8 = hasattr(torch, "float8_e4m3fn")
if TORCH_HAS_FP8 and is_cuda():
    torch.manual_seed(0)
    a = torch.randn((1), device='cuda', dtype=torch.float16)
    b = torch.randn((1), device='cuda', dtype=torch.float16)
    a_fp8 = a.to(torch.float8_e4m3fn)
    b_fp8 = b.to(torch.float8_e4m3fn)
    a_fp8 = a_fp8.view(torch.uint8)
    b_fp8 = b_fp8.view(torch.uint8)
    # print(triton.testing.do_bench(lambda: vecmul(a, b), warmup=10, rep=100))
    triton_output = vecmul(a_fp8, b_fp8)
    triton_output = triton_output.to(torch.float16)
    torch_output = torch.mul(a, b)
    print(f"triton_output_with_fp8_inputs={triton_output}")
    print(f"torch_output={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
else:
    print("Skipping the test since torch.float8_e4m3fn is not available.")
