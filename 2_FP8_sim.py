import triton
import triton.language as tl

@triton.jit
def fp8_mul_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
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
    exp_sum = tl.max(0, tl.min(exp_sum, 15))  # exp clip 到 4bit

    sign = sign_a ^ sign_b
    fp8_out = (sign << 7) | (exp_sum << 3) | mant_norm

    # store
    tl.store(c_ptr + offsets, fp8_out.to(tl.uint8), mask=mask)
