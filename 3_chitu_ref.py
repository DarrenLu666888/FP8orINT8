# code from https://github.com/thu-pacman/chitu/blob/public-main/chitu/triton_kernels.py

import triton
import triton.language as tl
from triton import Config

fp8_gemm_deepseek_v3_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


# @triton.autotune(configs=fp8_gemm_deepseek_v3_configs, key=["N", "K"])
@triton.jit
def fp8_gemm_deepseek_v3_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


soft_fp8_gemm_deepseek_v3_configs = [
    Config(
        {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for block_k in [128]
    for group_m in [1, 32]
    for num_stages in [3, 4, 5, 6]
    for num_warps in [4, 8]
]


# @triton.autotune(configs=soft_fp8_gemm_deepseek_v3_configs, key=["N", "K"])
@triton.jit
def soft_fp8_gemm_deepseek_v3_kernel(
    A,
    B,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    fp8_to_fp32_scale: tl.constexpr,
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

    Args:
        A (tl.tensor): Pointer to the first input matrix A.
        B (tl.tensor): Pointer to the second input matrix B.
        C (tl.tensor): Pointer to the output matrix C.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.
        GROUP_SIZE_M (tl.constexpr): Block-swizzle group size for the M dimension.

    Returns:
        None
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = B + (offs_k[:, None] + offs_bn[None, :] * K)


    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        a_uint32 = a.to(tl.uint8, bitcast=True).to(tl.uint32)
        a_fp32 = (((a_uint32 & 0x80) << 24) | ((a_uint32 & 0x7F) << 20)).to(
            tl.float32, bitcast=True
        )
        b_uint32 = b.to(tl.uint8, bitcast=True).to(tl.uint32)
        b_fp32 = (((b_uint32 & 0x80) << 24) | ((b_uint32 & 0x7F) << 20)).to(
            tl.float32, bitcast=True
        )
        a_fp32 = a_fp32 * fp8_to_fp32_scale
        b_fp32 = b_fp32 * fp8_to_fp32_scale
        a_compute = a_fp32.to(tl.float16)
        b_compute = b_fp32.to(tl.float16)
        accumulator += tl.dot(a_compute, b_compute)

        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    c = accumulator.to(C.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# test fp8_gemm_deepseek_v3_kernel
if __name__ == "__main__":
    import torch

    M, N, K = 128, 128, 256
    a = torch.randn(M, K).cuda().to(torch.float8_e4m3fn)
    b = torch.randn(N, K).cuda().to(torch.float8_e4m3fn)
    c = torch.zeros(M, N).cuda().half()
    c_soft = torch.zeros(M, N).cuda().half()
    
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    fp8_gemm_deepseek_v3_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=128,
    )
    torch.cuda.synchronize()
    print(c)

    grid_soft = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    import struct
    # 0x7b800000 = 2^127
    fp8_to_fp32_scale = struct.unpack(">f", bytes.fromhex("7b800000"))[0]
    soft_fp8_gemm_deepseek_v3_kernel[grid_soft](
        a.view(torch.uint8),
        b.view(torch.uint8),
        c_soft,
        M,
        N,
        K,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=32,
        fp8_to_fp32_scale=fp8_to_fp32_scale,
    )
    torch.cuda.synchronize()
    print(c_soft)
    # Check the result
    expected_result = torch.matmul(a.to(torch.float16), b.transpose(0, 1).to(torch.float16))
    # expected_result = expected_result.to(torch.float16)
    print("Expected result:", expected_result)
    assert torch.allclose(c, expected_result, rtol=1e-2), "Output mismatch!"
    assert torch.allclose(c_soft, expected_result, rtol=1e-2), "Output mismatch!"
    print("Output is correct!")