import collections
import functools
import hashlib
import re
from pathlib import Path

from triton import knobs
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def col(*offsets):
    return [[0, n] for n in offsets]


def row(*offsets):
    return [[m, 0] for m in offsets]


def xor_rows(*chunks):
    return [[1 << i, n] for i, n in enumerate(chunks)]


def b_smem_layout(*chunks):
    return gl.SharedLinearLayout(VECTOR + col(*chunks) + xor_rows(*chunks[:3]) + row(8, 16, 32))


def b_copy_layout(*chunks):
    order = col(*chunks) + row(1, 2, 4, 8, 16, 32)
    return gl.DistributedLinearLayout(VECTOR + order[8:], order[:5], order[5:8], [], [64, 8 << len(chunks)])


def out_layout(width, registers, lanes):
    return gl.DistributedLinearLayout(VECTOR + registers, lanes, col(8, 16) + row(16), [], [32, width])


def swizzled_layout(inner):
    per_phase = max(64 // inner, 1)
    return gl.SwizzledSharedLayout(8, per_phase, 8 // per_phase, [1, 0])


def blocked_layout(rows, cols, warps):
    vector = min(8, rows * cols // (32 * warps))
    threads = min(cols // vector, 32)
    warp_rows = min(warps, rows * threads // 32)
    return gl.BlockedLayout([1, vector], [32 // threads, threads], [warp_rows, warps // warp_rows], [1, 0])


VECTOR = col(1, 2, 4)
MMA = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[2, 4], instr_shape=[16, 8])
A_OP = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=2)
B_OP = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=2)
RING = gl.SwizzledSharedLayout(8, 1, 8, order=[1, 0])
STAGE = gl.SharedLinearLayout(VECTOR + col(32, 64, 128) + xor_rows(32, 64, 128) + row(8) + col(8, 16) + row(16))
B_SMEM256 = b_smem_layout(32, 64, 128, 8, 16)
B_SMEM128 = b_smem_layout(32, 8, 16, 64)
B_SMEM64 = b_smem_layout(32, 8, 16)
A_COPY = gl.BlockedLayout([1, 8], [4, 8], [8, 1], [1, 0])
B_COPY256 = b_copy_layout(32, 8, 16, 64, 128)
B_COPY128 = b_copy_layout(32, 8, 16, 64)
B_COPY64 = b_copy_layout(32, 8, 16)
OUT256 = out_layout(256, row(4, 8), col(32, 64, 128) + row(1, 2))
OUT128 = out_layout(128, row(8), col(32, 64) + row(1, 2, 4))
OUT64 = out_layout(64, [], col(32) + row(4, 1, 2, 8))


@gluon.jit
def row_index(size: gl.constexpr, layout: gl.constexpr):
    return gl.arange(0, size, layout=gl.SliceLayout(1, layout))


@gluon.jit
def col_index(size: gl.constexpr, layout: gl.constexpr):
    return gl.arange(0, size, layout=gl.SliceLayout(0, layout))


@gluon.jit
def body_columns(n):
    return gl.max_contiguous(gl.multiple_of(n // 8 % 4 * 64 + n // 32 * 8 + n % 8, 8), 8)


@gluon.jit
def tail_columns(n):
    return gl.max_contiguous(gl.multiple_of(n // 64 * 64 + n // 8 % 4 * 16 + n // 32 % 2 * 8 + n % 8, 8), 8)


@gluon.jit
def body_tile(t):
    return t % 54 // 2 * (128 * 4096), (2 * (t // 54) + t % 2) * 256


@gluon.jit
def tail_origin(pid):
    col0 = 192 * (pid // 4)
    return 27 * 128 + 160 * (pid % 4), col0, gl.minimum(col0 + 128, 5120 - 64)


@gluon.jit
def b_slot(b_ring, slot):
    return b_ring.index(slot)._reinterpret(gl.float16, [64, 256], B_SMEM256)


@gluon.jit
def tail_slot(a_ring, b_ring, slot):
    b = b_ring.index(slot)._reinterpret(gl.float16, [2, 64, 128], RING)
    r = b.index(1)._reinterpret(gl.float16, [2, 64, 64], RING)
    return (a_ring.index(slot), r.index(1)._reinterpret(gl.float16, [2, 32, 64], RING).index(0),
            b.index(0)._reinterpret(gl.float16, [64, 128], B_SMEM128),
            r.index(0)._reinterpret(gl.float16, [64, 64], B_SMEM64))


@gluon.jit
def copy_stage(a_ptr, b_ptr, a_ring, b_ring, slot, a_tile, b_tile, k):
    a_block = row_index(128, A_COPY)[:, None] * 4096 + col_index(64, A_COPY)[None, :]
    b_block = row_index(64, B_COPY256)[:, None] * 5120 + body_columns(col_index(256, B_COPY256))[None, :]
    cp.async_copy_global_to_shared(a_ring.index(slot), a_ptr + (a_tile + k * 64) + a_block)
    cp.async_copy_global_to_shared(b_slot(b_ring, slot), b_ptr + (b_tile + k * (64 * 5120)) + b_block)
    cp.commit_group()


@gluon.jit
def copy_tail_stage(a_ptr, b_ptr, views, k, origin):
    a_high, a_low, b_left, b_right = views
    row0, col_left, col_right = origin
    ka = col_index(64, A_COPY)
    ra = row_index(128, A_COPY)
    rl = row_index(32, A_COPY)
    a = a_ptr + (row0 * 4096 + k * 64)
    cp.async_copy_global_to_shared(a_high, a + (ra[:, None] * 4096 + ka[None, :]))
    cp.async_copy_global_to_shared(a_low, a + ((128 + rl)[:, None] * 4096 + ka[None, :]))
    kb = row_index(64, B_COPY128)
    nb = col_index(128, B_COPY128)
    kr = row_index(64, B_COPY64)
    nr = col_index(64, B_COPY64)
    b = b_ptr + k * (64 * 5120)
    cp.async_copy_global_to_shared(b_left, b + col_left + (kb[:, None] * 5120 + tail_columns(nb)[None, :]))
    cp.async_copy_global_to_shared(b_right, b + col_right + (kr[:, None] * 5120 + tail_columns(nr)[None, :]))
    cp.commit_group()


@gluon.jit
def load_quarter(a_ring, b_ring, slot, q: gl.constexpr):
    a = a_ring.index(slot).slice(q * 16, 16, dim=1).load(A_OP)
    b = b_slot(b_ring, slot).slice(q * 16, 16, dim=0).load(B_OP)
    return a, b


@gluon.jit
def load_tail_quarter(views, q: gl.constexpr):
    a_high, a_low, b_left, b_right = views
    return (a_high.slice(q * 16, 16, dim=1).load(A_OP), a_low.slice(q * 16, 16, dim=1).load(A_OP),
            b_left.slice(q * 16, 16, dim=0).load(B_OP), b_right.slice(q * 16, 16, dim=0).load(B_OP))


@gluon.jit
def tail_mma(ops, acc):
    return (mma_v2(ops[0], ops[2], acc[0]), mma_v2(ops[0], ops[3], acc[1]),
            mma_v2(ops[1], ops[2], acc[2]), mma_v2(ops[1], ops[3], acc[3]))


@gluon.jit
def three_quarters(a_ring, b_ring, slot, a0, b0, acc):
    a1, b1 = load_quarter(a_ring, b_ring, slot, 1)
    acc = mma_v2(a0, b0, acc)
    a2, b2 = load_quarter(a_ring, b_ring, slot, 2)
    acc = mma_v2(a1, b1, acc)
    a3, b3 = load_quarter(a_ring, b_ring, slot, 3)
    acc = mma_v2(a2, b2, acc)
    cp.wait_group(1)
    gl.barrier()
    return a3, b3, acc


@gluon.jit
def tail_step(a_ptr, b_ptr, slots, s: gl.constexpr, k, ops, acc, origin, refill: gl.constexpr, pending: gl.constexpr):
    ops1 = load_tail_quarter(slots[s], 1)
    acc = tail_mma(ops, acc)
    if refill:
        copy_tail_stage(a_ptr, b_ptr, slots[(s + 2) % 3], k + 2, origin)
    ops2 = load_tail_quarter(slots[s], 2)
    acc = tail_mma(ops1, acc)
    ops3 = load_tail_quarter(slots[s], 3)
    acc = tail_mma(ops2, acc)
    cp.wait_group(pending)
    gl.barrier()
    ops = load_tail_quarter(slots[(s + 1) % 3], 0)
    return ops, tail_mma(ops3, acc)


@gluon.jit
def row_quarters(acc):
    v = gl.permute(gl.reshape(acc.to(gl.float16), [2, 2, 32, acc.shape[1]]), [2, 3, 0, 1])
    v0, v1 = gl.split(v)
    v00, v01 = gl.split(v0)
    v10, v11 = gl.split(v1)
    return v00, v10, v01, v11


@gluon.jit
def store_rows(c_ptr, v, row0, col0, layout: gl.constexpr, columns: gl.constexpr, mask):
    rm = row0 + row_index(32, layout)
    rn = col0 + columns(col_index(layout.shape[1], layout))
    gl.store(c_ptr + rm[:, None] * 5120 + rn[None, :], v, mask=mask)


@gluon.jit
def stage_rows(stage, piece):
    stage.store(piece)
    return stage.load(OUT256)


@gluon.jit
def store_body_rows(c_ptr, stage, piece, row0, col0):
    store_rows(c_ptr, stage_rows(stage, piece), row0, col0, OUT256, body_columns, None)


@gluon.jit
def store_tail_rows(c_ptr, piece, row0, col0, layout: gl.constexpr, mask):
    store_rows(c_ptr, gl.convert_layout(piece, layout), row0, col0, layout, tail_columns, mask)


@gluon.jit
def store_tile(c_ptr, stage, acc, row0, col0):
    v0, v32, v64, v96 = row_quarters(acc)
    store_body_rows(c_ptr, stage, v0, row0, col0)
    store_body_rows(c_ptr, stage, v64, row0 + 64, col0)
    store_body_rows(c_ptr, stage, v32, row0 + 32, col0)
    store_body_rows(c_ptr, stage, v96, row0 + 96, col0)


@gluon.jit
def store_last_tile(c_ptr, stage, acc, row0, col0):
    v0, v32, v64, v96 = row_quarters(acc)
    w0 = stage_rows(stage, v0)
    w32 = stage_rows(stage, v32)
    store_rows(c_ptr, w0, row0, col0, OUT256, body_columns, None)
    w64 = stage_rows(stage, v64)
    store_rows(c_ptr, w32, row0 + 32, col0, OUT256, body_columns, None)
    w96 = stage_rows(stage, v96)
    store_rows(c_ptr, w64, row0 + 64, col0, OUT256, body_columns, None)
    store_rows(c_ptr, w96, row0 + 96, col0, OUT256, body_columns, None)


@gluon.jit
def store_tail_block(c_ptr, acc, row0, col0, layout: gl.constexpr, mask):
    v0, v32, v64, v96 = row_quarters(acc)
    store_tail_rows(c_ptr, v0, row0, col0, layout, mask)
    store_tail_rows(c_ptr, v32, row0 + 32, col0, layout, mask)
    store_tail_rows(c_ptr, v64, row0 + 64, col0, layout, mask)
    store_tail_rows(c_ptr, v96, row0 + 96, col0, layout, mask)


@gluon.jit(noinline=True)
def cold_tail(a_ptr, b_ptr, c_ptr, pid, a_ring, b_ring):
    slots = (tail_slot(a_ring, b_ring, 2), tail_slot(a_ring, b_ring, 0), tail_slot(a_ring, b_ring, 1))
    origin = tail_origin(pid)
    acc = (gl.zeros([128, 128], gl.float32, MMA), gl.zeros([128, 64], gl.float32, MMA),
           gl.zeros([32, 128], gl.float32, MMA), gl.zeros([32, 64], gl.float32, MMA))
    ops = load_tail_quarter(slots[0], 0)
    ops, acc = tail_step(a_ptr, b_ptr, slots, 0, 0, ops, acc, origin, False, 1)
    for m in range(20):
        for j in gl.static_range(3):
            ops, acc = tail_step(a_ptr, b_ptr, slots, (j + 1) % 3, 3 * m + j + 1, ops, acc, origin, True, 1)
    ops, acc = tail_step(a_ptr, b_ptr, slots, 1, 61, ops, acc, origin, True, 1)
    ops, acc = tail_step(a_ptr, b_ptr, slots, 2, 62, ops, acc, origin, False, 0)
    ops, acc = tail_step(a_ptr, b_ptr, slots, 0, 63, ops, acc, origin, False, 0)
    row0, col0 = origin[0], origin[1]
    full = col0 < 5120 - 128
    store_tail_block(c_ptr, acc[0], row0, col0, OUT128, None)
    store_tail_block(c_ptr, acc[1], row0, col0 + 128, OUT64, full)
    store_tail_rows(c_ptr, acc[2].to(gl.float16), row0 + 128, col0, OUT128, None)
    store_tail_rows(c_ptr, acc[3].to(gl.float16), row0 + 128, col0 + 128, OUT64, full)


@gluon.jit
def ranked_kernel(a_ptr, b_ptr, c_ptr):
    a_ring = gl.allocate_shared_memory(gl.float16, [3, 128, 64], RING)
    b_ring = gl.allocate_shared_memory(gl.float16, [3, 64, 256], RING)
    stage = gl.allocate_shared_memory(gl.float16, [32, 256], STAGE)
    pid = gl.program_id(0)
    a_tile, b_tile = body_tile(pid)
    copy_stage(a_ptr, b_ptr, a_ring, b_ring, 0, a_tile, b_tile, 0)
    copy_stage(a_ptr, b_ptr, a_ring, b_ring, 1, a_tile, b_tile, 1)
    cp.wait_group(1)
    gl.barrier()
    copy_stage(a_ptr, b_ptr, a_ring, b_ring, 2, a_tile, b_tile, 2)
    a0, b0 = load_quarter(a_ring, b_ring, 0, 0)
    slot = gl.to_tensor(0)
    acc = gl.zeros([128, 256], gl.float32, MMA)
    for i in range(5):
        a_next, b_next = body_tile(pid + 108 * gl.minimum(i + 1, 4))
        acc = gl.zeros([128, 256], gl.float32, MMA)
        for k in range(64 - 3 * (i // 4)):
            next_slot = gl.where(slot == 2, 0, slot + 1)
            a3, b3, acc = three_quarters(a_ring, b_ring, slot, a0, b0, acc)
            ahead = k + 3
            wraps = ahead >= 64
            copy_stage(a_ptr, b_ptr, a_ring, b_ring, slot, gl.where(wraps, a_next, a_tile),
                       gl.where(wraps, b_next, b_tile), ahead & 63)
            a0, b0 = load_quarter(a_ring, b_ring, next_slot, 0)
            acc = mma_v2(a3, b3, acc)
            slot = next_slot
        if i < 4:
            store_tile(c_ptr, stage, acc, a_tile // 4096, b_tile)
        a_tile = a_next
        b_tile = b_next
    origin = tail_origin(pid)
    for k in gl.static_range(3):
        a3, b3, acc = three_quarters(a_ring, b_ring, (k + 2) % 3, a0, b0, acc)
        copy_tail_stage(a_ptr, b_ptr, tail_slot(a_ring, b_ring, (k + 2) % 3), k, origin)
        if k < 2:
            a0, b0 = load_quarter(a_ring, b_ring, k, 0)
        acc = mma_v2(a3, b3, acc)
    store_last_tile(c_ptr, stage, acc, a_tile // 4096, b_tile)
    cold_tail(a_ptr, b_ptr, c_ptr, pid, a_ring, b_ring)


@gluon.jit
def k_tile(t, NK: gl.constexpr, ROT: gl.constexpr):
    return (t + (NK - 1)) % NK if ROT else t


@gluon.jit
def copy_tile(a_ptr, b_ptr, a_dst, b_dst, off_m, off_n, kt, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
              MASK: gl.constexpr, A_CP: gl.constexpr, B_CP: gl.constexpr):
    k0 = kt * a_dst.shape[1]
    rm = off_m + row_index(a_dst.shape[0], A_CP)
    ka = k0 + col_index(a_dst.shape[1], A_CP)
    kb = k0 + row_index(b_dst.shape[0], B_CP)
    rn = off_n + col_index(b_dst.shape[1], B_CP)
    a_ptrs = a_ptr + rm[:, None] * K + ka[None, :]
    b_ptrs = b_ptr + kb[:, None] * N + rn[None, :]
    cp.async_copy_global_to_shared(a_dst, a_ptrs, mask=(rm[:, None] < M) & (ka[None, :] < K) if MASK else None)
    cp.async_copy_global_to_shared(b_dst, b_ptrs, mask=(kb[:, None] < K) & (rn[None, :] < N) if MASK else None)


@gluon.jit
def load_slice(a_src, b_src, k, KS: gl.constexpr, LHS: gl.constexpr, RHS: gl.constexpr):
    return a_src.slice(k, KS, dim=1).load(LHS), b_src.slice(k, KS, dim=0).load(RHS)


@gluon.jit
def mma_slice(a, b, acc0, acc1, odd: gl.constexpr):
    if odd:
        return acc0, mma_v2(a, b, acc1)
    return mma_v2(a, b, acc0), acc1


@gluon.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr,
                BN: gl.constexpr, BK: gl.constexpr, KS: gl.constexpr, STAGES: gl.constexpr, ROT: gl.constexpr,
                SLICED: gl.constexpr, ACC: gl.constexpr, LHS: gl.constexpr, RHS: gl.constexpr, A_SH: gl.constexpr,
                B_SH: gl.constexpr, A_CP: gl.constexpr, B_CP: gl.constexpr, C_ST: gl.constexpr):
    GM: gl.constexpr = (M + BM - 1) // BM
    GROUP: gl.constexpr = 8 * ((N + BN - 1) // BN)
    NK: gl.constexpr = (K + BK - 1) // BK
    MASK: gl.constexpr = M % BM + N % BN + K % BK > 0
    AHEAD: gl.constexpr = STAGES - 1 if STAGES > 1 else 1
    pid = gl.program_id(0)
    first_m = pid // GROUP * 8
    size_m = gl.minimum(GM - first_m, 8)
    off_m = (first_m + pid % GROUP % size_m) * BM
    off_n = pid % GROUP // size_m * BN
    a_ring = gl.allocate_shared_memory(gl.float16, [STAGES, BM, BK], A_SH)
    b_ring = gl.allocate_shared_memory(gl.float16, [STAGES, BK, BN], B_SH)
    acc0 = gl.zeros([BM, BN], gl.float32, ACC)
    acc1 = gl.zeros([BM, BN], gl.float32, ACC)
    for s in gl.static_range(AHEAD):
        if s < NK:
            copy_tile(a_ptr, b_ptr, a_ring.index(s), b_ring.index(s), off_m, off_n, k_tile(s, NK, ROT),
                      M, N, K, MASK, A_CP, B_CP)
        cp.commit_group()
    cp.wait_group(AHEAD - 1)
    gl.barrier()
    a, b = load_slice(a_ring.index(0), b_ring.index(0), 0, KS, LHS, RHS)
    for i in range(NK):
        a_src = a_ring.index(i % STAGES)
        b_src = b_ring.index(i % STAGES)
        for k in gl.static_range(KS, BK, KS):
            a_next, b_next = load_slice(a_src, b_src, k, KS, LHS, RHS)
            acc0, acc1 = mma_slice(a, b, acc0, acc1, SLICED and (k - KS) // 32 % 2 == 1)
            a, b = a_next, b_next
        if STAGES == 1:
            a_next, b_next = a, b
        else:
            j = i + STAGES - 1
            if j < NK:
                copy_tile(a_ptr, b_ptr, a_ring.index(j % STAGES), b_ring.index(j % STAGES), off_m, off_n,
                          k_tile(j, NK, ROT), M, N, K, MASK, A_CP, B_CP)
            cp.commit_group()
            cp.wait_group(STAGES - 2)
            gl.barrier()
            a_next, b_next = load_slice(a_ring.index((i + 1) % STAGES), b_ring.index((i + 1) % STAGES), 0,
                                        KS, LHS, RHS)
        acc0, acc1 = mma_slice(a, b, acc0, acc1, SLICED and (BK - KS) // 32 % 2 == 1)
        a, b = a_next, b_next
    cp.wait_group(0)
    if SLICED:
        acc0 = acc0 + acc1
    out = gl.convert_layout(acc0.to(gl.float16), C_ST)
    rm = off_m + row_index(BM, C_ST)
    rn = off_n + col_index(BN, C_ST)
    gl.store(c_ptr + rm[:, None] * N + rn[None, :], out, mask=(rm[:, None] < M) & (rn[None, :] < N) if MASK else None)


MARKER = re.compile(r"\t// (begin|end) inline asm\n")
FUNCTION = re.compile(r"^\.(visible \.entry|func) [^\n]*\(\n.*?^\}\n", re.M | re.S)
HALVES = re.compile(r"(mov\.b32 \t(?:%r\d+, )?)\{(%rs\d+), (%rs\d+)\}")
EDGE = re.compile(r"^((?:\$L__BB|\t(?:@%p\d+ )?bra|\tcall|\tret).*\n)", re.M)
REFILL = re.compile(r"(\tcp\.async\.wait_group \t1;\n(?:\t\.loc.*\n)*\tbar\.sync \t0;\n)"
                    r"((?:(?!\$L__BB|\tmma\.sync).*\n)*)((?:\tmma\.sync.*\n){32})")
COPY = re.compile(r"(?:(?!\tcp\.async\.cg).*\n)*\tcp\.async\.cg.*\n")
SMEM_READ = re.compile(r"ldmatrix|ld\.shared")
SMEM_WRITE = re.compile(r"cp\.async\.cg|st\.shared")
REGISTER = re.compile(r"%\w+")
BARRIER = "\tbar.sync \t0;\n"
WARP_BARRIER = "\tbar.warp.sync \t-1;\n"
COMMIT = "cp.async.commit_group;\n"
COPIES_BEFORE_MMA = {12: (8, 18, 28), 11: (0, 10, 20)}
EPILOGUE_OPS = ("bar.sync", "st.shared", "ld.shared", "ldmatrix", "st.global", "cp.async")
BODY_EPILOGUES = [(8, 64, 16, 0, 16, 0), (7, 64, 16, 0, 16, 0)]
TAIL_EPILOGUES = [(9, 30, 20, 10, 15, 0)]
FIRST_LATE_SLOT = 28
STAGGER_CYCLES = 1200
SPIN_CYCLES = 1200
LATE_ROWS = ("\t.reg .b32 \t%staggerr;\n\tmov.u32 \t%staggerr, %ctaid.x;\n\trem.u32 \t%staggerr, %staggerr, 54;\n"
             f"\tsetp.ge.u32 \t%staggerp, %staggerr, {FIRST_LATE_SLOT};\n"
             f"\tselp.b64 \t%stagger2, {STAGGER_CYCLES}, 0, %staggerp;\n")


def expect(ok, what):
    if not ok:
        raise RuntimeError(f"ranked PTX: unexpected {what}")


def strip_asm_markers(fn):
    """Drop LLVM's inline-asm comment lines so the later passes see instructions on adjacent lines."""
    fn, markers = MARKER.subn("", fn)
    expect(markers > 0, "inline asm markers")
    return fn


def carry_words(fn):
    """Keep the 32 operand words carried across a K step whole.

    Triton splits each such 32-bit word into two 16-bit halves and packs them again before the mma. A pair of
    halves that is only ever split and packed together becomes one %rw register.
    """
    moves = collections.Counter(m.groups()[1:] for m in HALVES.finditer(fn))
    uses = collections.Counter(re.findall(r"%rs\d+", fn))
    words = {pair: f"%rw{i}" for i, pair in enumerate(p for p in moves if uses[p[0]] == uses[p[1]] == moves[p])}
    expect(len(words) == 32, f"carried words {len(words)}")
    fn = HALVES.sub(lambda m: m[1] + words[m.groups()[1:]] if m.groups()[1:] in words else m[0], fn)
    fn, declared = re.subn(r"(\t\.reg \.b32 \t%r)(<\d+>;\n)", r"\1\2\1w<32>;\n", fn, count=1)
    expect(declared == 1, "register declarations")
    return fn


def drop_copy_barriers(fn):
    """Remove the 11 CTA barriers that only follow cp.async copies.

    A barrier stays when its section stores to shared memory or waits for a copy group, or when it separates
    shared-memory reads from the next section's writes.
    """
    sections = fn.split(BARRIER)
    kept = [sections[0]]
    for before, after in zip(sections, sections[1:]):
        only_copies = "cp.async.cg" in before and "st.shared" not in before and "wait_group" not in before
        read_then_write = SMEM_READ.search(before) and SMEM_WRITE.search(after)
        kept.append(after if only_copies and not read_then_write else BARRIER + after)
    out = "".join(kept)
    expect(fn.count(BARRIER) - out.count(BARRIER) == 11, "copy barriers")
    return out


def spread_refill(refill):
    section, mma = refill[2], refill[3].splitlines(keepends=True)
    copies = COPY.findall(section)
    expect(BARRIER not in section and section.count(COMMIT) == 1
           and not set(REGISTER.findall(section)) & set(REGISTER.findall(refill[3])), "refill order")
    chunks = ["".join(copies[:4]), "".join(copies[4:8]), section[len("".join(copies[:8])):], ""]
    bounds = (0, *COPIES_BEFORE_MMA[len(copies)], 32)
    return refill[1] + "".join("".join(mma[i:j]) + chunk for i, j, chunk in zip(bounds, bounds[1:], chunks))


def spread_copies(fn):
    """Interleave each ring refill with the 32 mma.sync of the quarter that follows it.

    Triton issues the refill's cp.async copies in one burst before the quarter; here they go out in three groups
    after COPIES_BEFORE_MMA of the mma instructions. The copies and the mma share no register.
    """
    expect([len(COPY.findall(refill[2])) for refill in REFILL.finditer(fn)] == [12, 11, 11, 11], "refill sections")
    return REFILL.sub(spread_refill, fn)


def warp_private_epilogues(fn, shapes):
    """Turn the CTA barriers of the store epilogues into warp barriers.

    Each warp stages and stores its own part of the tile, so nothing in an epilogue crosses warps. An epilogue
    is the code after the last mma.sync of a block that stores to global memory; `shapes` lists the expected
    EPILOGUE_OPS counts of every epilogue.
    """
    blocks = EDGE.split(fn)
    found = []
    for i, block in enumerate(blocks):
        if "st.global" in block:
            body, mma, epilogue = block.rpartition("mma.sync")
            found.append(tuple(epilogue.count(op) for op in EPILOGUE_OPS))
            blocks[i] = body + mma + epilogue.replace(BARRIER, WARP_BARRIER)
    expect(found == shapes, f"epilogues {found}")
    return "".join(blocks)


def spin(name, cycles, setup=""):
    """PTX block that waits until %clock64 has advanced by `cycles` (a literal or a register set in `setup`)."""
    return (f"\t{{\n\t.reg .b64 \t%{name}<3>;\n\t.reg .pred \t%{name}p;\n{setup}"
            f"\tmov.u64 \t%{name}0, %clock64;\n$L__{name}:\n\tmov.u64 \t%{name}1, %clock64;\n"
            f"\tsub.s64 \t%{name}1, %{name}1, %{name}0;\n\tsetp.lt.s64 \t%{name}p, %{name}1, {cycles};\n"
            f"\t@%{name}p bra \t$L__{name};\n\t}}\n")


def pace_cold_start(fn):
    """Two waits in the prologue, which copies ring slot 0, commits, copies slot 1 and commits.

    Row tiles 14-26 (ctaid % 54 >= FIRST_LATE_SLOT, 52 of 108 CTAs) wait STAGGER_CYCLES before their first copy;
    the other CTAs wait 0 cycles there. Without it all 108 CTAs reach every tile seam together and their tile
    stores collide with the next tile's loads. Every CTA then waits SPIN_CYCLES between the copies of slot 0
    and slot 1, so slot 0 is served first.
    """
    expect(fn.count(COMMIT) >= 2, "prologue commits")
    first, second, rest = fn.split(COMMIT, 2)
    expect(first.count("cp.async.cg") == second.count("cp.async.cg") == 12 and "wait_group" not in first + second,
           "prologue")
    head = first.rindex("\n", 0, first.index("\tcp.async.cg")) + 1
    return (first[:head] + spin("stagger", "%stagger2", LATE_ROWS) + first[head:] + COMMIT
            + spin("spin", SPIN_CYCLES) + second + COMMIT + rest)


PASSES = {
    "visible .entry": (strip_asm_markers, carry_words, drop_copy_barriers, spread_copies,
                       functools.partial(warp_private_epilogues, shapes=BODY_EPILOGUES), pace_cold_start),
    "func": (strip_asm_markers, carry_words, functools.partial(warp_private_epilogues, shapes=TAIL_EPILOGUES)),
}


def patch_function(match):
    fn = match[0]
    for ptx_pass in PASSES[match[1]]:
        fn = ptx_pass(fn)
    return fn


def patch_ptx(ptx):
    """Patch ranked_kernel (the entry) and its out-of-line cold_tail function with their PASSES."""
    expect([m[1] for m in FUNCTION.finditer(ptx)] == list(PASSES), "functions")
    return FUNCTION.sub(patch_function, ptx)


def patch_stages(*hooked):
    """Triton stage hook: called with the compile stages to wrap the PTX stage, and without arguments for the
    cache key, which is this file's hash."""
    if hooked:
        stages = hooked[1]
        emit = stages["ptx"]
        stages["ptx"] = lambda src, metadata: patch_ptx(emit(src, metadata))
    return SOURCE_HASH, SOURCE_HASH


def launch_ranked(a, b, c):
    with knobs.runtime.scope():
        knobs.runtime.add_stages_inspection_hook = patch_stages
        ranked_kernel[(108,)](a, b, c, num_warps=8)


def launch_generic(M, N, K, BM, BN, BK, STAGES, W=2, KS=16, ROT=False, SLICED=False):
    acc = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[W, W], instr_shape=[16, 8])
    return functools.partial(
        gemm_kernel[(-(-M // BM) * -(-N // BN),)], num_warps=W * W, M=M, N=N, K=K, BM=BM, BN=BN, BK=BK, KS=KS,
        STAGES=STAGES, ROT=ROT, SLICED=SLICED, ACC=acc, LHS=gl.DotOperandLayout(0, acc, 2),
        RHS=gl.DotOperandLayout(1, acc, 2), A_SH=swizzled_layout(BK), B_SH=swizzled_layout(BN),
        A_CP=blocked_layout(BM, BK, W * W), B_CP=blocked_layout(BK, BN, W * W), C_ST=blocked_layout(BM, BN, W * W))


def default_config(M, N, K):
    block, stages = (128, 4) if M * N >= 1 << 22 else (64, 4) if M * N >= 1 << 18 else (32, 3)
    bk = 32 if K % 32 == 0 else 16
    return dict(BM=block, BN=block, BK=bk, STAGES=stages if K > bk else 1)


SPECIALIZED = {(4096, 5120, 4096): launch_ranked}
CONFIGS = {
    (16, 16, 16): dict(BM=16, BN=16, BK=16, STAGES=1, W=1),
    (64, 64, 64): dict(BM=32, BN=32, BK=64, STAGES=1),
    (128, 128, 128): dict(BM=32, BN=32, BK=128, STAGES=1),
    (256, 256, 256): dict(BM=32, BN=32, BK=256, STAGES=1),
    (32, 512, 32): dict(BM=32, BN=64, BK=32, STAGES=1),
    (64, 1024, 64): dict(BM=64, BN=64, BK=64, STAGES=1),
    (512, 512, 512): dict(BM=64, BN=64, BK=64, STAGES=4),
    (1024, 1536, 1024): dict(BM=128, BN=128, BK=32, STAGES=4),
    (2048, 3072, 2048): dict(BM=128, BN=128, BK=32, STAGES=4),
    (416, 272, 416): dict(BM=64, BN=64, BK=32, STAGES=3, ROT=True),
    (384, 240, 464): dict(BM=64, BN=64, BK=64, STAGES=3, KS=32, ROT=True, SLICED=True),
    (512, 256, 512): dict(BM=64, BN=64, BK=64, STAGES=3, KS=32, SLICED=True),
}


@functools.cache
def launcher(M, N, K):
    return SPECIALIZED.get((M, N, K)) or launch_generic(M, N, K, **CONFIGS.get((M, N, K)) or default_config(M, N, K))


def custom_kernel(data):
    a, b, c = data
    launcher(a.shape[0], b.shape[1], a.shape[1])(a, b, c)
    return c
