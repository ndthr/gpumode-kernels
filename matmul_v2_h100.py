from triton.backends.nvidia.driver import TMA_DTYPE_DEVICE_TO_HOST
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (fence_async_shared, mbarrier, tma, warpgroup_mma,
                                                             warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.runtime import driver
from task import input_t, output_t

NUM_SMS = 132
A_RING = gl.constexpr(131072)
L64 = gl.constexpr(gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, 16]))
L256 = gl.constexpr(gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 256, 16]))
N192_MMA = "wgmma.mma_async.sync.aligned.m64n192k16.f32.f16.f16 {%s}, da, db, p, 1, 1, 0, 1;" % ",".join(
    f"${i}" for i in range(96))
N192_ASM = gl.constexpr("\n".join([
    "{", ".reg .b32 sb, ta, tb;", ".reg .b64 da, db;", ".reg .pred p;", "mov.u32 sb, global_smem;",
    "setp.eq.u32 p, sb, sb;", "add.u32 ta, sb, $192;", "add.u32 tb, sb, $224;", "bfe.u32 ta, ta, 4, 14;",
    "bfe.u32 tb, tb, 4, 14;", "cvt.u64.u32 da, ta;", "cvt.u64.u32 db, tb;", "or.b64 da, da, 0x4000004000000000;",
    "or.b64 db, db, 0x4000004002000000;", "wgmma.fence.sync.aligned;",
    "\nadd.s64 da, da, 2;\nadd.s64 db, db, 128;\n".join([N192_MMA] * 4), "wgmma.commit_group.sync.aligned;", "}"]))
N192_CON = gl.constexpr(",".join(["=f"] * 96 + [str(i) for i in range(96)] + ["r"] * 64))


@gluon.jit
def tile_coords(tile, tm, tn, GROUP_M: gl.constexpr):
    size = GROUP_M * tn
    group = tile // size
    rows = gl.minimum(tm - group * GROUP_M, GROUP_M)
    pid_n = tile % size // rows
    return group * GROUP_M + tile % size % rows, gl.where(group % 2 == 1, tn - 1 - pid_n, pid_n)


@gluon.jit
def ltile_coords(t, ROWS: gl.constexpr, M0: gl.constexpr, N0: gl.constexpr):
    u = t // ROWS
    return M0 + t % ROWS * 128, N0 + u // 2 * 448 + u % 2 * 192, u % 2


@gluon.jit
def split_n(x, SPLIT: gl.constexpr):
    xs = (x, )
    for _ in gl.static_range(SPLIT.bit_length() - 1):
        halves = ()
        for j in gl.static_range(len(xs)):
            halves += xs[j].reshape(xs[j].shape[0], 2, xs[j].shape[1] // 2).permute(0, 2, 1).split()
        xs = halves
    return xs


@gluon.jit
def acc_layout(BN: gl.constexpr, num_warps: gl.constexpr):
    return gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, BN, 16])


@gluon.jit
def ring(desc, STAGES: gl.constexpr):
    return gl.allocate_shared_memory(gl.float16, [STAGES] + desc.block_type.shape, desc.layout)


@gluon.jit
def barriers(STAGES: gl.constexpr, CONSUMERS: gl.constexpr):
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(STAGES):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(empty.index(i), count=CONSUMERS)
    return ready, empty


@gluon.jit
def load_tile(a_desc, b_desc, e_desc, a_bufs, b_bufs, e_bufs, ready, empty, off_m, off_n, k_tiles, it):
    BK: gl.constexpr = a_desc.block_type.shape[1]
    STAGES: gl.constexpr = a_bufs.type.shape[0]
    NBYTES: gl.constexpr = a_desc.block_type.nbytes + b_desc.block_type.nbytes + (0 if e_desc is None else
                                                                                   e_desc.block_type.nbytes)
    for kk in range(k_tiles):
        s = it % STAGES
        mbarrier.wait(empty.index(s), it // STAGES & 1 ^ 1)
        mbarrier.expect(ready.index(s), NBYTES)
        tma.async_copy_global_to_shared(a_desc, [off_m, kk * BK], ready.index(s), a_bufs.index(s))
        tma.async_copy_global_to_shared(b_desc, [kk * BK, off_n], ready.index(s), b_bufs.index(s))
        if e_desc is not None:
            tma.async_copy_global_to_shared(e_desc, [kk * BK, off_n + b_desc.block_type.shape[1]], ready.index(s),
                                            e_bufs.index(s))
        it += 1
    return it


@gluon.jit
def load_partition(a_desc, b_desc, e_desc, a_bufs, b_bufs, e_bufs, ready, empty, tm: gl.constexpr, tn: gl.constexpr,
                   k_tiles: gl.constexpr, GROUP_M: gl.constexpr, GRID: gl.constexpr, FULL, TAIL: gl.constexpr = 0,
                   ROWS: gl.constexpr = 1, M0: gl.constexpr = 0, N0: gl.constexpr = 0):
    BM: gl.constexpr = a_desc.block_type.shape[0]
    BN: gl.constexpr = b_desc.block_type.shape[1] + (0 if e_desc is None else e_desc.block_type.shape[1])
    pid = gl.program_id(0)
    it = 0
    for tile in range(pid, FULL, GRID):
        pid_m, pid_n = tile_coords(tile, tm, tn, GROUP_M)
        it = load_tile(a_desc, b_desc, e_desc, a_bufs, b_bufs, e_bufs, ready, empty, pid_m * BM, pid_n * BN, k_tiles,
                       it)
    if pid < TAIL:
        off_m, off_n, _ = ltile_coords(pid, ROWS, M0, N0)
        load_tile(a_desc, b_desc, e_desc, a_bufs, b_bufs, e_bufs, ready, empty, off_m, off_n, k_tiles, it)


@gluon.jit
def coop_partition(c_desc, a_bufs, b_bufs, ready, empty, c_bufs, tm: gl.constexpr, tn: gl.constexpr,
                   k_tiles: gl.constexpr, GROUP_M: gl.constexpr, GRID: gl.constexpr, num_warps: gl.constexpr):
    BM: gl.constexpr = c_desc.block_type.shape[0]
    BN: gl.constexpr = b_bufs.type.shape[2]
    SPLIT: gl.constexpr = c_bufs.type.shape[0]
    STAGES: gl.constexpr = a_bufs.type.shape[0]
    it = 0
    for tile in range(gl.program_id(0), tm * tn, GRID):
        pid_m, pid_n = tile_coords(tile, tm, tn, GROUP_M)
        s = it % STAGES
        mbarrier.wait(ready.index(s), it // STAGES & 1)
        acc = gl.zeros((BM, BN), gl.float32, acc_layout(BN, num_warps))
        acc = warpgroup_mma(a_bufs.index(s), b_bufs.index(s), acc, is_async=True)
        acc = warpgroup_mma_wait(1, deps=(acc, ))
        prev = s
        it += 1
        for _ in range(1, k_tiles):
            s = it % STAGES
            mbarrier.wait(ready.index(s), it // STAGES & 1)
            acc = warpgroup_mma(a_bufs.index(s), b_bufs.index(s), acc, is_async=True)
            acc = warpgroup_mma_wait(1, deps=(acc, ))
            mbarrier.arrive(empty.index(prev), count=1)
            prev = s
            it += 1
        acc = warpgroup_mma_wait(0, deps=(acc, ))
        mbarrier.arrive(empty.index(prev), count=1)
        parts = split_n(acc, SPLIT)
        for i in gl.static_range(SPLIT):
            tma.store_wait(pendings=SPLIT - 1)
            c_bufs.index(i).store(parts[i].to(gl.float16))
            fence_async_shared()
            tma.async_copy_shared_to_global(c_desc, [pid_m * BM, pid_n * BN + i * (BN // SPLIT)], c_bufs.index(i))
    tma.store_wait(pendings=0)


@gluon.jit
def coop_kernel(a_desc, b_desc, c_desc, tm: gl.constexpr, tn: gl.constexpr, k_tiles: gl.constexpr,
                STAGES: gl.constexpr, GROUP_M: gl.constexpr, GRID: gl.constexpr, SPLIT: gl.constexpr,
                num_warps: gl.constexpr):
    a_bufs = ring(a_desc, STAGES)
    b_bufs = ring(b_desc, STAGES)
    c_bufs = ring(c_desc, SPLIT)
    ready, empty = barriers(STAGES, 1)
    gl.warp_specialize([
        (coop_partition, (c_desc, a_bufs, b_bufs, ready, empty, c_bufs, tm, tn, k_tiles, GROUP_M, GRID, num_warps)),
        (load_partition, (a_desc, b_desc, None, a_bufs, b_bufs, None, ready, empty, tm, tn, k_tiles, GROUP_M, GRID,
                          tm * tn)),
    ], [1], [40])


@gluon.jit
def pair_partition(c_desc, f_desc, a_bufs, b_bufs, e_bufs, ready, empty, c_buf, f_buf, tm: gl.constexpr,
                   tn: gl.constexpr, k_tiles: gl.constexpr, GROUP_M: gl.constexpr, GRID: gl.constexpr,
                   num_warps: gl.constexpr):
    BM: gl.constexpr = c_desc.block_type.shape[0]
    BN0: gl.constexpr = b_bufs.type.shape[2]
    BN1: gl.constexpr = e_bufs.type.shape[2]
    STAGES: gl.constexpr = a_bufs.type.shape[0]
    it = 0
    for tile in range(gl.program_id(0), tm * tn, GRID):
        pid_m, pid_n = tile_coords(tile, tm, tn, GROUP_M)
        acc0 = gl.zeros((BM, BN0), gl.float32, acc_layout(BN0, num_warps))
        acc1 = gl.zeros((BM, BN1), gl.float32, acc_layout(BN1, num_warps))
        prev = 0
        for kk in range(k_tiles):
            s = it % STAGES
            mbarrier.wait(ready.index(s), it // STAGES & 1)
            acc0 = warpgroup_mma(a_bufs.index(s), b_bufs.index(s), acc0, is_async=True)
            acc1 = warpgroup_mma(a_bufs.index(s), e_bufs.index(s), acc1, is_async=True)
            acc0, acc1 = warpgroup_mma_wait(2, deps=(acc0, acc1))
            mbarrier.arrive(empty.index(prev), count=1, pred=kk > 0)
            prev = s
            it += 1
        acc0, acc1 = warpgroup_mma_wait(0, deps=(acc0, acc1))
        mbarrier.arrive(empty.index(prev), count=1)
        tma.store_wait(pendings=0)
        c_buf.store(acc0.to(gl.float16))
        f_buf.store(acc1.to(gl.float16))
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [pid_m * BM, pid_n * (BN0 + BN1)], c_buf)
        tma.async_copy_shared_to_global(f_desc, [pid_m * BM, pid_n * (BN0 + BN1) + BN0], f_buf)
    tma.store_wait(pendings=0)


@gluon.jit
def pair_kernel(a_desc, b_desc, e_desc, c_desc, f_desc, tm: gl.constexpr, tn: gl.constexpr, k_tiles: gl.constexpr,
                STAGES: gl.constexpr, GROUP_M: gl.constexpr, GRID: gl.constexpr, num_warps: gl.constexpr):
    a_bufs = ring(a_desc, STAGES)
    b_bufs = ring(b_desc, STAGES)
    e_bufs = ring(e_desc, STAGES)
    c_buf = gl.allocate_shared_memory(gl.float16, c_desc.block_type.shape, c_desc.layout)
    f_buf = gl.allocate_shared_memory(gl.float16, f_desc.block_type.shape, f_desc.layout)
    ready, empty = barriers(STAGES, 1)
    gl.warp_specialize([
        (pair_partition, (c_desc, f_desc, a_bufs, b_bufs, e_bufs, ready, empty, c_buf, f_buf, tm, tn, k_tiles,
                          GROUP_M, GRID, num_warps)),
        (load_partition, (a_desc, b_desc, e_desc, a_bufs, b_bufs, e_bufs, ready, empty, tm, tn, k_tiles, GROUP_M,
                          GRID, tm * tn)),
    ], [1], [40])


@gluon.jit
def n192(acc, a_off, b_off):
    ao = gl.zeros([64, 64], gl.int32, L64) + a_off
    bo = gl.zeros([64, 64], gl.int32, L64) + b_off
    return gl.inline_asm_elementwise(N192_ASM, N192_CON, [acc[0], acc[1], acc[2], ao, bo],
                                     dtype=(gl.float32, gl.float32, gl.float32), is_pure=False, pack=32)


@gluon.jit
def k_issue(a_bufs, b_bufs, ready, it, acc, WG: gl.constexpr, NARROW: gl.constexpr):
    STAGES: gl.constexpr = a_bufs.type.shape[0]
    s = it % STAGES
    mbarrier.wait(ready.index(s), it // STAGES & 1)
    if NARROW:
        acc = n192(acc, A_RING + s * 16384 + WG * 8192, s * 32768 + (1 - WG) * 8192)
    else:
        acc = warpgroup_mma(a_bufs.index(s).slice(WG * 64, 64, dim=0), b_bufs.index(s), acc, is_async=True)
    return acc, s


@gluon.jit
def k_retire(acc, PENDING: gl.constexpr, NARROW: gl.constexpr):
    if NARROW:
        acc = warpgroup_mma_wait(PENDING, deps=acc)
    else:
        acc = warpgroup_mma_wait(PENDING, deps=(acc, ))
    return acc


@gluon.jit
def k_steps(a_bufs, b_bufs, ready, empty, it, prev, n, acc, WG: gl.constexpr, NARROW: gl.constexpr):
    for _ in range(n):
        acc, s = k_issue(a_bufs, b_bufs, ready, it, acc, WG, NARROW)
        acc = k_retire(acc, 1, NARROW)
        mbarrier.arrive(empty.index(prev), count=1)
        prev = s
        it += 1
    return acc, it, prev


@gluon.jit
def store_piece(c_desc, c_bufs, part, row, col, g: gl.constexpr, WG: gl.constexpr):
    view = c_bufs.index(2 * WG + g % 2)
    tma.store_wait(pendings=1)
    view.store(part)
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [row, col], view)


@gluon.jit
def store_all(c_desc, c_bufs, parts, row, col, WG: gl.constexpr):
    for g in gl.static_range(len(parts)):
        store_piece(c_desc, c_bufs, parts[g], row, col + g * 64, g, WG)


@gluon.jit
def wg_tile(a_bufs, b_bufs, ready, empty, c_desc, c_bufs, k_tiles, it, out, row, col, pending, WG: gl.constexpr,
            NARROW: gl.constexpr, P1: gl.constexpr, OFF: gl.constexpr):
    if NARROW:
        z = gl.zeros((64, 64), gl.float32, L64)
        acc = (z, z, z)
    else:
        acc = gl.zeros((64, 256), gl.float32, L256)
    acc, prev = k_issue(a_bufs, b_bufs, ready, it, acc, WG, NARROW)
    acc = k_retire(acc, 1, NARROW)
    acc, it, prev = k_steps(a_bufs, b_bufs, ready, empty, it + 1, prev, OFF, acc, WG, NARROW)
    parts = split_n(out, 4)
    for g in gl.static_range(4):
        acc, s = k_issue(a_bufs, b_bufs, ready, it, acc, WG, NARROW)
        if pending:
            store_piece(c_desc, c_bufs, parts[g], row, col + g * 64, g, WG)
        acc = k_retire(acc, 1, NARROW)
        mbarrier.arrive(empty.index(prev), count=1)
        acc, it, prev = k_steps(a_bufs, b_bufs, ready, empty, it + 1, s, P1 - 1, acc, WG, NARROW)
    acc, it, prev = k_steps(a_bufs, b_bufs, ready, empty, it, prev, k_tiles - 1 - OFF - 4 * P1, acc, WG, NARROW)
    acc = k_retire(acc, 0, NARROW)
    mbarrier.arrive(empty.index(prev), count=1)
    return acc, it


@gluon.jit
def wg_partition(c_desc, a_bufs, b_bufs, ready, empty, c_bufs, tm: gl.constexpr, tn: gl.constexpr,
                 k_tiles: gl.constexpr, FULL: gl.constexpr, TAIL: gl.constexpr, ROWS: gl.constexpr, M0: gl.constexpr,
                 N0: gl.constexpr, GROUP_M: gl.constexpr, GRID: gl.constexpr, P1: gl.constexpr, OFF: gl.constexpr,
                 WG: gl.constexpr):
    pid = gl.program_id(0)
    out = gl.zeros((64, 256), gl.float16, L256)
    row = 0
    col = 0
    it = 0
    for tile in range(pid, FULL, GRID):
        pid_m, pid_n = tile_coords(tile, tm, tn, GROUP_M)
        acc, it = wg_tile(a_bufs, b_bufs, ready, empty, c_desc, c_bufs, k_tiles, it, out, row, col, tile != pid, WG,
                          False, P1, OFF)
        out = acc.to(gl.float16)
        row = pid_m * 128 + WG * 64
        col = pid_n * 256
    if pid < TAIL:
        off_m, off_n, kind = ltile_coords(pid, ROWS, M0, N0)
        if kind == WG:
            acc, it = wg_tile(a_bufs, b_bufs, ready, empty, c_desc, c_bufs, k_tiles, it, out, row, col, pid < FULL, WG,
                              False, P1, OFF)
            store_all(c_desc, c_bufs, split_n(acc.to(gl.float16), 4), off_m + WG * 64, off_n, WG)
        else:
            thirds, it = wg_tile(a_bufs, b_bufs, ready, empty, c_desc, c_bufs, k_tiles, it, out, row, col, pid < FULL,
                                 WG, True, P1, OFF)
            parts = (thirds[0].to(gl.float16), thirds[1].to(gl.float16), thirds[2].to(gl.float16))
            store_all(c_desc, c_bufs, parts, off_m + WG * 64, off_n + 64 - 64 * WG, WG)
    elif pid < FULL:
        store_all(c_desc, c_bufs, split_n(out, 4), row, col, WG)
    tma.store_wait(pendings=0)


@gluon.jit
def ltile_kernel(a_desc, b_desc, c_desc, tm: gl.constexpr, tn: gl.constexpr, k_tiles: gl.constexpr,
                 STAGES: gl.constexpr, FULL: gl.constexpr, TAIL: gl.constexpr, ROWS: gl.constexpr, M0: gl.constexpr,
                 N0: gl.constexpr, GROUP_M: gl.constexpr, GRID: gl.constexpr, P1: gl.constexpr,
                 OFF1: gl.constexpr):
    a_bufs = ring(a_desc, STAGES)
    b_bufs = ring(b_desc, STAGES)
    c_bufs = ring(c_desc, 4)
    ready, empty = barriers(STAGES, 2)
    gl.warp_specialize([
        (wg_partition, (c_desc, a_bufs, b_bufs, ready, empty, c_bufs, tm, tn, k_tiles, FULL, TAIL, ROWS, M0, N0,
                         GROUP_M, GRID, P1, 0, 0)),
        (wg_partition, (c_desc, a_bufs, b_bufs, ready, empty, c_bufs, tm, tn, k_tiles, FULL, TAIL, ROWS, M0, N0,
                         GROUP_M, GRID, P1, OFF1, 1)),
        (load_partition, (a_desc, b_desc, None, a_bufs, b_bufs, None, ready, empty, tm, tn, k_tiles, GROUP_M, GRID,
                          FULL, TAIL, ROWS, M0, N0)),
    ], [4, 1], [232, 40])


def host_tile_coords(tile, tm, tn, group_m):
    size = group_m * tn
    group = tile // size
    rows = min(tm - group * group_m, group_m)
    pid_n = tile % size // rows
    return group * group_m + tile % size % rows, tn - 1 - pid_n if group % 2 else pid_n


def ltile_plan(tm, tn, group_m):
    full = tm * tn // NUM_SMS * NUM_SMS
    left = {host_tile_coords(t, tm, tn, group_m) for t in range(full, tm * tn)}
    (r0, c0), (r1, c1) = min(left), max(left)
    pairs, rem = divmod((c1 - c0 + 1) * 256, 448)
    assert left == {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)} and not rem
    assert 2 * (r1 - r0 + 1) * pairs <= NUM_SMS
    return full, 2 * (r1 - r0 + 1) * pairs, r1 - r0 + 1, r0 * 128, c0 * 256


def desc(t, *block):
    return TensorDescriptor.from_tensor(t, list(block), gl.NVMMASharedLayout.get_default_for(list(block), gl.float16))


def ltile_launch(a, b, c, STAGES, GROUP_M, P1, OFF1):
    tm, tn = a.shape[0] // 128, b.shape[1] // 256
    args = (desc(a, 128, 64), desc(b, 64, 256), desc(c, 64, 64), tm, tn, a.shape[1] // 64, STAGES,
            *ltile_plan(tm, tn, GROUP_M), GROUP_M, NUM_SMS, P1, OFF1)
    return ltile_kernel, NUM_SMS, args, dict(num_warps=4, maxnreg=168)


def coop_launch(a, b, c, BM, BN, BK, STAGES, GROUP_M, SPLIT=1, GRID=NUM_SMS):
    (m, k), n = a.shape, b.shape[1]
    tm, tn = -(-m // BM), -(-n // BN)
    grid = min(GRID, tm * tn)
    args = (desc(a, BM, BK), desc(b, BK, BN), desc(c, BM, BN // SPLIT), tm, tn, -(-k // BK), STAGES, GROUP_M, grid,
            SPLIT)
    return coop_kernel, grid, args, dict(num_warps=BM // 16, maxnreg=168 if BM == 128 else 232)


def pair_launch(a, b, c, BM, BN0, BN1, BK, STAGES, GROUP_M):
    (m, k), n = a.shape, b.shape[1]
    tm, tn = m // BM, n // (BN0 + BN1)
    grid = min(NUM_SMS, tm * tn)
    args = (desc(a, BM, BK), desc(b, BK, BN0), desc(b, BK, BN1), desc(c, BM, BN0), desc(c, BM, BN1), tm, tn, k // BK,
            STAGES, GROUP_M, grid)
    return pair_kernel, grid, args, dict(num_warps=BM // 16, maxnreg=168)


class Launcher:

    def __init__(self, kernel, grid, args, kw, tensors):
        compiled = kernel.warmup(*args, grid=(grid, ), **kw)
        run = compiled.run
        bound = kernel.device_caches[driver.active.get_current_device()][4](*args, **kw)[0]
        metas = iter(compiled.metadata.tensordesc_meta)
        self.template, self.maps = [], []
        for v in bound.values():
            if isinstance(v, TensorDescriptor):
                md = next(metas)
                src = next(i for i, t in enumerate(tensors) if v.base is t)
                self.maps.append((len(self.template), src, md["swizzle"], md["elem_size"],
                                  TMA_DTYPE_DEVICE_TO_HOST[md["elem_type"]], list(md["block_size"]), list(v.shape),
                                  list(v.strides)))
                self.template += [None, *v.shape, *v.strides]
            else:
                self.template.append(v)
        self.head = (grid, 1, 1, 0, compiled.function, run.launch_cooperative_grid, run.launch_pdl,
                     compiled.packed_metadata, None, None, None, None, None, run.arg_annotations, run.kernel_signature)

    def __call__(self, tensors):
        args = self.template.copy()
        for pos, src, *encoding in self.maps:
            args[pos] = driver.active.utils.fill_tma_descriptor_tiled(tensors[src].data_ptr(), *encoding, 0)
        driver.active.utils.launch(*self.head, args)


CONFIGS = {
    (4096, 5120, 4096): (ltile_launch, dict(STAGES=4, GROUP_M=16, P1=8, OFF1=4)),
    (2048, 3072, 2048): (coop_launch, dict(BM=128, BN=128, BK=64, STAGES=6, GROUP_M=8, SPLIT=2)),
    (2048, 2048, 2048): (coop_launch, dict(BM=128, BN=128, BK=64, STAGES=6, GROUP_M=8, SPLIT=2, GRID=128)),
    (1024, 1536, 1024): (pair_launch, dict(BM=64, BN0=128, BN1=64, BK=64, STAGES=6, GROUP_M=4)),
    (1024, 1024, 1024): (coop_launch, dict(BM=64, BN=128, BK=64, STAGES=8, GROUP_M=8)),
    (512, 512, 512): (coop_launch, dict(BM=64, BN=32, BK=128, STAGES=4, GROUP_M=8)),
    (256, 256, 256): (coop_launch, dict(BM=64, BN=16, BK=64, STAGES=4, GROUP_M=8)),
    (128, 128, 128): (coop_launch, dict(BM=64, BN=32, BK=128, STAGES=2, GROUP_M=8)),
}
DEFAULT = (coop_launch, dict(BM=64, BN=64, BK=64, STAGES=3, GROUP_M=8))
LAUNCHERS = {}


def custom_kernel(data: input_t) -> output_t:
    a, b, c = data
    key = (a.shape[0], b.shape[1], a.shape[1])
    if key not in LAUNCHERS:
        launch, cfg = CONFIGS.get(key, DEFAULT)
        LAUNCHERS[key] = Launcher(*launch(a, b, c, **cfg), data)
    LAUNCHERS[key](data)
    return c
