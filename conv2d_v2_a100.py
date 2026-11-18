import ctypes
import glob
import os
import sys
from functools import lru_cache

import torch
import helion
import helion.language as hl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


# ---------------------------------------------------------------------------------------------------------------
# Exact fallback (small shapes): direct convolution as one IEEE GEMM over (co) x (output position).
@helion.kernel(static_shapes=True, dot_precision='ieee', config=helion.Config(block_sizes=[64, 64, 32], num_warps=4, num_stages=3))
def exact_conv(x, w, out):
    batch, channels, size, _ = x.shape
    width = w.shape[-1]
    side = size - width + 1
    for co, pos in hl.tile([channels, batch * side * side]):
        p = pos.index
        b = p // (side * side)
        oy = p // side % side
        ox = p % side
        acc = hl.zeros([co, pos], dtype=torch.float32)
        for red in hl.tile(channels * width * width):
            r = red.index
            ci = r // (width * width)
            ky = r // width % width
            kx = r % width
            a = w[co.index[:, None], ci[None, :], ky[None, :], kx[None, :]]
            v = x[b[None, :], ci[:, None], ky[:, None] + oy[None, :], kx[:, None] + ox[None, :]]
            acc = hl.dot(a, v, acc)
        out[b[None, :], co.index[:, None], oy[None, :], ox[None, :]] = acc


# ---------------------------------------------------------------------------------------------------------------
# Helion overlap-save FFT convolution (S128 K8/K16 with 48-point tiles, S256 K16 with 80-point tiles).
# Stage 1: cut x into overlapping t x t tiles (zero beyond the border).
@helion.kernel(config=helion.Config(block_sizes=[256], num_warps=4))
def gather_tiles(x, tiles, step: hl.constexpr, nt: hl.constexpr):
    b, c, s, _ = x.shape
    t = tiles.shape[-1]
    flat = tiles.view(-1)
    source = x.view(-1)
    for pos in hl.tile(flat.numel()):
        p = pos.index
        tile = p // (c * t * t)
        ci = p // (t * t) % c
        y = tile // nt % nt * step + p // t % t
        z = tile % nt * step + p % t
        flat[pos] = hl.load(source, [((tile // (nt * nt) * c + ci) * s + y) * s + z], extra_mask=(y < s) & (z < s))


# [rows, freq] <-> [freq, rows] for the real and imaginary planes of a spectrum.
@helion.kernel(config=helion.Config(block_sizes=[32, 32], num_warps=4))
def transpose_spectra(xr, xi, yr, yi):
    rows, freq = xr.shape
    for r, f in hl.tile([rows, freq]):
        yr[f, r] = xr[r, f].T
        yi[f, r] = xi[r, f].T


def transpose_complex(z):
    z = z.reshape(-1, z.shape[-1])
    out = torch.empty((z.shape[1], z.shape[0]), dtype=z.dtype, device=z.device)
    transpose_spectra(z.real, z.imag, out.real, out.imag)
    return out


# Stage 2 (S256 only): weight spectrum as two tf32x3 DFT GEMMs (rows, then columns) instead of a padded cuFFT.
@helion.kernel(static_shapes=True, dot_precision='tf32x3', config=helion.Config(block_sizes=[64, 64, 32], num_warps=4, num_stages=2))
def dft_rows(w, cr, si, yr, yi):
    rows, width = w.shape
    freq = cr.shape[1]
    for mt, nt in hl.tile([rows, freq]):
        real = hl.zeros([mt, nt], dtype=torch.float32)
        imag = hl.zeros([mt, nt], dtype=torch.float32)
        for kt in hl.tile(width):
            a = w[mt, kt]
            real = hl.dot(a, cr[kt, nt], real)
            imag = hl.dot(a, si[kt, nt], imag)
        yr[nt, mt] = real.T
        yi[nt, mt] = imag.T


@helion.kernel(static_shapes=True, dot_precision='tf32x3', config=helion.Config(block_sizes=[1, 32, 64, 32], num_warps=4, num_stages=2))
def dft_columns(xr, xi, cr, si, yr, yi):
    freq, filters, width = xr.shape
    height = cr.shape[1]
    for ft, mt, nt in hl.tile([freq, height, filters]):
        real = hl.zeros([ft, mt, nt], dtype=torch.float32)
        imag = hl.zeros([ft, mt, nt], dtype=torch.float32)
        for kt in hl.tile(width):
            ar = cr[ft, mt, kt]
            ai = si[ft, mt, kt]
            br = xr[ft, nt, kt].transpose(1, 2)
            bi = xi[ft, nt, kt].transpose(1, 2)
            real = hl.dot(ar, br, real)
            real = hl.dot(-ai, bi, real)
            imag = hl.dot(ar, bi, imag)
            imag = hl.dot(ai, br, imag)
        yr[mt.index[None, :, None] * freq + ft.index[:, None, None], nt.index[None, None, :]] = real
        yi[mt.index[None, :, None] * freq + ft.index[:, None, None], nt.index[None, None, :]] = imag


@lru_cache(None)
def twiddles(width, t, device):
    phase = -2 * torch.pi * torch.arange(t, dtype=torch.float64)[:, None] * torch.arange(width, dtype=torch.float64)[None, :] / t
    cr, si = (phase.cos().float().to(device), phase.sin().float().to(device))
    freq = t // 2 + 1
    return (cr[:freq].T.contiguous(), si[:freq].T.contiguous(), cr[None].expand(freq, -1, -1), si[None].expand(freq, -1, -1))


def weight_dft(w, t):
    c, _, width, _ = w.shape
    freq = t // 2 + 1
    cr, si, hc, hs = twiddles(width, t, w.device)
    tmp = torch.empty((freq, c * c * width), device=w.device, dtype=torch.complex64)
    dft_rows(w.view(-1, width), cr, si, tmp.real, tmp.imag)
    out = torch.empty((t * freq, c * c), device=w.device, dtype=torch.complex64)
    dft_columns(tmp.real.view(freq, c * c, width), tmp.imag.view(freq, c * c, width), hc, hs, out.real, out.imag)
    return out.view(t * freq, c, c)


# Stage 3: per-frequency channel mixer Y[f, m, co] = sum_ci X[f, m, ci] * conj(W[f, co, ci]).
@helion.kernel(dot_precision='tf32x3', static_shapes=True, config=helion.Config(block_sizes=[1, 32, 32, 32], num_warps=4, num_stages=2))
def mix_tiles(xr, xi, wr, wi, yr, yi):
    freq, m, c = xr.shape
    n = wr.shape[1]
    for ft, mt, nt in hl.tile([freq, m, n]):
        real = hl.zeros([ft, mt, nt], dtype=torch.float32)
        imag = hl.zeros([ft, mt, nt], dtype=torch.float32)
        for kt in hl.tile(c):
            ar, ai = (xr[ft, mt, kt], xi[ft, mt, kt])
            br, bi = (wr[ft, nt, kt].transpose(1, 2), wi[ft, nt, kt].transpose(1, 2))
            real = hl.dot(ar, br, real)
            real = hl.dot(ai, bi, real)
            imag = hl.dot(ai, br, imag)
            imag = hl.dot(-ar, bi, imag)
        yr[ft, mt, nt] = real
        yi[ft, mt, nt] = imag


# Stage 4: keep the valid step x step corner of every inverse-transformed tile.
@helion.kernel(config=helion.Config(block_sizes=[256], num_warps=4))
def scatter_tiles(tiles, out, step: hl.constexpr, nt: hl.constexpr):
    b, c, o, _ = out.shape
    flat = out.view(-1)
    source = tiles.view(-1)
    t = tiles.shape[-1]
    for pos in hl.tile(flat.numel()):
        p = pos.index
        y, x = (p // o % o, p % o)
        tile = p // (c * o * o) * nt * nt + y // step * nt + x // step
        flat[pos] = source[((tile * c + p // (o * o) % c) * t + y % step) * t + x % step]


def overlap_conv(data, t):
    x, w, out = data
    b, c, s, _ = x.shape
    k = w.shape[-1]
    step = t - k + 1
    nt = (s - k + step) // step
    m, freq = (b * nt * nt, t * (t // 2 + 1))
    tiles = torch.empty((m, c, t, t), device=x.device, dtype=x.dtype)
    gather_tiles(x, tiles, step, nt)
    xf = transpose_complex(torch.fft.rfft2(tiles).flatten(2)).view(freq, m, c)
    wf = transpose_complex(torch.fft.rfft2(w, s=(t, t)).flatten(2)).view(freq, c, c) if s == 128 else weight_dft(w, t)
    yf = torch.empty_like(xf)
    mix_tiles(xf.real, xf.imag, wf.real, wf.imag, yf.real, yf.imag)
    yf = transpose_complex(yf.view(freq, m * c)).view(m, c, t, t // 2 + 1)
    result = torch.fft.irfft2(yf, s=(t, t))
    scatter_tiles(result, out, step, nt)
    return out


# ---------------------------------------------------------------------------------------------------------------
# CUDA runtime: NVRTC compile + driver launch through ctypes, on torch's current context and queue.
def nvrtc_candidates():
    roots = sorted({p for p in sys.path if p.endswith("site-packages")})
    hits = []
    for r in roots:
        hits += sorted(glob.glob(os.path.join(r, "nvidia", "**", "libnvrtc.so*"), recursive=True))
    return hits + sorted(glob.glob("/usr/local/cuda*/lib64/libnvrtc.so*"))


class Cuda:
    def __init__(self):
        self.nvrtc = None
        for h in nvrtc_candidates():
            try:
                self.nvrtc = ctypes.CDLL(h)
                break
            except OSError:
                pass
        V, P = ctypes.c_void_p, ctypes.POINTER
        n = self.nvrtc
        n.nvrtcCreateProgram.argtypes = [P(V), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, V, V]
        n.nvrtcCompileProgram.argtypes = [V, ctypes.c_int, P(ctypes.c_char_p)]
        n.nvrtcGetProgramLogSize.argtypes = [V, P(ctypes.c_size_t)]
        n.nvrtcGetProgramLog.argtypes = [V, ctypes.c_char_p]
        n.nvrtcGetPTXSize.argtypes = [V, P(ctypes.c_size_t)]
        n.nvrtcGetPTX.argtypes = [V, ctypes.c_char_p]
        d = self.drv = ctypes.CDLL("libcuda.so.1")
        d.cuCtxGetCurrent.argtypes = [P(V)]
        d.cuModuleLoadData.argtypes = [P(V), V]
        d.cuModuleGetFunction.argtypes = [P(V), V, ctypes.c_char_p]
        d.cuFuncSetAttribute.argtypes = [V, ctypes.c_int, ctypes.c_int]
        d.cuLaunchKernel.argtypes = [V] + [ctypes.c_uint] * 7 + [V, V, V]
        torch.zeros(1, device="cuda")
        ctx = V()
        rc = d.cuCtxGetCurrent(ctypes.byref(ctx))
        assert rc == 0 and ctx.value
        cap = torch.cuda.get_device_capability()
        self.arch = f"sm_{cap[0]}{cap[1]}"

    def build(self, source, name, defs=()):
        opts = [f"--gpu-architecture={self.arch}", "-std=c++17"] + [f"-D{k}={v}" for k, v in defs]
        arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
        prog = ctypes.c_void_p()
        rc = self.nvrtc.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), b"k.cu", 0, None, None)
        assert rc == 0, f"create {rc}"
        rc = self.nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
        size = ctypes.c_size_t()
        self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value + 1)
        self.nvrtc.nvrtcGetProgramLog(prog, log)
        assert rc == 0, f"compile {rc}: {log.value.decode(errors='replace')[-3000:]}"
        self.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(size))
        ptx = ctypes.create_string_buffer(size.value + 1)
        self.nvrtc.nvrtcGetPTX(prog, ptx)
        mod = ctypes.c_void_p()
        rc = self.drv.cuModuleLoadData(ctypes.byref(mod), ctypes.cast(ptx, ctypes.c_void_p))
        assert rc == 0, f"module load {rc}"
        fn = ctypes.c_void_p()
        rc = self.drv.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode())
        assert rc == 0, f"get function {rc}"
        return fn

    def launch(self, fn, grid, block, smem, args):
        rc = self.drv.cuFuncSetAttribute(fn, 8, smem)
        assert rc == 0, f"set smem {rc}"
        arr = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
        rc = self.drv.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, smem, None, ctypes.cast(arr, ctypes.c_void_p), None)
        assert rc == 0, f"launch {rc}"


@lru_cache(None)
def cuda_runtime():
    return Cuda()


def pointers(*tensors):
    return [ctypes.c_void_p(t.data_ptr()) for t in tensors]


# ---------------------------------------------------------------------------------------------------------------
# S256 K32 C128 B1: 16 tiles of 88x88 (step 57), 88 = 8 x 11 mixed-radix FFTs, spectra f = fh*45 + fw (3960 bins).
# Shared CUDA pieces: inline-PTX tensor-core / fp16 helpers and the 88-point DFT building blocks.
CU_MMA = r"""
#define MMA(d, a0, a1, a2, a3, b0, b1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};" : "=f"((d)[0]), "=f"((d)[1]), "=f"((d)[2]), "=f"((d)[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"((d)[0]), "f"((d)[1]), "f"((d)[2]), "f"((d)[3]))
#define CVT2(o, hi, lo) asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(o) : "f"(hi), "f"(lo))
static __device__ __forceinline__ unsigned f2u(float v) { unsigned u; asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v)); return u; }
static __device__ __forceinline__ float u2f(unsigned u) { float v; asm("mov.b32 %0, %1;" : "=f"(v) : "r"(u)); return v; }
"""

CU_FFT = r"""
__constant__ float C88[88] = {1.000000000e+00f,9.974521146e-01f,9.898214419e-01f,9.771468660e-01f,9.594929736e-01f,9.369497250e-01f,9.096319954e-01f,8.776789896e-01f,8.412535328e-01f,8.005412409e-01f,7.557495744e-01f,7.071067812e-01f,6.548607339e-01f,5.992776665e-01f,5.406408175e-01f,4.792489867e-01f,4.154150130e-01f,3.494641796e-01f,2.817325568e-01f,2.125652896e-01f,1.423148383e-01f,7.133918320e-02f,2.832769449e-16f,-7.133918320e-02f,-1.423148383e-01f,-2.125652896e-01f,-2.817325568e-01f,-3.494641796e-01f,-4.154150130e-01f,-4.792489867e-01f,-5.406408175e-01f,-5.992776665e-01f,-6.548607339e-01f,-7.071067812e-01f,-7.557495744e-01f,-8.005412409e-01f,-8.412535328e-01f,-8.776789896e-01f,-9.096319954e-01f,-9.369497250e-01f,-9.594929736e-01f,-9.771468660e-01f,-9.898214419e-01f,-9.974521146e-01f,-1.000000000e+00f,-9.974521146e-01f,-9.898214419e-01f,-9.771468660e-01f,-9.594929736e-01f,-9.369497250e-01f,-9.096319954e-01f,-8.776789896e-01f,-8.412535328e-01f,-8.005412409e-01f,-7.557495744e-01f,-7.071067812e-01f,-6.548607339e-01f,-5.992776665e-01f,-5.406408175e-01f,-4.792489867e-01f,-4.154150130e-01f,-3.494641796e-01f,-2.817325568e-01f,-2.125652896e-01f,-1.423148383e-01f,-7.133918320e-02f,-1.836970199e-16f,7.133918320e-02f,1.423148383e-01f,2.125652896e-01f,2.817325568e-01f,3.494641796e-01f,4.154150130e-01f,4.792489867e-01f,5.406408175e-01f,5.992776665e-01f,6.548607339e-01f,7.071067812e-01f,7.557495744e-01f,8.005412409e-01f,8.412535328e-01f,8.776789896e-01f,9.096319954e-01f,9.369497250e-01f,9.594929736e-01f,9.771468660e-01f,9.898214419e-01f,9.974521146e-01f};
__constant__ float S88[88] = {0.000000000e+00f,7.133918320e-02f,1.423148383e-01f,2.125652896e-01f,2.817325568e-01f,3.494641796e-01f,4.154150130e-01f,4.792489867e-01f,5.406408175e-01f,5.992776665e-01f,6.548607339e-01f,7.071067812e-01f,7.557495744e-01f,8.005412409e-01f,8.412535328e-01f,8.776789896e-01f,9.096319954e-01f,9.369497250e-01f,9.594929736e-01f,9.771468660e-01f,9.898214419e-01f,9.974521146e-01f,1.000000000e+00f,9.974521146e-01f,9.898214419e-01f,9.771468660e-01f,9.594929736e-01f,9.369497250e-01f,9.096319954e-01f,8.776789896e-01f,8.412535328e-01f,8.005412409e-01f,7.557495744e-01f,7.071067812e-01f,6.548607339e-01f,5.992776665e-01f,5.406408175e-01f,4.792489867e-01f,4.154150130e-01f,3.494641796e-01f,2.817325568e-01f,2.125652896e-01f,1.423148383e-01f,7.133918320e-02f,5.665538898e-16f,-7.133918320e-02f,-1.423148383e-01f,-2.125652896e-01f,-2.817325568e-01f,-3.494641796e-01f,-4.154150130e-01f,-4.792489867e-01f,-5.406408175e-01f,-5.992776665e-01f,-6.548607339e-01f,-7.071067812e-01f,-7.557495744e-01f,-8.005412409e-01f,-8.412535328e-01f,-8.776789896e-01f,-9.096319954e-01f,-9.369497250e-01f,-9.594929736e-01f,-9.771468660e-01f,-9.898214419e-01f,-9.974521146e-01f,-1.000000000e+00f,-9.974521146e-01f,-9.898214419e-01f,-9.771468660e-01f,-9.594929736e-01f,-9.369497250e-01f,-9.096319954e-01f,-8.776789896e-01f,-8.412535328e-01f,-8.005412409e-01f,-7.557495744e-01f,-7.071067812e-01f,-6.548607339e-01f,-5.992776665e-01f,-5.406408175e-01f,-4.792489867e-01f,-4.154150130e-01f,-3.494641796e-01f,-2.817325568e-01f,-2.125652896e-01f,-1.423148383e-01f,-7.133918320e-02f};
__constant__ float C11[11] = {1.000000000e+00f,8.412535328e-01f,4.154150130e-01f,-1.423148383e-01f,-6.548607339e-01f,-9.594929736e-01f,-9.594929736e-01f,-6.548607339e-01f,-1.423148383e-01f,4.154150130e-01f,8.412535328e-01f};
__constant__ float S11[11] = {0.000000000e+00f,5.406408175e-01f,9.096319954e-01f,9.898214419e-01f,7.557495744e-01f,2.817325568e-01f,-2.817325568e-01f,-7.557495744e-01f,-9.898214419e-01f,-9.096319954e-01f,-5.406408175e-01f};
#define RS 91
struct cf { float x, y; };
__device__ __forceinline__ cf cadd(cf a, cf b) { return {a.x + b.x, a.y + b.y}; }
__device__ __forceinline__ cf csub(cf a, cf b) { return {a.x - b.x, a.y - b.y}; }
template <int SG> __device__ __forceinline__ void dft4(cf &a0, cf &a1, cf &a2, cf &a3) {
    cf t0 = cadd(a0, a2), t1 = csub(a0, a2), t2 = cadd(a1, a3), d = csub(a1, a3);
    cf t3 = {-SG * d.y, SG * d.x};
    a0 = cadd(t0, t2); a2 = csub(t0, t2); a1 = cadd(t1, t3); a3 = csub(t1, t3);
}
template <int SG> __device__ __forceinline__ void dft8(cf *a) {
    const float r = 0.70710678118654752f;
    cf u0 = cadd(a[0], a[4]), u1 = cadd(a[1], a[5]), u2 = cadd(a[2], a[6]), u3 = cadd(a[3], a[7]);
    cf v0 = csub(a[0], a[4]), v1 = csub(a[1], a[5]), v2 = csub(a[2], a[6]), v3 = csub(a[3], a[7]);
    v1 = {r * (v1.x - SG * v1.y), r * (SG * v1.x + v1.y)};
    v2 = {-SG * v2.y, SG * v2.x};
    v3 = {r * (-v3.x - SG * v3.y), r * (SG * v3.x - v3.y)};
    dft4<SG>(u0, u1, u2, u3); dft4<SG>(v0, v1, v2, v3);
    a[0] = u0; a[2] = u1; a[4] = u2; a[6] = u3; a[1] = v0; a[3] = v1; a[5] = v2; a[7] = v3;
}
template <int SG> __device__ __forceinline__ void dft11(cf *a) {
    cf s[5], d[5];
#pragma unroll
    for (int n = 1; n <= 5; ++n) { s[n - 1] = cadd(a[n], a[11 - n]); d[n - 1] = csub(a[n], a[11 - n]); }
    cf x0 = a[0], y0 = a[0];
#pragma unroll
    for (int n = 0; n < 5; ++n) y0 = cadd(y0, s[n]);
#pragma unroll
    for (int k = 1; k <= 5; ++k) {
        float cr = x0.x, ci = x0.y, sr = 0.f, si = 0.f;
#pragma unroll
        for (int n = 1; n <= 5; ++n) {
            const int m = (n * k) % 11;
            cr += s[n - 1].x * C11[m]; ci += s[n - 1].y * C11[m];
            sr += d[n - 1].y * S11[m]; si += d[n - 1].x * S11[m];
        }
        a[k] = {cr - SG * sr, ci + SG * si};
        a[11 - k] = {cr + SG * sr, ci - SG * si};
    }
    a[0] = y0;
}
"""

# Stage 1 (one launch, two block roles interleaved 1:1 so both overlap on every SM):
#   even blocks: rfft2 of one (tile, ci) signal in smem -> z [m*128 + ci][3960] complex
#   odd blocks : row DFT of the weights (kx -> fw) on tensor cores -> T[hi/lo][fw][co*128 + ci][re 32 ky | im 32 ky] fp16
PREP_SRC = CU_MMA + CU_FFT + r"""
#define NSIG 1
#define NTS 256
#define MINB 2
#define RNW 8
#define RPP 1
#define FPR 1
__device__ __forceinline__ cf rotr(cf a, float c, float s) { return {a.x * c - a.y * s, a.x * s + a.y * c}; }
__device__ __forceinline__ unsigned smaddr(const void *p) {
    unsigned a; asm("{.reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t;}" : "=r"(a) : "l"(p)); return a;
}
// row DFT of the weights (kx -> fw), hand-written for Ampere: T[hi/lo][fw][pair][re 32 ky | im 32 ky] as fp16 hi/lo.
// One warp owns one (co, ci) pair at a time: 4 mma column tiles cover its 32 ky rows, mma rows = 16 fw; no smem, no barriers.
// Column n of tile q is ky = 8*(n>>1) + 2q + (n&1), so a thread ends up with ky 8t .. 8t+7 of two fw rows: 16-byte stores.
static __device__ __forceinline__ float rn11r(float v) {
    unsigned u = f2u(v);
    u = (u + 0x1000u) & 0xFFFFE000u;
    return u2f(u);
}
static __device__ __forceinline__ void rows_body(const int bxr, const float4 *__restrict__ wt, const uint4 *__restrict__ tab, uint4 *__restrict__ tp)
{
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
    const int TLO = 45 * 16384 * 8;
    for (int it = 0; it < RPP; ++it) {
        const int pair = (bxr * RNW + w) * RPP + it;
        unsigned Bh[4][4], Bl[4][4];                      // [column tile][kx fragment]: kx 8t + 2j (low half), 8t + 2j + 1 (high half)
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int ky = 8 * (g >> 1) + 2 * q + (g & 1);
            const float4 a = wt[(pair * 32 + ky) * 8 + 2 * t], b = wt[(pair * 32 + ky) * 8 + 2 * t + 1];
            const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float h0 = rn11r(v[2 * j]), h1 = rn11r(v[2 * j + 1]);
                CVT2(Bh[q][j], h1, h0);
                CVT2(Bl[q][j], v[2 * j + 1] - h1, v[2 * j] - h0);
            }
        }
        #pragma unroll
        for (int mtl = 0; mtl < 3; ++mtl) {               // fw = 16*mtl + g (+8)
            #pragma unroll
            for (int cs = 0; cs < 2; ++cs) {              // cos -> re, -sin -> im
                const uint4 *ta = tab + ((mtl * 2 + cs) * 4) * 32 + lane;
                const uint4 ah0 = ta[0], ah1 = ta[32], al0 = ta[64], al1 = ta[96];
                float O[2][8];                            // [fw row g / g+8][ky 8t .. 8t+7]
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    float acc[4] = {0.f, 0.f, 0.f, 0.f};
                    MMA(acc, ah0.x, ah0.y, ah0.z, ah0.w, Bl[q][0], Bl[q][1]);
                    MMA(acc, ah1.x, ah1.y, ah1.z, ah1.w, Bl[q][2], Bl[q][3]);
                    MMA(acc, al0.x, al0.y, al0.z, al0.w, Bh[q][0], Bh[q][1]);
                    MMA(acc, al1.x, al1.y, al1.z, al1.w, Bh[q][2], Bh[q][3]);
                    MMA(acc, ah0.x, ah0.y, ah0.z, ah0.w, Bh[q][0], Bh[q][1]);
                    MMA(acc, ah1.x, ah1.y, ah1.z, ah1.w, Bh[q][2], Bh[q][3]);
                    O[0][2 * q] = acc[0]; O[0][2 * q + 1] = acc[1]; O[1][2 * q] = acc[2]; O[1][2 * q + 1] = acc[3];
                }
                #pragma unroll
                for (int row = 0; row < 2; ++row) {
                    const int fw = 16 * mtl + g + 8 * row;
                    uint4 hi, lo;
                    unsigned ph[4], pl[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float h0 = rn11r(O[row][2 * j]), h1 = rn11r(O[row][2 * j + 1]);
                        CVT2(ph[j], h1, h0);
                        CVT2(pl[j], O[row][2 * j + 1] - h1, O[row][2 * j] - h0);
                    }
                    hi.x = ph[0]; hi.y = ph[1]; hi.z = ph[2]; hi.w = ph[3];
                    lo.x = pl[0]; lo.y = pl[1]; lo.z = pl[2]; lo.w = pl[3];
                    if (fw < 45) {
                        const int o = (fw * 16384 + pair) * 8 + cs * 4 + t;
                        tp[o] = hi;
                        tp[TLO + o] = lo;
                    }
                }
            }
        }
    }
}
static __device__ __forceinline__ void fwd_body(const int bxf, const float *__restrict__ x, float2 *__restrict__ z) {
    extern __shared__ float sm[];
    float *twc = sm + NSIG * 88 * RS, *tws = twc + 88;
    const int t = bxf / (128 / NSIG), ci0 = (bxf % (128 / NSIG)) * NSIG, Y0 = 57 * (t >> 2), X0 = 57 * (t & 3);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (tid < 88) { twc[tid] = C88[tid]; tws[tid] = -S88[tid]; }
    for (int r = warp; r < NSIG * 88; r += NTS / 32) {
        const int s = r / 88, Y = Y0 + r % 88;
        const float *src = x + ((size_t)(ci0 + s) * 256 + min(Y, 255)) * 256;
        for (int j = lane; j < 88; j += 32) {
            const int X = X0 + j;
            const int ok = (Y < 256 && X < 256) ? 4 : 0;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(smaddr(sm + r * RS + j)), "l"(src + min(X, 255)), "r"(ok));
        }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    // rows: pair row y with row y+44 of the same signal as one complex signal (re in row a, im in row b)
    for (int idx = tid; idx < NSIG * 44 * 11; idx += NTS) {
        const int p = idx % (NSIG * 44), n2 = idx / (NSIG * 44), s = p / 44, y = p % 44;
        float *ra = sm + (s * 88 + y) * RS, *rb = ra + 44 * RS;
        cf a[8];
#pragma unroll
        for (int n1 = 0; n1 < 8; ++n1) a[n1] = cf{ra[11 * n1 + n2], rb[11 * n1 + n2]};
        dft8<-1>(a);
#pragma unroll
        for (int k1 = 0; k1 < 8; ++k1) {
            cf v = a[k1];
            if (k1) { const int m = (n2 * k1) % 88; v = rotr(v, twc[m], tws[m]); }
            ra[11 * k1 + n2] = v.x; rb[11 * k1 + n2] = v.y;
        }
    }
    __syncthreads();
    for (int idx = tid; idx < NSIG * 44 * 8; idx += NTS) {
        const int p = idx >> 3, k1 = idx & 7, s = p / 44, y = p % 44;
        float *ra = sm + (s * 88 + y) * RS, *rb = ra + 44 * RS;
        cf b[11];
#pragma unroll
        for (int n2 = 0; n2 < 11; ++n2) b[n2] = cf{ra[11 * k1 + n2], rb[11 * k1 + n2]};
        dft11<-1>(b);
        __syncwarp();
#pragma unroll
        for (int k2 = 0; k2 < 11; ++k2) { ra[k1 + 8 * k2] = b[k2].x; rb[k1 + 8 * k2] = b[k2].y; }
    }
    __syncthreads();
    for (int p = warp; p < NSIG * 44; p += NTS / 32) {
        const int s = p / 44, y = p % 44;
        float *ra = sm + (s * 88 + y) * RS, *rb = ra + 44 * RS;
        cf A[2], B[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int k = lane + 32 * h;
            if (k < 45) {
                const int m = (88 - k) % 88;
                const cf zk{ra[k], rb[k]}, zm{ra[m], rb[m]};
                A[h] = cf{0.5f * (zk.x + zm.x), 0.5f * (zk.y - zm.y)};
                const cf d{zk.x - zm.x, zk.y + zm.y};
                B[h] = cf{0.5f * d.y, -0.5f * d.x};
            }
        }
        __syncwarp();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int k = lane + 32 * h;
            if (k < 45) { ra[2 * k] = A[h].x; ra[2 * k + 1] = A[h].y; rb[2 * k] = B[h].x; rb[2 * k + 1] = B[h].y; }
        }
        __syncwarp();
    }
    __syncthreads();
    // columns: complex 88-point along y for each (signal, fw)
    for (int idx = tid; idx < NSIG * 45 * 11; idx += NTS) {
        const int fw = idx % 45, r = idx / 45, s = r % NSIG, n2 = r / NSIG;
        float *col = sm + s * 88 * RS + 2 * fw;
        cf a[8];
#pragma unroll
        for (int n1 = 0; n1 < 8; ++n1) { const int n = 11 * n1 + n2; a[n1] = cf{col[n * RS], col[n * RS + 1]}; }
        dft8<-1>(a);
#pragma unroll
        for (int k1 = 0; k1 < 8; ++k1) {
            cf v = a[k1];
            if (k1) { const int m = (n2 * k1) % 88; v = rotr(v, twc[m], tws[m]); }
            const int sl = 11 * k1 + n2;
            col[sl * RS] = v.x; col[sl * RS + 1] = v.y;
        }
    }
    __syncthreads();
    for (int idx = tid; idx < NSIG * 45 * 8; idx += NTS) {
        const int fw = idx % 45, r = idx / 45, s = r % NSIG, k1 = r / NSIG;
        const float *col = sm + s * 88 * RS + 2 * fw;
        cf b[11];
#pragma unroll
        for (int n2 = 0; n2 < 11; ++n2) { const int sl = 11 * k1 + n2; b[n2] = cf{col[sl * RS], col[sl * RS + 1]}; }
        dft11<-1>(b);
        float2 *dst = z + (size_t)(t * 128 + ci0 + s) * 3960 + fw;
#pragma unroll
        for (int k2 = 0; k2 < 11; ++k2) dst[(k1 + 8 * k2) * 45] = make_float2(b[k2].x, b[k2].y);
    }
    __syncthreads();
}


extern "C" __global__ void __launch_bounds__(NTS, MINB) combo(const float *__restrict__ x, float2 *__restrict__ z, const float4 *__restrict__ wt,
                                                              const uint4 *__restrict__ tab, uint4 *__restrict__ tp) {
    const int g = blockIdx.x / (FPR + 1), p = blockIdx.x % (FPR + 1);
    if (p < FPR) fwd_body(g * FPR + p, x, z);
    else rows_body(g, wt, tab, tp);
}
"""

# Stage 3: fused weight-spectrum (column DFT of T) + channel mixer, split-fp16 hi/lo operands on mma.m16n8k16.
FUSED_SRC = CU_MMA + r"""
#define RNADD 0x1000u
// fused weight-spectrum + channel mixer, hand-written for Ampere (mma m16n8k16). One CTA = (fw, 8 base rows -> 32 fh, 32 co), 16 warps.
// Per chunk of 16 ci: every warp builds the W tile of one co (all 32 rows) with tensor cores, splits it to fp16 hi/lo and parks
// it in smem; after one barrier every warp mixes two rows (b or b+44, and the mirror) for all 16 co / 16 tiles.
static __device__ __forceinline__ float rn11(float v) {
    // round to 11 significant bits (exactly representable in fp16), still float
    unsigned u = f2u(v);
    u = (u + RNADD) & 0xFFFFE000u;
    return u2f(u);
}
#define MMX(...) MMA(__VA_ARGS__)
#define AR(v) (v).x, (v).y, (v).z, (v).w
#define XB(v, mt) ((mt) ? (v).z : (v).x), ((mt) ? (v).w : (v).y)

// W tile of chunk cw for co = cb*16 + w: ci = cw*16 + kh*8 + (0..7), 8 base rows b = fb*8 + g -> the four rows b, 88-b, b+44, 44-b.
// mma rows = [cos rows ; sin rows] of the 8 base rows, K = the 16 even or the 16 odd ky (rows b and b+44 differ by the sign of
// the odd-ky part, rows b and 88-b by the sign of the sin part), so 12 mma per 8 pairs build 32 spectrum rows.
static __device__ __forceinline__ void wgen(const uint4 *__restrict__ tp, const uint4 *tabp, unsigned *sbw, int pbase, int cw,
                                            int lane, int w)
{
    const int g = lane >> 2, t = lane & 3;
    const int TLO = 45 * 16384 * 8;
    const int wslot = (w & 7) * 4 + t, wreg = w >> 3;
    #pragma unroll
    for (int kh = 0; kh < 2; ++kh) {
        const int p8 = (pbase + cw * 16 + kh * 8 + g) * 8 + t;
        uint4 tv[2][2];                                   // [hi / lo][re / im] : ky 8t .. 8t+7 as 4 fp16 pairs
        tv[0][0] = tp[p8]; tv[1][0] = tp[TLO + p8]; tv[0][1] = tp[p8 + 4]; tv[1][1] = tp[TLO + p8 + 4];
        unsigned B[2][2][2][2];                           // [hi / lo][re / im][even / odd ky][frag]
        #pragma unroll
        for (int hl = 0; hl < 2; ++hl)
            #pragma unroll
            for (int pt = 0; pt < 2; ++pt) {
                const uint4 v = tv[hl][pt];
                B[hl][pt][0][0] = (v.x & 0xffffu) | (v.y << 16);
                B[hl][pt][0][1] = (v.z & 0xffffu) | (v.w << 16);
                B[hl][pt][1][0] = (v.x >> 16) | (v.y & 0xffff0000u);
                B[hl][pt][1][1] = (v.z >> 16) | (v.w & 0xffff0000u);
            }
        uint4 TA[2][2];                                   // [even / odd ky][hi / lo] : rows = cos(8 base rows) ; sin(8 base rows)
        #pragma unroll
        for (int e = 0; e < 2; ++e)
            #pragma unroll
            for (int hl = 0; hl < 2; ++hl) TA[e][hl] = tabp[(e * 2 + hl) * 32];
        float P[2][2][4];                                 // [even / odd][re / im][cos c0 c1, sin c0 c1]
        #pragma unroll
        for (int e = 0; e < 2; ++e)
            #pragma unroll
            for (int pt = 0; pt < 2; ++pt)
                #pragma unroll
                for (int i = 0; i < 4; ++i) P[e][pt][i] = 0.f;
#define WS(e, pt, ahl, bhl) MMX(P[e][pt], AR(TA[e][ahl]), B[bhl][pt][e][0], B[bhl][pt][e][1])
        WS(0, 0, 0, 1); WS(1, 0, 0, 1); WS(0, 1, 0, 1); WS(1, 1, 0, 1);
        WS(0, 0, 1, 0); WS(1, 0, 1, 0); WS(0, 1, 1, 0); WS(1, 1, 1, 0);
        WS(0, 0, 0, 0); WS(1, 0, 0, 0); WS(0, 1, 0, 0); WS(1, 1, 0, 0);
#undef WS
        #pragma unroll
        for (int hs = 0; hs < 2; ++hs) {                  // rows (b, 88-b) : even + odd ; rows (b+44, 44-b) : even - odd
            float Wv[4][2];                               // [W re, W im, mirror re, mirror im][ci 2t, 2t+1]
            #pragma unroll
            for (int x = 0; x < 2; ++x) {
                const float cr = hs ? P[0][0][x] - P[1][0][x] : P[0][0][x] + P[1][0][x];
                const float sr = hs ? P[0][0][2 + x] - P[1][0][2 + x] : P[0][0][2 + x] + P[1][0][2 + x];
                const float ci = hs ? P[0][1][x] - P[1][1][x] : P[0][1][x] + P[1][1][x];
                const float si = hs ? P[0][1][2 + x] - P[1][1][2 + x] : P[0][1][2 + x] + P[1][1][2 + x];
                Wv[0][x] = cr + si; Wv[1][x] = ci - sr; Wv[2][x] = cr - si; Wv[3][x] = ci + sr;
            }
            #pragma unroll
            for (int ts = 0; ts < 4; ++ts) {
                const float v0 = Wv[ts][0], v1 = Wv[ts][1];
                const float h0 = rn11(v0), h1 = rn11(v1);
                unsigned ph, pl;
                CVT2(ph, h1, h0);
                CVT2(pl, v1 - h1, v0 - h0);
                const int r = g + 8 * hs;                 // = the warp that mixes these two rows
                const int o = ((r * 8 + ts * 2) * 4 + wreg + 2 * kh) * 32 + ((wslot + 4 * r) & 31);
                sbw[o] = ph;
                sbw[o + 128] = pl;
            }
        }
    }
}

// mixer of chunk c for one spectrum row and one block of 16 co (all 16 tiles); the W fragments are read one tensor at a time
#define ALOAD(q) { A[0] = sr[(q) * 128]; A[1] = sr[(q) * 128 + 32]; A[2] = sr[(q) * 128 + 64]; A[3] = sr[(q) * 128 + 96]; }
#define MS(dst, xv) MMX(dst[0], A[0], A[1], A[2], A[3], XB(xv, 0)); MMX(dst[1], A[0], A[1], A[2], A[3], XB(xv, 1))
static __device__ __forceinline__ void mix(const uint4 xrh, const uint4 xrl, const uint4 xih, const uint4 xil, const uint4 nxh, const uint4 nxl,
                                           const unsigned *sr, float (&yr)[2][4], float (&yi)[2][4])
{
    unsigned A[4];
    float lr[2][4], li[2][4];
    #pragma unroll
    for (int b = 0; b < 2; ++b)
        #pragma unroll
        for (int i = 0; i < 4; ++i) { lr[b][i] = 0.f; li[b][i] = 0.f; }
    // yr += wr*xr + wi*xi ; yi += wr*xi - wi*xr  (hi*hi on the long accumulators, hi*lo + lo*hi on the per-chunk ones)
    ALOAD(0); MS(yr, xrh); MS(yi, xih); MS(lr, xrl); MS(li, xil);       // W re hi
    ALOAD(2); MS(yr, xih); MS(yi, nxh); MS(lr, xil); MS(li, nxl);       // W im hi
    ALOAD(1); MS(lr, xrh); MS(li, xih);                                 // W re lo
    ALOAD(3); MS(lr, xih); MS(li, nxh);                                 // W im lo
    #pragma unroll
    for (int b = 0; b < 2; ++b)
        #pragma unroll
        for (int i = 0; i < 4; ++i) { yr[b][i] += lr[b][i]; yi[b][i] += li[b][i]; }
}

static __device__ __forceinline__ void mixrow(const uint4 *__restrict__ xp, const unsigned *sb, float (&yr)[2][2][4], float (&yi)[2][2][4])
{
    const uint4 xrh = xp[0], xrl = xp[32], xih = xp[64], xil = xp[96];
    uint4 nxh, nxl;
    nxh.x = xrh.x ^ 0x80008000u; nxh.y = xrh.y ^ 0x80008000u; nxh.z = xrh.z ^ 0x80008000u; nxh.w = xrh.w ^ 0x80008000u;
    nxl.x = xrl.x ^ 0x80008000u; nxl.y = xrl.y ^ 0x80008000u; nxl.z = xrl.z ^ 0x80008000u; nxl.w = xrl.w ^ 0x80008000u;
    mix(xrh, xrl, xih, xil, nxh, nxl, sb, yr[0], yi[0]);
    mix(xrh, xrl, xih, xil, nxh, nxl, sb + 16384, yr[1], yi[1]);
}

#define DOMIX(c) { const unsigned *sb = sm + ((lane + 4 * w) & 31) + w * 1024; \
                   if (fv) mixrow(xq + ((f0 * 45 + fw) * 1024 + (c) * 128 + lane), sb, yr[0], yi[0]); \
                   if (pv) mixrow(xq + ((f1 * 45 + fw) * 1024 + (c) * 128 + lane), sb + 512, yr[1], yi[1]); }
#define DOW(cw) { wgen(tp, tabp, sm, pbase, cw, lane, w); wgen(tp, tabp, sm + 16384, pbase + 16 * 128, cw, lane, w); }

extern "C" __global__ void __launch_bounds__(512, 1) fused_cu(const uint4 *__restrict__ xq, const uint4 *__restrict__ tp, const uint4 *__restrict__ tab, float2 *__restrict__ yf)
{
    extern __shared__ unsigned sm[];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
    const int id = blockIdx.x, fb = id % 3, cb = (id / 3) & 3, fw = id / 12;      // cb: block of 32 co
    const int bq = fb * 8 + (w & 7), hs = w >> 3;        // mixer role: base row and which pair of its four rows
    const bool fv = bq <= 22, pv = bq >= 1 && bq <= 21;
    const int f0 = hs ? bq + 44 : bq, f1 = hs ? 44 - bq : 88 - bq;
    const int pbase = fw * 16384 + (cb * 32 + w) * 128;
    // cos/sin fragments of this half-row block live in smem (keeps them out of the registers during the mixer)
    uint4 *tsm = (uint4 *)(sm + 32768);
    if (w == 0) {
        #pragma unroll
        for (int a = 0; a < 4; ++a) tsm[a * 32 + lane] = tab[(fb * 4 + a) * 32 + lane];
    }
    __syncthreads();
    const uint4 *tabp = tsm + lane;
    float yr[2][2][2][4], yi[2][2][2][4];                 // [row / mirror][co block][m tile][frag]
    #pragma unroll
    for (int a = 0; a < 2; ++a)
        #pragma unroll
        for (int b = 0; b < 2; ++b)
            #pragma unroll
            for (int k = 0; k < 2; ++k)
                #pragma unroll
                for (int i = 0; i < 4; ++i) { yr[a][b][k][i] = 0.f; yi[a][b][k][i] = 0.f; }
    // both co blocks of a chunk share one X load per row; the W tile (2 x 64 KB) is single-buffered: two barriers per chunk
    for (int c = 0; c < 8; ++c) {
        DOW(c);
        __syncthreads();
        DOMIX(c);
        __syncthreads();
    }
    #pragma unroll
    for (int fp = 0; fp < 2; ++fp) {
        if (fp ? pv : fv) {
            const int f = (fp ? f1 : f0) * 45 + fw;
            #pragma unroll
            for (int k = 0; k < 2; ++k)
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int m = mt * 8 + 2 * t + (i & 1), co = cb * 32 + k * 16 + g + 8 * (i >> 1);
                        float2 v;
                        v.x = yr[fp][k][mt][i];
                        v.y = yi[fp][k][mt][i];
                        yf[(f * 16 + m) * 128 + co] = v;
                    }
                }
        }
    }
}
"""

# Stage 4: inverse 2D FFT per (tile, co) and crop to the 57x57 valid block of the output.
INV_SRC = CU_FFT + r"""
template <int SG> __device__ __forceinline__ cf rot(cf a, int m) {
    float c = C88[m % 88], s = SG * S88[m % 88];
    return {a.x * c - a.y * s, a.x * s + a.y * c};
}
// 88 = 8 x 11: n = 11 n1 + n2, k = k1 + 8 k2. ld(n) -> cf, st(k, cf).
template <int SG, int K1LO = 0, int K1HI = 8, class LD, class ST> __device__ __forceinline__ void fft88(LD ld, ST st) {
    cf A[88];
#pragma unroll
    for (int n2 = 0; n2 < 11; ++n2) {
        cf a[8];
#pragma unroll
        for (int n1 = 0; n1 < 8; ++n1) a[n1] = ld(11 * n1 + n2);
        dft8<SG>(a);
#pragma unroll
        for (int k1 = 0; k1 < 8; ++k1) A[n2 * 8 + k1] = (n2 * k1 == 0) ? a[k1] : rot<SG>(a[k1], n2 * k1);
    }
#pragma unroll
    for (int k1 = K1LO; k1 < K1HI; ++k1) {
        cf b[11];
#pragma unroll
        for (int n2 = 0; n2 < 11; ++n2) b[n2] = A[n2 * 8 + k1];
        dft11<SG>(b);
#pragma unroll
        for (int k2 = 0; k2 < 11; ++k2) st(k1 + 8 * k2, b[k2]);
    }
}
// yf [3960][16][128] complex -> out [128,225,225]: crop 57x57 per tile. Block = (tile, 4 co).
extern "C" __global__ void __launch_bounds__(256, 1) inv88(const float2 *__restrict__ yf, float *__restrict__ out) {
    extern __shared__ float sm[];
    const int t = blockIdx.x >> 5, co0 = (blockIdx.x & 31) * 4, Y0 = 57 * (t >> 2), X0 = 57 * (t & 3);
    if (threadIdx.x < 180) {
        const int s = threadIdx.x & 3, fw = threadIdx.x >> 2;
        const float2 *src = yf + (size_t)fw * 2048 + t * 128 + co0 + s;
        float *col = sm + s * 57 * RS + 2 * fw;
        fft88<1>([&](int n) { float2 v = src[(size_t)n * 45 * 2048]; return cf{v.x, v.y}; },
                 [&](int k, cf v) { if (k < 57) { col[k * RS] = v.x; col[k * RS + 1] = v.y; } });
    }
    __syncthreads();
    if (threadIdx.x < 228) {
        float *row = sm + threadIdx.x * RS;
        fft88<1>([&](int n) {
                     const int m = n <= 44 ? n : 88 - n;
                     cf v{row[2 * m], row[2 * m + 1]};
                     if (n == 0 || n == 44) v.y = 0.f; else if (n > 44) v.y = -v.y;
                     return v; },
                 [&](int k, cf v) { if (k < 57) row[k] = v.x * (1.f / 7744.f); });
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 4 * 3249; i += 256) {
        int s = i / 3249, r = i % 3249, oy = r / 57, ox = r % 57, Y = Y0 + oy, X = X0 + ox;
        if (Y < 225 && X < 225) out[((co0 + s) * 225 + Y) * 225 + X] = sm[(s * 57 + oy) * RS + ox];
    }
}

"""


# Stage 1 constants: mma A fragments (rows = 16 fw) of cos / -sin(2 pi fw kx / 88) as fp16 hi/lo,
# [fw tile][cos, -sin][hi, lo][k block][lane][reg][2] in the kx order the row-DFT blocks read the weights.
@lru_cache(None)
def cu_rows_tables(device):
    m = torch.arange(3).view(3, 1, 1, 1, 1, 1)
    cs = torch.arange(2).view(1, 2, 1, 1, 1, 1)
    kb = torch.arange(2).view(1, 1, 2, 1, 1, 1)
    lane = torch.arange(32).view(1, 1, 1, 32, 1, 1)
    reg = torch.arange(4).view(1, 1, 1, 1, 4, 1)
    lo = torch.arange(2).view(1, 1, 1, 1, 1, 2)
    fw = 16 * m + lane // 4 + 8 * (reg % 2)
    k = 16 * kb + 2 * (lane % 4) + 8 * (reg // 2) + lo
    kx = 8 * (k % 8 // 2) + 4 * (k // 16) + 2 * (k // 8 % 2) + k % 2
    th = 2 * torch.pi * fw.double() * kx.double() / 88 + 0 * cs
    v = torch.where((cs == 1).expand_as(th), -th.sin(), th.cos())
    v = torch.where((fw < 45).expand_as(th), v, torch.zeros_like(v)).float()
    h = v.half()
    return torch.stack((h, (v - h.float()).half()), 2).contiguous().to(device)


# Stage 3 constants: mma A fragments (rows = cos | sin of 8 base rows fh) of 16*trig(2 pi fh ky / 88) as fp16 hi/lo,
# [fh block][even / odd ky][hi, lo][lane][reg][2].
@lru_cache(None)
def cu_tables(device):
    b = torch.arange(3).view(3, 1, 1, 1, 1)
    pe = torch.arange(2).view(1, 2, 1, 1, 1)
    lane = torch.arange(32).view(1, 1, 32, 1, 1)
    reg = torch.arange(4).view(1, 1, 1, 4, 1)
    lo = torch.arange(2).view(1, 1, 1, 1, 2)
    j = 2 * (lane % 4) + 8 * (reg // 2) + lo
    ky = 8 * (j % 8 // 2) + 4 * (j // 8) + 2 * (j % 2) + pe
    th = 2 * torch.pi * (b * 8 + lane // 4).double() * ky.double() / 88
    v = (16 * torch.where((reg % 2 == 1).expand_as(th), th.sin(), th.cos())).float()
    h = v.half()
    return torch.stack((h, (v - h.float()).half()), 2).contiguous().to(device)


CUDA_SOURCES = {"combo": PREP_SRC, "fused_cu": FUSED_SRC, "inv88": INV_SRC}


@lru_cache(None)
def cuda_kernel(name):
    return cuda_runtime().build(CUDA_SOURCES[name], name)


# Stage 2 (Gluon): z -> "XF8" fp16 planes of X/16 (p = re-hi, re-lo, im-hi, im-lo) in the mixer's mma-fragment order:
#   offset(f, m, ci, p) = f*8192 + (ci//16)*1024 + p*256 + (m%8)*32 + ((ci%8)//2)*8 + (m//8)*4 + ((ci%16)//8)*2 + ci%2
@g.constexpr_function
def _xt_layout():
    return gl.DistributedLinearLayout([[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 4, 0], [0, 0, 8, 0]], [[0, 1, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]], [[0, 0, 1, 0], [0, 0, 2, 0]], [], [8, 4, 16, 8])


@g.jit
def _split16(v):
    h = v.to(gl.float16)
    return (h, (v - h.to(gl.float32)).to(gl.float16))


@g.jit
def transpose_xf8(X, Y, F: gl.constexpr):
    src: gl.constexpr = gl.BlockedLayout([1, 1, 2, 1], [1, 1, 8, 4], [2, 2, 1, 1], [2, 3, 1, 0])
    dst: gl.constexpr = _xt_layout()
    c = gl.program_id(0)
    f0 = gl.program_id(1) * 16
    sg = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, src))))
    st = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, src))))
    sf = f0 + gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, src))))
    se = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, src))))
    row = (sg * 128)[:, None, None, None] + (st * 2)[None, :, None, None] + (se // 4 * 1024 + se // 2 % 2 * 8 + se % 2)[None, None, None, :] + c * 16
    v = gl.load(X + row * F + sf[None, None, :, None], sf[None, None, :, None] < F, other=0)
    v = gl.convert_layout(v, dst)
    rh, rl = _split16(v.to(gl.uint32).to(gl.float32, bitcast=True) * 0.0625)
    ih, il = _split16((v >> 32).to(gl.uint32).to(gl.float32, bitcast=True) * 0.0625)
    dg = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, dst))))
    dt = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, dst))))
    df = f0 + gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, dst))))
    de = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, dst))))
    o = (dg * 32)[:, None, None, None] + (dt * 8)[None, :, None, None] + (df * 8192)[None, None, :, None] + de[None, None, None, :] + c * 1024
    msk = df[None, None, :, None] < F
    gl.store(Y + o, rh, msk)
    gl.store(Y + o + 256, rl, msk)
    gl.store(Y + o + 512, ih, msk)
    gl.store(Y + o + 768, il, msk)


def spectral_conv(x, w, out):
    rt = cuda_runtime()
    z = torch.empty((2048, 3960), device=x.device, dtype=torch.complex64)
    tp = torch.empty((2, 45, 16384, 64), device=w.device, dtype=torch.float16)
    rt.launch(cuda_kernel("combo"), 4096, 256, (88 * 91 + 176) * 4, pointers(x, z, w, cu_rows_tables(w.device), tp))
    xq = torch.empty((3960 * 8192,), device=x.device, dtype=torch.float16)
    transpose_xf8[(8, 248)](z.view(torch.int64), xq, 3960, num_warps=4)
    yf = torch.empty((3960, 16, 128), device=x.device, dtype=torch.complex64)
    rt.launch(cuda_kernel("fused_cu"), 540, 512, 133120, pointers(xq, tp, cu_tables(xq.device), yf))
    rt.launch(cuda_kernel("inv88"), 512, 256, 4 * 57 * 91 * 4, pointers(yf, out))
    return out


# ---------------------------------------------------------------------------------------------------------------
def custom_kernel(data):
    x, w, out = data
    channels, size, width = x.shape[1], x.shape[-1], w.shape[-1]
    if tuple(x.shape) == (1, 128, 256, 256) and width == 32:
        return spectral_conv(x, w, out)
    if (channels, size) == (64, 128) and width in (8, 16):
        return overlap_conv(data, 48)
    if (channels, size, width) == (128, 256, 16):
        return overlap_conv(data, 80)
    exact_conv(x, w, out)
    return out
