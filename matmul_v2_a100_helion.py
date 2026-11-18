import hashlib
import re
from importlib.metadata import version
from pathlib import Path

import helion
import helion.language as hl
import torch
from triton import knobs


KERNEL_OPTIONS = dict(backend="triton", static_shapes=True, autotune_effort="none",
                      dot_precision="ieee", fast_math=False, print_output_code=False)


@helion.kernel(config=helion.Config(block_sizes=[64, 64, 32], num_warps=4,
                                  num_stages=3, pid_type="flat", l2_groupings=[8]),
               **KERNEL_OPTIONS)
def matmul_tiles(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile([m, n]):
        acc = hl.zeros([rows, cols], dtype=torch.float32)
        for ks in hl.tile(k):
            acc = hl.dot(a[rows, ks], b[ks, cols], acc, out_dtype=torch.float32)
        c[rows, cols] = acc.to(torch.float16)
    return c


@helion.kernel(config=helion.Config(block_sizes=[], num_warps=8, num_stages=4,
                                  pid_type="persistent_interleaved", num_sm_multiplier=1,
                                  range_flattens=[True, None, None, None]),
               **KERNEL_OPTIONS)
def matmul_ranked_tiles(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    for tile in hl.tile(640, block_size=1):
        task = tile.begin
        tail = task // 540
        tm = (1 - tail) * (task % 54 // 2) + tail * (27 + (task - 540) % 5)
        tn = (1 - tail) * (2 * (task // 54) + task % 2) + tail * ((task - 540) // 5)
        for rows, cols in hl.tile([128, 256], block_size=[128, 256]):
            r = rows.index + tm * 128
            s = cols.index + tn * 256
            acc = hl.zeros([rows, cols], dtype=torch.float32)
            for ks in hl.tile(4096, block_size=64):
                bo = ks.index[:, None] * 5120 + s[None, :]
                bo = hl.inline_triton('tl.max_contiguous(tl.multiple_of({0}, [1, 256]), [1, 256])', [bo], output_like=bo)
                acc = hl.dot(hl.load(a, [r, ks]), hl.load(b, [bo]), acc, out_dtype=torch.float32)
            hl.store(c, [r, s], acc.to(torch.float16))
    return c


SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def ptx_digest(text):
    name = re.search(r"\.visible\s+\.entry\s+(\w+)", text)[1]
    text = re.split(r"(?m)^\s*\.(?:file|section)\b", text, maxsplit=1)[0].replace(name, "KERNEL")
    lines = [line.split("//", 1)[0].strip() for line in text.splitlines()]
    text = " ".join(" ".join(line.split()) for line in lines if line and not line.startswith(".loc"))
    return hashlib.sha256(text.encode()).hexdigest()


def warp_output_ptx():
    code = """
shr.u32 %wo0, %r1, 5; shl.b32 %wo1, %wo0, 11;
add.s32 %wo1, %wo1, 147456; add.s32 %wo1, %wo1, %r127;
shr.u32 %wo2, %r7, 2; shl.b32 %wo3, %wo2, 2;
and.b32 %wo4, %r7, 3; xor.b32 %wo4, %wo4, %wo3;
shl.b32 %wo4, %wo4, 2; shl.b32 %wo2, %wo2, 7;
add.s32 %wo4, %wo4, %wo2; shr.u32 %wo5, %r7, 3;
and.b32 %wo6, %r7, 7; shl.b32 %wo6, %wo6, 4;
shl.b32 %wo7, %wo5, 4; xor.b32 %wo7, %wo7, %wo6;
shl.b32 %wo8, %wo5, 7; add.s32 %wo7, %wo7, %wo8;
and.b32 %wo8, %wo0, 4; shl.b32 %wo8, %wo8, 4;
add.s32 %wo8, %wo8, %r545; add.s32 %wo8, %wo8, %wo5;
and.b32 %wo9, %wo0, 3; shl.b32 %wo9, %wo9, 7;
and.b32 %wo10, %r543, -256; shl.b32 %wo10, %wo10, 1;
add.s32 %wo9, %wo9, %wo10; add.s32 %wo9, %wo9, %wo6;
mad.lo.s32 %wo9, %wo8, 10240, %wo9; cvt.u64.u32 %wod0, %wo9;
add.s64 %wod0, %wod0, %rd3;
"""
    for m in range(4):
        for n in range(8):
            r = 553 + 32 * m + 4 * n
            code += f"""
xor.b32 %wo10, %wo4, {16*n}; add.s32 %wo10, %wo10, %wo1;
cvt.rn.f16x2.f32 %wo11, %r{r+1}, %r{r}; st.shared.b32 [%wo10], %wo11;
cvt.rn.f16x2.f32 %wo11, %r{r+3}, %r{r+2}; st.shared.b32 [%wo10+1024], %wo11;
"""
        code += "bar.warp.sync 0xffffffff;\n"
        for j in range(4):
            code += f"""
xor.b32 %wo10, %wo7, {(j&1)*64}; add.s32 %wo10, %wo10, %wo1;
ld.shared.v4.b32 {{%wo11, %wo12, %wo13, %wo14}}, [%wo10+{j*512}];
st.global.v4.b32 [%wod0+{m*163840+j*40960}], {{%wo11, %wo12, %wo13, %wo14}};
"""
        code += "bar.warp.sync 0xffffffff;\n"
    return code


def lower_ranked_ptx(text):
    if ptx_digest(text) != "a8d2cec1b8c2a5e4ea3b288197f48227fbe4f8aa6e84917ff5c132e760cb1fb0":
        raise RuntimeError("Unexpected native PTX")
    text = text.replace("%r<690>", "%r<693>", 1).replace("%p<26>", "%p<27>", 1)
    text = text.replace("\t.reg .b64 \t%rd<85>;", "\t.reg .b64 \t%rd<85>;\n"
                        "\t.reg .b32 %dw<8>;\n\t.reg .b32 %wo<16>;\n\t.reg .b64 %wod<1>;", 1)
    start = text.index("\tadd.s32 \t%r384,")
    end = text.index("\t// begin inline asm", start)
    stage = text[start:end]
    advance = stage[:stage.index("\t.loc")]
    address = "\n".join(line for line in stage.splitlines()
                        if line.lstrip().startswith(("shl.", "add.")) and "%r384" not in line) + "\n"
    a = re.findall(r"(?m)^\tldmatrix\.sync\.aligned\.m8n8\.x4\.shared\.b16[^\n]+", stage)
    pattern = (r"ldmatrix\.sync\.aligned\.m8n8\.x4\.trans\.shared\.b16 "
               r"\{(%r\d+), (%r\d+), (%r\d+), (%r\d+)\}, \[(%r39[23])(?:\+(\d+))?\];")
    b = {(m[4], int(m[5] or 0)): m[:4] for m in re.findall(pattern, stage)}
    quarters = []
    for q in range(4):
        loads = a[4*q:4*q+4]
        first = int(b["%r392", (q//2)*16384][2*(q%2)][2:])
        for j, pointer in enumerate(("%r392", "%r393", "%dw3", "%dw4")):
            regs = ", ".join(f"%r{first+4*j+t}" for t in range(4))
            loads.append(f"\tldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{{regs}}}, [{pointer}+{q*8192}];")
        quarters.append("\n".join(loads) + "\n")
    text = text[:start] + text[end:]
    start = text.index("\tadd.s32 \t%r394,")
    end = text.index("\tsetp.ne.b32 \t%p24,", start)
    copies = text[start:end].replace("\tbar.sync \t0;", "\tcp.async.wait_group 2;\n\tbar.sync \t0;", 1)
    text = text[:start] + text[end:]
    mmas = list(re.finditer(r"(?m)^\tmma\.sync[^\n]+", text))
    carry = ("\tadd.s32 %r691, %r682, 1;\n\tsetp.lt.s32 %p26, %r691, %r26;\n"
             "\t@!%p26 bra.uni $L__k64_carry_done;\n" + advance + address + quarters[0]
             + "$L__k64_carry_done:\n")
    for index, code in ((96, copies + carry), (64, quarters[3]), (32, quarters[2]), (0, quarters[1])):
        at = mmas[index].start()
        text = text[:at] + code + text[at:]
    text = text.replace("\tmov.b32 \t%r549, -1;", "\tmov.b32 \t%r549, 0;", 1)
    at = text.index("\tbra.uni \t$L__BB0_2;")
    text = text[:at] + "\tcp.async.wait_group 4;\n\tbar.sync \t0;\n" + address + quarters[0] + text[at:]
    start = text.index("// %bb.5:")
    end = text.index("$L__BB0_7:", start)
    epilogue = text[start:end]
    text = text[:start] + text[end:]
    dispatch = ("$L__k64_output_targets: .branchtargets $L__k64_cold_output, $L__BB0_6;\n"
                "\tselp.u32 %r692, 1, 0, %p24;\n\tbrx.idx.uni %r692, $L__k64_output_targets;\n")
    text = text.replace("\t@%p24 bra \t$L__BB0_6;\n", dispatch, 1)
    text = text.replace("\tret;\n", "\tret;\n$L__k64_cold_output:\n" + epilogue, 1)
    start = text.index("\tadd.s32 \t%r31, %r689, 108;")
    end = text.index("$L__BB0_4:", start)
    text = text[:start] + """
add.s32 %r31, %r689, 108; add.s32 %r187, %r31, -540;
mul.hi.u32 %r188, %r187, 3435973837; shr.u32 %r188, %r188, 2;
mad.lo.s32 %r189, %r188, -5, %r187; mad.lo.s32 %r190, %r189, 128, 3456;
shl.b32 %r191, %r188, 8; or.b32 %r192, %r191, %r8;
add.s32 %r193, %r688, 1024; setp.ge.u32 %p12, %r31, 540;
selp.b32 %r683, %r190, %r683, %p12; selp.b32 %r688, %r192, %r193, %p12;
or.b32 %r684, %r683, %r2; or.b32 %r685, %r683, %r3;
or.b32 %r686, %r683, %r4; or.b32 %r687, %r683, %r5;
mov.b32 %r689, %r31;
""" + text[end:]
    text = text.replace("\tand.b32 \t%r149, %r14, 2160;", """
and.b32 %r149, %r14, 112; and.b32 %dw0, %r1, 128;
shl.b32 %dw0, %dw0, 6; or.b32 %r149, %r149, %dw0;
""", 1)
    text = re.sub(r"(ldmatrix\.sync\.aligned\.m8n8\.x4\.shared\.b16[^\n]+\[%r(?:387|388|389|390)\+)(\d+)(\];)",
                  lambda m: m[1] + str(98304 + (int(m[2])-98304)//2) + m[3], text)
    start = text.index("\tshl.b32 \t%r152, %r7, 9;")
    end = text.index("\tand.b32 \t%r157,", start)
    text = text[:start] + """
and.b32 %dw0, %r7, 15; shl.b32 %r152, %dw0, 9;
and.b32 %dw0, %r7, 7; shl.b32 %r153, %dw0, 4;
and.b32 %r154, %r1, 96; shl.b32 %r155, %r154, 2;
xor.b32 %r156, %r153, %r155; or.b32 %r22, %r156, %r152;
xor.b32 %r22, %r22, %r150; xor.b32 %r23, %r22, 32;
xor.b32 %dw1, %r22, 64; xor.b32 %dw2, %r22, 96;
""" + text[end:]
    pointer = "\tadd.s32 \t%r393, %r391, %r23;"
    text = text.replace(pointer, pointer + "\n\tadd.s32 %dw3, %r391, %dw1;\n\tadd.s32 %dw4, %r391, %dw2;")
    start = text.index("$L__k64_cold_output:\n") + len("$L__k64_cold_output:\n")
    end = text.index("\tmov.b32 \t%r553, 0f00000000;", start)
    text = text[:start] + warp_output_ptx() + text[end:]
    if ptx_digest(text) != "a3ce05c6bb21ac1ffa6d190e8a4bff8f1b33c5cd46cd0851a9eee3da8e637f4a":
        raise RuntimeError("Unexpected lowered PTX")
    return text


def patch_triton_stages(self=None, stages=None, options=None, language=None, capability=None):
    if stages is not None:
        original = stages["ptx"]

        def emit_ranked_ptx(src, metadata):
            text = original(src, metadata)
            if metadata["name"] != "_helion_matmul_ranked_tiles":
                raise RuntimeError("Unexpected ranked kernel")
            actual = (capability, options.num_warps, options.num_stages, metadata["shared"],
                      version("triton"), version("helion"))
            if actual != (80, 8, 4, 155648, "3.7.0", "1.2.0"):
                raise RuntimeError("Unsupported compiler configuration")
            text = lower_ranked_ptx(text)
            metadata["shared"] = 163840
            return text

        stages["ptx"] = emit_ranked_ptx
    return SOURCE_HASH, SOURCE_HASH


FROZEN_CONFIGS={(2048, 2048, 2048): {'block_sizes': [128, 64, 32], 'l2_groupings': [8], 'num_warps': 4, 'num_stages': 4, 'pid_type': 'flat'}, (1024, 1536, 1024): {'block_sizes': [128, 128, 64], 'num_warps': 4, 'num_stages': 3, 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1}, (2048, 3072, 2048): {'block_sizes': [128, 128, 32], 'l2_groupings': [8], 'num_warps': 4, 'num_stages': 3, 'pid_type': 'flat'}, (128, 128, 128): {'block_sizes': [32, 32, 32], 'l2_groupings': [8], 'num_warps': 4, 'num_stages': 4, 'pid_type': 'flat'}, (256, 256, 256): {'block_sizes': [32, 32, 32], 'l2_groupings': [8], 'num_warps': 4, 'num_stages': 4, 'pid_type': 'flat'}, (512, 512, 512): {'block_sizes': [64, 64, 32], 'l2_groupings': [8], 'num_warps': 4, 'num_stages': 4, 'pid_type': 'flat'}, (1024, 1024, 1024): {'block_sizes': [128, 128, 64], 'num_warps': 4, 'num_stages': 3, 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1}}
FROZEN_KERNELS={shape:helion.kernel(matmul_tiles.fn,config=helion.Config(**config),**KERNEL_OPTIONS) for shape,config in FROZEN_CONFIGS.items()}



SMALL_PTX = {'be244d4fba6609b06bef5b0cf934fffa297b9028f45529b11a1f8ff7721815e3': {'after': '150500ae758e7776a2496391ba7207e0b4b810dfa4da37b740a4cef984240916', 'warps': 4, 'shared': 12288, 'removed': 6}, 'e8dd3389a5c61db4664a883953777c0e216c6d4ccec73e97fdc1d2c5108833a6': {'after': 'f72ce4c760814644210fb424c4747d94011fc0e7b446df326de3438b03070462', 'warps': 1, 'shared': 6144, 'removed': 12}}


def lower_small_ptx(text):
    expected = SMALL_PTX.get(ptx_digest(text))
    if expected is None:
        raise RuntimeError("Unexpected small native PTX")
    rows, group, wait, removed = [], 0, 0, 0
    for line in text.splitlines(keepends=True):
        op = line.strip()
        if op.startswith("cp.async.cg.shared.global") and group >= 8:
            removed += 1
            continue
        if op == "cp.async.commit_group;":
            group += 1
            if group > 8:
                continue
        if op.startswith("cp.async.wait_group"):
            line = re.sub(r"(cp.async.wait_group\s+)\d+", lambda m: m[1] + str([4, 4, 2, 0, 0][wait]), line, count=1)
            wait += 1
        rows.append(line)
    result = "".join(rows)
    if (group, wait, removed) != (14, 5, expected["removed"]) or ptx_digest(result) != expected["after"]:
        raise RuntimeError("Unexpected small drain transformation")
    return result


def patch_small_stages(self=None, stages=None, options=None, language=None, capability=None):
    if stages is not None:
        actual = (capability, options.arch, options.num_stages, options.num_ctas,
                  version("triton"), version("helion"), version("torch"))
        if (actual[:6] != (80, "sm80", 4, 1, "3.7.0", "1.2.0") or
                actual[6] not in ("2.12.0", "2.12.0+cu130") or
                (str(torch.__version__), torch.version.cuda) != ("2.12.0+cu130", "13.0")):
            raise RuntimeError("Unsupported small compiler configuration")
        original = stages["ptx"]

        def emit_small_ptx(src, metadata):
            text = original(src, metadata)
            expected = SMALL_PTX.get(ptx_digest(text))
            if metadata["name"] != "_helion_matmul_tiles" or expected is None:
                raise RuntimeError("Unexpected small kernel")
            if (options.num_warps, metadata["shared"]) != (expected["warps"], expected["shared"]):
                raise RuntimeError("Unexpected small resource configuration")
            return lower_small_ptx(text)

        stages["ptx"] = emit_small_ptx
    return SOURCE_HASH, SOURCE_HASH



def lower_256_ptx(text):
    if ptx_digest(text) != "9361a04147f675c47a3136ef6f716e4d6ad000131fe50f8ae27d95bc610479c9":
        raise RuntimeError("Unexpected256 native PTX")
    start = text.index("$L__BB0_1:")
    body_start = text.index("\n", start) + 1
    branch = "\t@%p4 bra \t$L__BB0_1;\n"
    end = text.index(branch)
    text = text[:start] + text[body_start:end] * 8 + text[end + len(branch):]
    if ptx_digest(text) != "88d940beefca92eb36a232dbc9d69005478daee2b4508cfac9c6943b26fffe4c":
        raise RuntimeError("Unexpected256 expanded PTX")
    rows, group, wait, removed = [], 0, 0, 0
    for line in text.splitlines(keepends=True):
        op = line.strip()
        if op.startswith("cp.async.cg.shared.global") and group >= 16:
            removed += 1
            continue
        if op == "cp.async.commit_group;":
            group += 1
            if group > 16:
                continue
        if op.startswith("cp.async.wait_group"):
            line = re.sub(r"(cp.async.wait_group\s+)\d+", lambda m: m[1] + str([4,4,4,4,4,4,2,0,0][wait]), line, count=1)
            wait += 1
        rows.append(line)
    text = "".join(rows)
    if (group, wait, removed) != (22,9,6) or ptx_digest(text) != "5cd8c18a52ac8d24a189e4b65b9cbfee0ec5c40cc5ccb3eadc94698223e24498":
        raise RuntimeError("Unexpected256 terminal drain transformation")

    return text


def patch_256_stages(self=None, stages=None, options=None, language=None, capability=None):
    if stages is not None:
        original = stages["ptx"]
        def emit_256_ptx(src, metadata):
            text = original(src, metadata)
            actual = (metadata["name"], capability, options.num_warps, options.num_stages,
                      metadata["shared"], version("triton"), version("helion"))
            if actual != ("_helion_matmul_tiles", 80, 4, 4, 12288, "3.7.0", "1.2.0"):
                raise RuntimeError("Unsupported256 compiler configuration")
            return lower_256_ptx(text)
        stages["ptx"] = emit_256_ptx
    return SOURCE_HASH, SOURCE_HASH


@helion.kernel(config=helion.Config(block_sizes=[],num_warps=4,num_stages=3,pid_type="flat",l2_groupings=[8]),
               backend="triton",static_shapes=True,autotune_effort="none",dot_precision="ieee",fast_math=False)
def residue_first(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    m, k = a.shape
    n = b.shape[1]
    tail = k % 64
    body = k - tail
    for rows, cols in hl.tile([m, n], block_size=[64, 64]):
        acc = hl.zeros([rows, cols], dtype=torch.float32)
        for ks in hl.tile(tail, block_size=32):
            index = ks.index + body
            acc = hl.dot(hl.load(a, [rows, index]), hl.load(b, [index, cols]),
                         acc, out_dtype=torch.float32)
        for ks in hl.tile(body, block_size=32):
            acc = hl.dot(hl.load(a, [rows, ks]), hl.load(b, [ks, cols]),
                         acc, out_dtype=torch.float32)
        hl.store(c, [rows, cols], acc.to(torch.float16))
    return c


def custom_kernel(data):
    a, b, c = data
    shape = a.shape[0], b.shape[1], a.shape[1]
    if shape == (416, 272, 416):
        return residue_first(a, b, c)
    if shape == (256, 256, 256):
        with knobs.runtime.scope():
            if knobs.runtime.add_stages_inspection_hook is not None:
                raise RuntimeError("Compiler hook already occupied")
            knobs.runtime.add_stages_inspection_hook = patch_256_stages
            return FROZEN_KERNELS[shape](a, b, c)
    if shape == (128, 128, 128):
        with knobs.runtime.scope():
            if knobs.runtime.add_stages_inspection_hook is not None:
                raise RuntimeError("Compiler hook already occupied")
            knobs.runtime.add_stages_inspection_hook = patch_small_stages
            return FROZEN_KERNELS[shape](a, b, c)
    if shape != (4096, 5120, 4096):
        return FROZEN_KERNELS.get(shape, matmul_tiles)(a, b, c)
    with knobs.runtime.scope():
        if knobs.runtime.add_stages_inspection_hook is not None:
            raise RuntimeError("Compiler hook already occupied")
        knobs.runtime.add_stages_inspection_hook = patch_triton_stages
        return matmul_ranked_tiles(a, b.view(-1), c)
