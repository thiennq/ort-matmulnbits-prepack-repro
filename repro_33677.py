#!/usr/bin/env python3
"""Self-contained repro probe for onnxruntime issue #33677.

Builds synthetic ONNX graphs with int4 MatMulNBits weights (random, seeded),
saves them with ``session.save_external_prepacked_constant_initializers`` and
compares the reloaded (prepacked) model against the same model loaded the
normal way. Needs only numpy, onnx and onnxruntime.

Finding: on x86-64 a MatMulNBits (bits=4, accuracy_level=4) weight computes
wrong values after reload whenever its prepacked blob sits at a file offset
that is not a multiple of 64. Blobs smaller than 1 MiB are written unaligned,
right after the previous initializer. The MLAS x86 packed layout aligns its
sub-buffers (packed B, block sums, scales) to 64 bytes relative to the
buffer's own address. The blob is used in place at the new, unaligned
address, so the kernel reads every sub-buffer from the wrong position.
One node and M=1 are enough; graph fusions play no part.

    python repro_33677.py                 # one MatMulNBits node, exit 1 on mismatch
    python repro_33677.py --sweep         # matrix incl. aligned / accuracy_level 0 controls
    python repro_33677.py --model transformer --m 16 --k 256 --n 1024
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OPSET = [helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]


# --------------------------------------------------------------------------- weights
def quantize_int4(w: np.ndarray, block: int):
    """Symmetric block-wise int4 of w [K, N] in the MatMulNBits layout.

    Returns (packed uint8 [N, K/block, block/2], scales float32 [N, K/block],
    dequantized float32 [K, N]). No zero_points input: the kernel's default
    zero point for 4 bits is 8.
    """
    k, n = w.shape
    assert k % block == 0
    nb = k // block
    wt = w.T.reshape(n, nb, block)                       # [N, nb, block]
    scale = np.abs(wt).max(axis=2, keepdims=True) / 7.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(wt / scale), -8, 7).astype(np.int32) + 8   # 0..15
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)  # low nibble first
    deq = ((q - 8) * scale).reshape(n, k).T.astype(np.float32)
    return packed, scale.reshape(n, nb).astype(np.float32), deq


class Builder:
    def __init__(self, seed: int, block: int, acc: int, fp32: bool):
        self.rng = np.random.default_rng(seed)
        self.block, self.acc, self.fp32 = block, acc, fp32
        self.nodes, self.inits, self.i = [], [], 0

    def name(self, p: str) -> str:
        self.i += 1
        return f"{p}_{self.i}"

    def const(self, arr: np.ndarray, p: str = "c") -> str:
        n = self.name(p)
        self.inits.append(numpy_helper.from_array(arr, n))
        return n

    def node(self, op: str, inputs, domain: str = "", **attrs) -> str:
        out = self.name(op.lower())
        self.nodes.append(helper.make_node(op, list(inputs), [out], name=out + "_n",
                                           domain=domain, **attrs))
        return out

    def linear(self, x: str, k: int, n: int, bias: bool = True) -> str:
        w = (self.rng.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32)
        packed, scales, deq = quantize_int4(w, self.block)
        if self.fp32:  # reference twin: same dequantized weights, plain MatMul
            y = self.node("MatMul", [x, self.const(deq, "w")])
        else:
            y = self.node("MatMulNBits", [x, self.const(packed, "qw"), self.const(scales, "s")],
                          domain="com.microsoft", K=k, N=n, bits=4, block_size=self.block,
                          accuracy_level=self.acc)
        if bias:
            y = self.node("Add", [y, self.const((0.02 * self.rng.standard_normal(n)).astype(np.float32), "b")])
        return y

    def layernorm(self, x: str, d: int) -> str:
        return self.node("LayerNormalization",
                         [x, self.const(np.ones(d, np.float32) + 0.05 * self.rng.standard_normal(d).astype(np.float32), "g"),
                          self.const((0.05 * self.rng.standard_normal(d)).astype(np.float32), "beta")],
                         axis=-1, epsilon=1e-5)

    def gelu(self, x: str) -> str:  # erf form, the pattern GeluFusion/BiasGelu match
        h = self.node("Div", [x, self.const(np.array(np.sqrt(2.0), np.float32))])
        h = self.node("Erf", [h])
        h = self.node("Add", [h, self.const(np.array(1.0, np.float32))])
        h = self.node("Mul", [x, h])
        return self.node("Mul", [h, self.const(np.array(0.5, np.float32))])

    def attention(self, x: str, d: int, heads: int) -> str:
        hd = d // heads
        shape = self.const(np.array([0, 0, heads, hd], np.int64), "shp")

        def split(t):
            return self.node("Transpose", [self.node("Reshape", [t, shape])], perm=[0, 2, 1, 3])
        q, k, v = (split(self.linear(x, d, d)) for _ in range(3))
        kt = self.node("Transpose", [k], perm=[0, 1, 3, 2])
        s = self.node("Mul", [self.node("MatMul", [q, kt]), self.const(np.array(1 / np.sqrt(hd), np.float32))])
        p = self.node("Softmax", [s], axis=-1)
        o = self.node("Transpose", [self.node("MatMul", [p, v])], perm=[0, 2, 1, 3])
        o = self.node("Reshape", [o, self.const(np.array([0, 0, d], np.int64), "shp")])
        return self.linear(o, d, d)


def build(kind: str, args, fp32: bool) -> onnx.ModelProto:
    b = Builder(args.seed, args.block, args.acc, fp32)
    d, ff = args.k, args.n
    if kind == "single":  # one MatMulNBits node, 2-D input [M, K]
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["M", args.k])
        y = b.linear("x", args.k, args.n, bias=False)
    elif kind == "chain":  # MatMulNBits K->N then N->K, no other ops
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["M", args.k])
        y = b.linear(b.linear("x", d, ff, bias=False), ff, d, bias=False)
    elif kind in ("mlp", "transformer"):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "M", d])
        h = "x"
        for _ in range(args.layers):
            if kind == "transformer":
                h = b.node("Add", [h, b.attention(b.layernorm(h, d), d, args.heads)])
            f = b.linear(b.gelu(b.linear(b.layernorm(h, d), d, ff)), ff, d)
            h = b.node("Add", [h, f])
        y = b.layernorm(h, d)
    else:
        raise SystemExit(f"unknown model {kind}")
    out_shape = [1, "M", d] if kind in ("mlp", "transformer") else ["M", args.n if kind == "single" else d]
    g = helper.make_graph(b.nodes, kind, [x], [helper.make_tensor_value_info(y, TensorProto.FLOAT, out_shape)],
                          b.inits)
    m = helper.make_model(g, opset_imports=OPSET, producer_name="ort-33677-repro")
    m.ir_version = 8
    return m


# --------------------------------------------------------------------------- sessions
def options(args, level=None) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.graph_optimization_level = level if level is not None else LEVELS[args.opt]
    so.intra_op_num_threads = args.threads
    so.log_severity_level = 3
    return so


LEVELS = {"all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
          "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
          "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
          "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL}


def save_prepacked(src: str, dst: str, args) -> None:
    """The flow from the issue: optimize + write packed weights to dst(.data)."""
    so = options(args)
    so.intra_op_num_threads = 1
    so.add_session_config_entry("session.save_external_prepacked_constant_initializers", "1")
    so.add_session_config_entry("session.optimized_model_external_initializers_file_name",
                                os.path.basename(dst).replace(".onnx", ".data"))
    so.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "1024")
    so.optimized_model_filepath = dst
    ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])


def run(path: str, x: np.ndarray, args) -> np.ndarray:
    s = ort.InferenceSession(path, options(args), providers=["CPUExecutionProvider"])
    return s.run(None, {s.get_inputs()[0].name: x})[0]


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if np.isnan(a).any() or np.isnan(b).any():
        return {"max_abs": float("nan"), "cos": float("nan")}
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return {"max_abs": float(np.abs(a - b).max()), "cos": cos}


def prepacked_offsets(path: str) -> list[int]:
    """Byte offsets of the prepacked blobs inside the saved .data file."""
    offs = []
    for t in onnx.load(path, load_external_data=False).graph.initializer:
        for e in t.external_data:
            if e.key.startswith("prepacked"):  # "<key>|<offset>;<length>;<alignment>"
                offs.append(int(e.value.split("|")[1].split(";")[0]))
    return offs


def case(args) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        if args.keep:
            os.makedirs(args.keep, exist_ok=True)
            tmp = args.keep
        q, ref, packed = (os.path.join(tmp, f) for f in ("model.onnx", "ref.onnx", "model.packed.onnx"))
        onnx.save(build(args.model, args, fp32=False), q)
        onnx.save(build(args.model, args, fp32=True), ref)
        save_prepacked(q, packed, args)
        rng = np.random.default_rng(args.seed + 1)
        shape = (1, args.m, args.k) if args.model in ("mlp", "transformer") else (args.m, args.k)
        x = rng.standard_normal(shape).astype(np.float32)
        y_ref, y_q, y_p = run(ref, x, args), run(q, x, args), run(packed, x, args)
        offs = prepacked_offsets(packed)
        r = {"model": args.model, "M": args.m, "K": args.k, "N": args.n, "block": args.block,
             "acc": args.acc, "layers": args.layers, "opt": args.opt, "threads": args.threads,
             "prepacked_offsets_mod64": [o % 64 for o in offs],
             "unpacked_vs_fp32": compare(y_q, y_ref), "prepacked_vs_unpacked": compare(y_p, y_q)}
        # The prepacked weights run the same kernel on the same packed bytes, so
        # the two loads must agree (bit-exact on every platform where it works).
        d = r["prepacked_vs_unpacked"]["max_abs"]
        r["mismatch"] = not (d <= 1e-5)  # also true for NaN
        return r


def cpu_flags() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    fl = set(line.split(":")[1].split())
                    return " ".join(sorted(x for x in fl if x.startswith(("avx", "amx")) or x in ("f16c", "fma")))
    except OSError:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            k = ctypes.windll.kernel32  # PF_AVX2_INSTRUCTIONS_AVAILABLE=40, PF_AVX512F=41
            return " ".join(n for n, f in (("avx", 39), ("avx2", 40), ("avx512f", 41))
                            if k.IsProcessorFeaturePresent(f))
        except Exception:
            return "?"
    if sys.platform == "darwin":
        import subprocess
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return "?"


# model:M:K:N:accuracy_level. The prepacked blob is written right after the
# N*K/2-byte B initializer, so N*K/2 % 64 decides where it lands in the file.
DEFAULT_SWEEP = " ".join([
    "single:1:64:32:4",     # B = 1024 bytes -> blob offset % 64 == 0  (control)
    "single:1:64:33:4",     # B = 1056 bytes -> blob offset % 64 == 32
    "single:4:32:65:4",     # B = 1040 bytes -> % 64 == 16
    "single:4:32:67:4",     # B = 1072 bytes -> % 64 == 48
    "single:4:64:33:0",     # same misaligned blob, accuracy_level 0 (control)
    "single:16:800:767:4",  # realistic size, % 64 == 48
    "single:16:768:768:4",  # realistic size, aligned (control)
    "single:4:1056:1535:4",  # B % 64 == 48 but blob >= 1 MiB, which the writer aligns to 4096 (control)
    "transformer:16:256:1024:4",  # 2-layer transformer, all blobs aligned (control)
])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="single", choices=["single", "chain", "mlp", "transformer"])
    ap.add_argument("--m", type=int, default=1, help="rows of A (sequence length)")
    ap.add_argument("--k", type=int, default=64, help="K (model dim for mlp/transformer)")
    ap.add_argument("--n", type=int, default=33, help="N (FF dim for mlp/transformer)")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--acc", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--opt", default="all", choices=list(LEVELS))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", nargs="?", const="default",
                    help="run the default matrix, or a space-separated list of model:M:K:N:acc[:threads]")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--keep", help="write the models to this directory instead of a temp dir")
    args = ap.parse_args()

    print(f"onnxruntime {ort.__version__}  onnx {onnx.__version__}  numpy {np.__version__}  "
          f"python {platform.python_version()}  {platform.system()} {platform.machine()}")
    print(f"cpu: {platform.processor() or '?'} | flags: {cpu_flags()}")

    if args.sweep:
        spec = DEFAULT_SWEEP if args.sweep == "default" else args.sweep
        grid = []
        for item in spec.split():
            f = item.split(":")
            g = dict(model=f[0], m=int(f[1]), k=int(f[2]), n=int(f[3]), acc=int(f[4]))
            if len(f) > 5:
                g["threads"] = int(f[5])
            grid.append(g)
        results = [case(argparse.Namespace(**{**vars(args), **g})) for g in grid]
    else:
        results = [case(args)]

    bad = False
    for r in results:
        bad |= r["mismatch"]
        if args.json:
            print(json.dumps(r))
            continue
        pv = r["prepacked_vs_unpacked"]
        print(f"{r['model']:<11} M={r['M']:<3} K={r['K']:<4} N={r['N']:<4} block={r['block']} acc={r['acc']} "
              f"blob_off%64={','.join(map(str, sorted(set(r['prepacked_offsets_mod64'])))) or '-':<9} | "
              f"prepacked vs unpacked: max_abs={pv['max_abs']:<9.3g} cos={pv['cos']:<9.5f} | "
              f"{'MISMATCH' if r['mismatch'] else 'ok'}")
    print("RESULT:", "MISMATCH (bug reproduced)" if bad else "all match")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
