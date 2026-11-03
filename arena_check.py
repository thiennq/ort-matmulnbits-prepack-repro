#!/usr/bin/env python3
"""Does an in-memory MatMulNBits pack stay correct when the CPU arena is off?

If the allocator behind PrePack only guaranteed 64-byte alignment, the in-memory
packed buffer could land on an address the x86 CompInt8 layout misreads. We
compare the in-memory (never saved) model against the fp32 twin from
repro_33677.py with the arena on and off, and across several allocation
patterns. A large error against fp32 means the packed buffer was misread.
Usage: python arena_check.py /path/to/repro_33677.py
"""
import argparse, importlib.util, os, sys, tempfile
import numpy as np, onnx
import onnxruntime as ort

spec = importlib.util.spec_from_file_location("repro", sys.argv[1])
repro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repro)

CASES = [(1, 64, 33), (4, 32, 65), (4, 32, 67), (16, 800, 767), (16, 768, 768)]
TOLERANCE = 0.5  # int8 activation quantization error is well below this; a misread layout is ~1-7


def sess(path, arena, mem_pattern):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 1
    so.enable_cpu_mem_arena = arena
    so.enable_mem_pattern = mem_pattern
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def main():
    bad = 0
    print(f"onnxruntime {ort.__version__}")
    print(f"{'case':<22}{'arena':<7}{'mempat':<8}{'max_abs vs fp32':<18}status")
    for m, k, n in CASES:
        args = argparse.Namespace(model="single", m=m, k=k, n=n, block=32, acc=4, layers=2, heads=4,
                                  opt="all", threads=1, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            q, ref = os.path.join(tmp, "q.onnx"), os.path.join(tmp, "ref.onnx")
            onnx.save(repro.build("single", args, fp32=False), q)
            onnx.save(repro.build("single", args, fp32=True), ref)
            x = np.random.default_rng(1).standard_normal((m, k)).astype(np.float32)
            s = sess(ref, True, True)
            y_ref = s.run(None, {s.get_inputs()[0].name: x})[0]
            for arena in (True, False):
                for pat in (True, False):
                    s = sess(q, arena, pat)
                    y = s.run(None, {s.get_inputs()[0].name: x})[0]
                    d = float(np.max(np.abs(y - y_ref)))
                    ok = d <= TOLERANCE
                    bad += not ok
                    print(f"M={m} K={k} N={n}".ljust(22) + f"{arena!s:<7}{pat!s:<8}{d:<18.4f}{'ok' if ok else 'WRONG'}")
    print("RESULT:", "all in-memory packs correct" if not bad else f"{bad} wrong")
    return 1 if bad else 0


sys.exit(main())
