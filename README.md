# MatMulNBits prepacked-weights repro (onnxruntime #33677)

Self-contained repro for [microsoft/onnxruntime#33677](https://github.com/microsoft/onnxruntime/issues/33677).
On x86-64, a `MatMulNBits` node (int4, `accuracy_level=4`) computes wrong values after the model is
saved with `session.save_external_prepacked_constant_initializers` and loaded again.

## The bug

1. The prepacked blob is written to the external data file. Blobs smaller than 1 MiB are not aligned:
   each one is written right after the previous initializer. Only blobs of 1 MiB or more are aligned to
   4096 bytes.
2. On x86, MLAS lays out the packed buffer (packed B data, block sums, scales) by aligning each part to
   64 bytes **relative to the buffer address** (`PackedQuantBDataStruct` in `mlas/lib/qnbitgemm.h`).
3. At load time the blob is used in place, so its address has the same alignment as its file offset.
   When `offset % 64 != 0`, the kernel reads every part from the wrong position.

One node with M=1 is enough, and graph fusions play no part. In our runs arm64 and `accuracy_level=0` are
not affected.

## Run

```sh
pip install onnxruntime==1.31.1 onnx numpy
python repro_33677.py           # one node: K=64, N=33, M=1; exits 1 on mismatch
python repro_33677.py --sweep   # matrix with controls
```

The script builds int4 weights from a seeded RNG (no external model). It saves the prepacked model and
compares it with the same model loaded normally. Both loads run the same kernel on the same packed bytes,
so they should agree bit for bit.

## Expected vs actual (`--sweep`, onnxruntime 1.31.1)

| Case | Blob offset % 64 | x86-64 (AVX2 / AVX-512) | arm64 |
|---|---|---|---|
| K=64, N=32, acc 4 | 0 | match | match |
| K=64, N=33, acc 4 | 32 | **max abs diff 1.7, cos 0.15** | match |
| K=32, N=65 / 67, acc 4 | 16 / 48 | **mismatch** | match |
| K=64, N=33, **acc 0** | 32 | match | match |
| K=800, N=767, acc 4 | 48 | **mismatch** | match |
| K=1056, N=1535, acc 4 (blob ≥ 1 MiB, aligned to 4096) | 0 | match | match |

Expected: every row matches. The CI workflow runs on `ubuntu-latest`, `windows-latest` and
`macos-latest` with 1.31.1 and the nightly build. The x86 jobs fail on purpose while the bug exists.

## License

MIT
