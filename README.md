# MatMulNBits prepacked-weights repro (onnxruntime #33677)

Self-contained repro for [microsoft/onnxruntime#33677](https://github.com/microsoft/onnxruntime/issues/33677).
On x86-64, a `MatMulNBits` node (int4, `accuracy_level=4`) computes wrong values after the model is
saved with `session.save_external_prepacked_constant_initializers` and loaded again.

## The bug

1. The prepacked blob is written to the external data file. Blobs smaller than 1 MiB are not aligned:
   each one is written right after the previous initializer. Only blobs of 1 MiB or more are aligned to
   4096 bytes.
2. On x86, the `accuracy_level=4` (CompInt8) packed layout (packed B data, block sums, scales) aligns the
   packed data and the block sums to 64 bytes **relative to the buffer address** (scales follow the block
   sums; `PackedQuantBDataStruct` in `mlas/lib/qnbitgemm.h`). `accuracy_level=0` has no block sums.
3. When memory-mapped, the blob is used in place at an address with the same residue mod 64 as its file
   offset (the copy fallback does not preserve it). For the `accuracy_level=4` layout a nonzero residue
   shifts the interpreted packed-data, block-sum and scale offsets.

One node with M=1 is enough, and graph fusions play no part. In our runs arm64 and `accuracy_level=0` are
not affected.

## Run

```sh
pip install onnxruntime==1.31.1 onnx==1.23.2 numpy==2.5.4
python repro_33677.py           # one node: K=64, N=33, M=1; exits 1 on mismatch
python repro_33677.py --sweep   # matrix with controls
python repro_33677.py --sweep --keep DIR   # also keep each case's models in DIR/<case>/
```

The script builds int4 weights from a seeded RNG (no external model). It saves the prepacked model and
compares it with the same model loaded normally. Both loads run the same kernel on the same packed bytes,
so they should agree bit for bit.

At load time ORT packs each weight again and uses the blob from disk only when its key (a hash of the
fresh packed bytes) matches, so a silent in-memory repack would also "match". The script therefore
XORs every prepacked blob range in a copy of the `.data` file with 0xFF, reloads and runs again. The
output must change (`consumed=yes`); otherwise the case is `INCONCLUSIVE`. The `blob_len` column gives
the blob sizes.

Exit codes: `0` every case matches and its blob was consumed, `1` mismatch (bug reproduced),
`2` no mismatch but at least one case is inconclusive.

## Expected vs actual (`--sweep`, onnxruntime 1.31.1)

Mismatch cells show max abs diff and cosine similarity between the two loads.

| Case | Blob offset % 64 (x86) | x86-64 AVX2 | x86-64 AVX-512 | arm64 |
|---|---|---|---|---|
| K=64, N=32, acc 4 | 0 | match | match | match |
| K=64, N=33, acc 4 | 32 | **1.74, cos 0.154** | **2.62, cos 0.187** | match |
| K=32, N=65, acc 4 | 16 | **5.58, cos 0.051** | **5.58, cos 0.051** | match |
| K=32, N=67, acc 4 | 48 | **4.77, cos -0.070** | **4.77, cos -0.070** | match |
| K=64, N=33, **acc 0** | 32 | match | match | match |
| K=800, N=767, acc 4 | 48 | **7.22, cos 0.019** | **6.09, cos 0.019** | match |
| K=768, N=768, acc 4 | 0 | match | match | match |
| K=1056, N=1535, acc 4 (blob ≥ 1 MiB, aligned to 4096) | 0 | match | match | match¹ |
| transformer M=16, K=256, N=1024, acc 4 | 0 | match | match | match |

¹ On arm64 the packed layout is smaller, so this blob is < 1 MiB, is not aligned, and lands at
offset % 64 = 48. It still matches, which supports "arm64 not affected".

Expected: every row matches.

## Results by platform

| Platform | CPU | onnxruntime | Result |
|---|---|---|---|
| Windows x86-64 VM | Intel (QEMU), AVX2 | 1.31.1, 1.32.0.dev20261031001 | mismatch (bug) |
| Linux x86-64 VM | AVX-512 (avx512f/bw/cd/dq/vl) | 1.31.1 | mismatch |
| GitHub Actions `ubuntu-latest` | AMD EPYC 9V45 / 9V74, AVX-512 | 1.31.1, nightly | mismatch |
| GitHub Actions `windows-latest` | AMD EPYC 7763 (AVX2) / 9V74 (AVX-512) | 1.31.1 / nightly | mismatch |
| GitHub Actions `macos-latest` | Apple M2 Pro (arm64) | 1.31.1, nightly | all match |
| Mac M1 | Apple M1 (arm64) | 1.31.1 | all match |

The CI workflow runs on `ubuntu-latest`, `windows-latest` and `macos-latest` with 1.31.1 and the
nightly build. The x86 jobs fail on purpose while the bug exists; the job summary tells a mismatch
(exit 1) from an inconclusive run (exit 2). Example run:
<https://github.com/thiennq/ort-matmulnbits-prepack-repro/actions/runs/41380771700>

## License

MIT
