# Workload Graph Manifests

This directory is for source-controlled graph manifests that are maintained by
hand.  Generated sweep graphs should be recreated from scripts instead of
checked in here.

Current hand-maintained graphs:

- `attention_smoke`: transformer-block-style graph used as the graph-sweep
  template.
- `attention_flash_smoke`: transformer-block-style graph variant that replaces
  canonical `qk -> softmax -> pv` attention with the Triton FlashAttention-style
  online-softmax node.  This is kept as the SOTA-style comparison track, not the
  current CPU/SPM optimization lead.
- `attention_mh_causal_smoke`: two-head causal decoder-block graph that keeps
  decomposed per-head `qk -> softmax -> pv` attention and folds the attention
  scale into Q.  This is the stronger transformer-block workload for paper
  claims that need multi-head causal attention without requiring high-rank graph
  kernels.
- `attention_mh_flash_smoke`: two-head causal decoder-block comparison graph
  that swaps each decomposed per-head attention chain for a FlashAttention-style
  online-softmax node while reusing the same output projection and FFN tail.
- `decoder_canonical_mh8`: large eight-head causal decoder-block graph at
  `SEQ=512`, `D_MODEL=512`, `HEADS=8`, `HEAD_DIM=64`, `FFN_DIM=2048`.
  Each head uses canonical attention (`QK^T -> softmax -> PV`).  This is the
  current main decoder graph; use `--preset large_profile` only for per-kernel
  gem5 dump/reset attribution runs.
- `canonical_attention`: attention fixture that isolates
  `qk -> softmax -> pv` and reuses the existing `matmul` workload for both GEMM
  contractions.  This is the active CPU/SPM attention direction.  Use
  `--preset s256h64-c1` for the steady causal comparison point.
- `layer_norm_qkv`: producer-consumer fusion fixture.

Historical generated P0.5/P4 sweep manifests are no longer source-controlled.
The maintained graph reproduction route is `workloads/scripts/paper_experiments.py`,
with generated outputs written under ignored `workloads/m5out/` paths.
