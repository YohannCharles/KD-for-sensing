# JEPA Image+GPS experiment configs

This directory stores experiment-specific image+GPS supervised reuse configs for GPS-conditioned JEPA checkpoints. They are retained for reproducibility of BeamBench-fair and arXiv:2604.05668 alignment checks, but are not canonical `configs/fusion/` root entrypoints.

Root-level fusion YAML files are reserved for long-lived canonical or currently recommended thin entries. Low-memory variants, best/last checkpoint comparisons, and scene-matrix replicas belong here unless promoted by a future OpenSpec change.

Mainline reporting uses the GPS-biased JEPA reuse configs:

- BeamBench-fair: `image_gps_jepa_gps_biased_best_beambench_fair_lowmem.yaml`.
- arXiv:2604.05668 S32/S33/S34 comparison: `image_gps_jepa_gps_biased_best_2604_s32_s34_lowmem.yaml`.

The current 2604-style primary result is the `fair_gps_biased` evaluation, with S32/S33/S34 DBA
`0.8777 / 0.8853 / 0.8796` and macro DBA `0.8809`. Treat this as the main Image+GPS+JEPA
result for BEV-Fusion comparisons. The supervised and random-mask variants are controls.

The next-beam downstream ablation matrix uses the same JEPA context image checkpoint and supervised beam recipe while changing only the fusion core and required horizon settings:

- `jepa_gru.yaml`: `early_concat_gru` history baseline.
- `jepa_snapshot.yaml`: `snapshot_frame` single-frame baseline with `seq_len=1` and `num_pred=1`.
- `jepa_plain_token_transformer.yaml`: existing `token_transformer` baseline without next-query metadata.
- `jepa_next_query_transformer.yaml`: `next_beam_query_transformer` with learned time embedding, modality embedding, and next-beam query.

These next-beam configs are supporting ablations. They should not replace the GPS-biased mainline
unless a retrained run beats `fair_gps_biased` on the same 2604-style stratified S32/S33/S34 macro
DBA protocol.
