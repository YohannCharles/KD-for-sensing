# JEPA Image+GPS experiment configs

This directory stores experiment-specific image+GPS supervised reuse configs for GPS-conditioned JEPA checkpoints. They are retained for reproducibility of BeamBench-fair and arXiv:2604.05668 alignment checks, but are not canonical `configs/fusion/` root entrypoints.

Root-level fusion YAML files are reserved for long-lived canonical or currently recommended thin entries. Low-memory variants, best/last checkpoint comparisons, and scene-matrix replicas belong here unless promoted by a future OpenSpec change.

Mainline reporting uses the GPS-biased JEPA reuse configs:

- BeamBench-fair: `image_gps_jepa_gps_biased_best_beambench_fair_lowmem.yaml`.
- arXiv:2604.05668 S32/S33/S34 comparison: `image_gps_jepa_gps_biased_best_2604_s32_s34_lowmem.yaml`.

GPS-query pooling configs are paired derivatives of the GPS-biased mainline. They reuse the same
multi-scene GPS-biased JEPA checkpoint and supervised beam recipe, but set the image encoder
`pooling: gps_query_attention` so projected GPS features query the JEPA patch tokens before fusion:

- BeamBench-fair GPS-query: `image_gps_jepa_gps_query_pool_best_beambench_fair_lowmem.yaml`.
- arXiv:2604.05668 GPS-query: `image_gps_jepa_gps_query_pool_best_2604_s32_s34_lowmem.yaml`.
- BeamBench-fair explicit pooler/parameter-groups example:
  `image_gps_jepa_gps_biased_pooler_param_groups_beambench_fair_lowmem.yaml`.

Use the original `fair_gps_biased` mean-pooling configs as the primary baseline for any GPS-query
claim. Report GPS-query runs against the matching split protocol, checkpoint, label space, learning
rate schedule, and batch-size recipe rather than mixing BeamBench-fair and 2604-style numbers.

Pooler/adapter derivatives should only override the experimental variable under test: `pooler`,
`adapter`, freeze flags, `training.optimizer.parameter_groups`, `experiment.ablation`, and
`output.run_name`. Keep the inherited GPS-biased multi-scene checkpoint, Image+GPS modalities,
beam objective, label space, split protocol, and baseline learning recipe aligned with the matching
`fair_gps_biased` config.

Optimizer parameter groups are intended for paired sensitivity checks. Compare them against the
matching baseline and report the runtime `optimizer_param_groups` summary together with JEPA
downstream metadata so readers can see each group name, learning rate, weight decay, and trainable
parameter count.

New downstream pooler/adapter experiments should remain paired derivatives of the same
`fair_gps_biased` baseline. Prefer the explicit form:

```yaml
model:
  primary:
    encoders:
      image:
        pooler:
          type: gps_query_attention
          k_queries: 4
          num_heads: 4
          condition_source: projected_gps
        adapter:
          type: identity
```

The legacy `pooling: mean` and `pooling: gps_query_attention` fields are retained as aliases for
existing configs. Parameter-group learning-rate sweeps belong in derived configs and should only
change optimizer groups, pooler/adapter settings, freeze flags, run name, and ablation metadata.
Do not mix BeamBench-fair and 2604-style checkpoints, label spaces, split protocols, or schedule
recipes when comparing adapter or parameter-group variants.

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
