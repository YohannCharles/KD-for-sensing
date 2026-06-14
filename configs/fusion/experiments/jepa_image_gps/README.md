# JEPA Image+GPS experiment configs

This directory stores experiment-specific image+GPS supervised reuse configs for GPS-conditioned JEPA checkpoints. They are retained for reproducibility of BeamBench-fair and arXiv:2604.05668 alignment checks, but are not canonical `configs/fusion/` root entrypoints.

Root-level fusion YAML files are reserved for long-lived canonical or currently recommended thin entries. Low-memory variants, BeamBench-fair checks, 2604-style checks, and paired query-pool controls belong here unless promoted by a future OpenSpec change.

Protocol and claim status are centralized in:

- `docs/mainline_model_catalog.md`
- `docs/experiment_protocols.md`
- `docs/result_claims_registry.md`

Status summary:

| family | configs | status | claim registry |
| --- | --- | --- | --- |
| BeamBench-fair supervised/random/GPS-biased/GPS-query | `*beambench_fair_lowmem.yaml` | `lowmem formal/control`; current target, `seq_len=1`, `num_pred=1`, `paper_distance_angle`, linear DBA | `CLAIM-JEPA-BBFAIR-PENDING`, `CLAIM-JEPA-QUERY-PENDING` |
| 2604-style supervised/random/GPS-biased/GPS-query | `*2604_s32_s34_lowmem.yaml` and `*fasttrain.yaml` | `lowmem formal/control`; S32/S33/S34 stratified 80/10/10, `seq_len=5`, `num_pred=1` | `CLAIM-JEPA-2604-LOCAL-001` |
| BeamBench Table III Camera AE+GPS Direct | not in this directory | use `configs/fusion/beambench_image_ae_gps_direct.yaml` and dedicated runner | `CLAIM-BB-TIII-CURRENT-BLOCKED` |

Mainline reporting uses the GPS-biased JEPA reuse configs:

- BeamBench input/split/metric alignment: `image_gps_jepa_gps_biased_best_beambench_fair_lowmem.yaml`.
- arXiv:2604.05668 S32/S33/S34 comparison: `image_gps_jepa_gps_biased_best_2604_s32_s34_lowmem.yaml`.

The retained `beambench_fair` filenames now mean the downstream Image+GPS/JEPA controls use
BeamBench Table III style `seq_len=1`, `num_pred=1`, `paper_distance_angle` GPS Direct features,
scene paper calibration angles, current-beam targets, S32-S34 training scenes, S31-S34 test scenes,
and linear DBA with Top-1/3/5. They do not become the Table III Camera AE+GPS Direct model. For that row, use
`configs/fusion/beambench_image_ae_gps_direct.yaml` with
`kd-sensing-run-beambench-image-ae-gps-tableiii`.

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
result for local 2604-style BEV-Fusion comparisons, with the split caveat recorded in
`docs/result_claims_registry.md`. The supervised and random-mask variants are controls.

The next-beam downstream ablation matrix has been retired. Current JEPA downstream comparisons
should stay on the GPS-biased mean-pooling baseline, GPS-query pooling derivative, supervised
baseline, random-best control, and retained BeamBench-fair checks.
