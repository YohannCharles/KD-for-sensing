# Training Throughput Notes

The standalone training-throughput CLI is retired in the post-C2 cleanup. Throughput tuning is now handled as local profiling around the current training command:

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/u_mask_beam_jepa_smoke.yaml
```

Keep profiler traces, TensorBoard events and timing CSVs under ignored `outputs/` or `logs/`. Reintroducing a public throughput CLI requires a new OpenSpec change and focused tests.
