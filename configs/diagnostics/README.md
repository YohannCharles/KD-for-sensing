# Diagnostic and benchmark manifests

Protocol status and claim provenance are tracked in `docs/experiment_protocols.md` and `docs/result_claims_registry.md`.

| manifest | status | boundary |
| --- | --- | --- |
| `jepa_gps_shortcut_benchmark_smoke.yaml` | `smoke` | uses synthetic metrics to validate runner, aggregation and visual-analysis ingestion; not a research result |
| `jepa_gps_shortcut_benchmark_beambench_fair.yaml` | `formal benchmark / evaluation-only` | requires audited local checkpoint paths before reporting; metrics and figures remain ignored outputs |
| `jepa_gps_shortcut_benchmark_scenario_d_smoke.yaml` | `smoke / evaluation-only` | validates Scenario D CxD phase, crossing, failure decomposition and dominance-status artifacts with synthetic metrics; dominance evidence remains mock/unavailable without real diagnostics |
| `jepa_visual_analysis_2604.yaml` | `diagnostic-only` | reads existing model/config/checkpoint/cache; attention, embedding and case figures are supporting evidence |
| `modality_visualization.yaml` | `diagnostic-only` | drives viewer manifest export; does not restore the retired static PNG overview or repository Web UI |

All generated reports, figures, cache and manifests belong under ignored `outputs/analysis/`, `outputs/visual_analysis/` or `outputs/cache/`.
