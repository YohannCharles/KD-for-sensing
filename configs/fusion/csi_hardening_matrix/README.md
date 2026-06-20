# GPS + CSI Validation Matrix

Protocol status and claim provenance are tracked in `docs/experiment_protocols.md` and `docs/result_claims_registry.md`. E0 is the GPS-only control; E1-E3 are `formal/control` GPS+CSI validation entries and should be compared only after CSI-only debug/formal gates are clear.
E configs are lightweight overlays on `_base/gps_only.yaml` or `_base/gps_csi.yaml`; inspect the resolved config from `kd-sensing-train` for full merged values.

Use `scripts/run_csi_hardening_matrix.sh` for staged execution and logging.

Recommended order:

1. E0-E3 after CSI-only A/B/C/D candidates identify a slow high-ceiling CSI setting.

E0 is the GPS-only easy-modality baseline. E1-E3 use `modalities: [gps, csi]`. E3 expresses a CSI-prioritized warmup phase without relying on retired distillation methods.
