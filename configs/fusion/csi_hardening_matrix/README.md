# GPS + CSI Validation Matrix

Use `scripts/run_csi_hardening_matrix.sh` for staged execution and logging.

Recommended order:

1. E0-E3 after CSI-only A/B/C/D candidates identify a slow high-ceiling CSI setting.
2. E4 last, after GPS and CSI single-modality teacher checkpoints are available.

E0 is the GPS-only easy-modality baseline. E1-E4 use `modalities: [gps, csi]`. E3 expresses a CSI-prioritized warmup phase, while E4 enables G2D feature/logit KD, teacher confidence diagnostics, and SMP.
