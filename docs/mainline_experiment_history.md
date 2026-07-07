# Mainline Experiment History

## Post-C2 Decision

After final C2, the maintained research surface is final C2 / U-MaskBeamJEPA missing-modality beam prediction, with MMW/CSI retained as future/current supporting dataset workflows. This cleanup preserves the useful lesson from the retired branches: evidence has to be family-consistent, split-consistent and claim-gated before promotion.

## Migrated Historical Notes

- Image+GPS JEPA and GPS-query variants were useful exploration but are now historical; do not rerun the deleted JEPA visual/GPS shortcut CLI as current evidence.
- BeamBench/Arnold22 substitutes, BEV-Fusion 2604, Vision-Position, WCL/TII audits and old RBMA/KD/BTAPA/weakKD sweep were removed from current workflow because they either lacked strict provenance, duplicated the current U-Mask direction, or depended on local-only artifacts.
- Scene31 one-shot runbooks and summaries were collapsed into the retained Scene31-34 main/final C2 launchers plus package diagnostics. Old conclusions stay caveats, not promoted claims.

All historical outputs remain local under ignored `outputs/` or archive context; no checkpoint, cache or log is part of source provenance.
