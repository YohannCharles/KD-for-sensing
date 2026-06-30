## 1. OpenSpec

- [x] 1.1 Add proposal for U-MaskBeamJEPA evaluation matrix only.
- [x] 1.2 Add design documenting non-invasive evaluation integration.
- [x] 1.3 Add `u-mask-beam-jepa-eval-matrix` requirements spec.

## 2. Core Eval Utilities

- [x] 2.1 Add fixed missing-pattern mask generator.
- [x] 2.2 Add random eval missing-mask helper.
- [x] 2.3 Add Top-K, ECE and reliability error metrics aggregation helpers.
- [x] 2.4 Add eval matrix runner with fixed and random pattern support.
- [x] 2.5 Add CSV/JSON/Markdown export helpers.

## 3. CLI, Config and Docs

- [x] 3.1 Add `kd-sensing-eval-u-mask-matrix` CLI using existing config/model/dataloader/checkpoint helpers.
- [x] 3.2 Add Scenario 32 eval matrix config.
- [x] 3.3 Add usage and interpretation documentation.

## 4. Tests and Verification

- [x] 4.1 Add fake-model focused tests for patterns, masks, metrics, exports and runner.
- [x] 4.2 Run `openspec validate add-u-mask-beam-jepa-eval-matrix --strict`.
- [x] 4.3 Run `conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa_eval_matrix.py -q`.
- [x] 4.4 Run `conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa.py -q`.
- [x] 4.5 Run small checkpoint smoke eval when a suitable checkpoint is available.
