## ADDED Requirements

### Requirement: CSI pilot disabled identity diagnostics
CSI pilot estimator MUST expose diagnostics that prove disabled pilot estimation leaves the normalized complex CSI unchanged. When pilot estimation is disabled, `h_hat` MUST be exactly equal to the input `h` within floating point precision and the debug output MUST include `max_abs(h_hat - h)`.

#### Scenario: pilot disabled returns identity
- **WHEN** `pilot_estimator.enabled` is `false`
- **THEN** `PilotCSIChannelEstimator` MUST return `h_hat` equal to the input normalized CSI
- **AND** debug diagnostics MUST report `max_abs(h_hat - h)` as 0 or floating point zero

#### Scenario: mild pilot SNR records noise ratio
- **WHEN** pilot estimation is enabled with training SNR sampled between 25 dB and 35 dB
- **THEN** diagnostics MUST record sampled SNR or equivalent SNR tensor
- **AND** diagnostics MUST record `noise_power/signal_power`
- **AND** the expected ratio SHOULD be approximately between 0.003 and 0.0003 before stochastic tolerance is applied

### Requirement: CSI hardening invariant diagnostics
CSI hardening diagnostics MUST verify that hardening preserves complex CSI shape, finite values and expected scale unless a configuration explicitly requests gain scaling. The diagnostics MUST be available for each enabled hardening transform.

#### Scenario: hardening preserves shape and finite values
- **WHEN** hardening is applied to normalized complex CSI with shape `[B,T,Nsc,Nant]`
- **THEN** the hardening output MUST keep shape `[B,T,Nsc,Nant]`
- **AND** diagnostics MUST report `nan_count=0`
- **AND** diagnostics MUST report zero ratio and magnitude statistics

#### Scenario: hardening scale drift warning
- **WHEN** hardening output abs_mean or abs_std changes by more than 20 percent relative to hardening input without explicit gain scaling
- **THEN** diagnostics MUST mark the batch as suspicious
- **AND** the warning MUST include before and after abs_mean and abs_std values

#### Scenario: fixed antenna transforms are not resampled per batch
- **WHEN** antenna calibration or fixed antenna permutation uses a fixed seed
- **THEN** the transform MUST remain stable across batches in the same run
- **AND** diagnostics MUST expose enough transform identity information to detect accidental per-batch resampling

### Requirement: CSI encoder path structure diagnostics
`pilot_dual_view_csi` MUST report its resolved structure at run start when model debug summary is enabled. The summary MUST include `use_internal_gru`, `view_fusion`, `delay_taps`, `d_model`, total parameters and trainable parameters.

#### Scenario: default internal GRU is visible
- **WHEN** a run builds a CSI encoder without explicitly setting `use_internal_gru`
- **THEN** the resolved summary MUST show `use_internal_gru=true`
- **AND** the encoder MUST keep its existing internal GRU path

#### Scenario: no internal GRU path remains connected
- **WHEN** a run sets `use_internal_gru=false`
- **THEN** the resolved summary MUST show the no-internal-GRU path
- **AND** the encoder output MUST still have shape `[B,T,D]`
- **AND** final CSI feature norm diagnostics MUST be nonzero for nonzero CSI input

### Requirement: CSI view fusion warmup diagnostics
CSI view fusion warmup MUST preserve nonzero feature flow and expose gate/fusion diagnostics. Warmup MUST not zero both views or produce a zero fused feature for nonzero CSI input.

#### Scenario: view gate warmup keeps feature flow
- **WHEN** `view_gate_warmup_epochs` is active and the input CSI batch is nonzero
- **THEN** diagnostics MUST report nonzero freq_feat norm
- **AND** diagnostics MUST report nonzero delay_feat norm unless delay view is separately disabled
- **AND** diagnostics MUST report nonzero fused_feat norm

#### Scenario: gate broadcast is valid
- **WHEN** `view_fusion=symmetric_gate` or warmup mean fusion is active
- **THEN** gate or equivalent fusion weights MUST broadcast over `[B,T,D]` features correctly
- **AND** diagnostics MUST not report NaN values for gate, fused feature or final CSI feature
