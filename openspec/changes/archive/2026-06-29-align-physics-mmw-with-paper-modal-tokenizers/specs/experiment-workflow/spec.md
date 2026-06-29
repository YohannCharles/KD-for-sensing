## ADDED Requirements

### Requirement: Paper-style physics MMW experiment configs
项目 SHALL provide opt-in configs for the paper-style physics-informed MMW baseline. These configs MUST use current `kd-sensing-train` / evaluation entry points, MUST keep sparse or restricted wireless observation separate from full CSI supervision, and MUST record whether the run is formal, debug/smoke, or oracle upper-bound.

#### Scenario: sparse-pilot multimodal 配置
- **WHEN** 用户加载 paper-style sparse-pilot multimodal physics MMW 配置
- **THEN** final config MUST enable `model.primary.type=pinn_multimodal_beam`
- **AND** model config MUST enable paper-style tokenizer frontend and shared Transformer fusion
- **AND** data config MUST use restricted CSI input such as `sparse_pilot` rather than default full current CSI

#### Scenario: debug 配置不可进入正式结论
- **WHEN** debug/smoke 配置允许随机初始化 `jepa_context_image` 或使用 synthetic batch
- **THEN** run metadata MUST mark `formal_experiment_eligible=false`
- **AND** report MUST NOT be treated as formal paper-style baseline evidence

#### Scenario: oracle 配置明确标记
- **WHEN** oracle full CSI input is explicitly enabled for upper-bound comparison
- **THEN** run metadata MUST mark `oracle_upper_bound=true`
- **AND** run metadata MUST mark `main_conclusion_eligible=false`
- **AND** summary MUST state that current full CSI was used as model input

### Requirement: Paper-style physics MMW validation
Paper-style physics MMW implementation MUST include focused validation that does not depend on real `dataset/` contents. Validation MUST cover config loading, registry build, synthetic forward, physics loss/backward, output adaptation, metadata, and shape handling for `[B, T, Nsc, Nant, 2]` CSI targets.

#### Scenario: synthetic forward/loss smoke
- **WHEN** focused tests construct a synthetic paper-style physics MMW batch
- **THEN** model forward MUST produce finite logits, `path_hat` and `h_hat`
- **AND** physics-informed loss MUST complete backward with finite gradients
- **AND** tests MUST NOT read real `dataset/`

#### Scenario: JEPA image tokenizer without GPS context smoke
- **WHEN** focused tests build the image tokenizer path
- **THEN** test config MUST use `jepa_context_image` with a non-GPS pooler
- **AND** forward MUST succeed without `gps_condition_features`
