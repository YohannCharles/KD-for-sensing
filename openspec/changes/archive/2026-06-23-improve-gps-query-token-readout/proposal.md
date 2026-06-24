## Why

当前 `pooler_gps_query_k2_tokens` 已经证明 GPS-query token 输出比 mean pooling 更强，但相对 `gps_query_k2_frame` 只提升到千分位到百分点级，说明瓶颈不再是简单增加 query 数，而是 K 个 GPS-conditioned token 是否被下游 readout 真正利用。现在需要把“query token 保留、读出、诊断、claim gate”收敛成可实施的改进方案，避免继续盲目堆 `k_queries`、heads 或新增大模型。

## What Changes

- 为 GPS-query K-token 路径增加明确的 token readout 改进路线：保留现有 `output_mode: tokens` 和 `token_aware_transformer`，新增更小的 opt-in learned readout / query-weighted readout 候选，而不是重写 JEPA 或恢复旧路线。
- 增加 query token 诊断和证据门控：记录 query diversity、attention entropy、query 间余弦相似度、每个 query 的 DBA/Top-k 贡献 proxy、readout 权重和失败样本类别。
- 扩展 sweep/benchmark 矩阵：以 `pooler_mean`、`pooler_gps_query_k2_frame`、`pooler_gps_query_k2_tokens` 为最小 paired baseline，补充最小 readout 候选、seed 复核和 Scene31 / S32-S34 分组结果。
- 保持默认行为不变：未显式启用 token readout 候选时，`GPSQueryPool` 默认仍输出 `[B,T,D]`，现有 mean、frame GPS-query、hybrid residual query 和 Predictive GPS-query++ 配置语义不变。
- 不新增外部依赖、不提交 checkpoint/cache/output；所有训练、评估和诊断产物继续写入 ignored `outputs/`。

## Capabilities

### New Capabilities

- 无。本 change 在现有 JEPA downstream 与 GPS-query pooling 能力内收窄增强，不新增独立当前能力。

### Modified Capabilities

- `gps-query-jepa-pooling`: 增加 GPS-query token 输出路径的 readout、query diversity 诊断和 paired evidence 要求。
- `jepa-downstream-extensibility`: 扩展 K-token downstream fusion opt-in 契约，要求 token readout candidate、兼容性校验、metadata 和默认兼容边界。
- `jepa-visual-architecture-sweep`: 增加 GPS-query token/readout ablation 的最小候选矩阵、严格可比性字段和 claim gate。

## Impact

- 代码：`src/kd_sensing/models/jepa_downstream.py`、`src/kd_sensing/models/modular.py`、`src/kd_sensing/diagnostics/cnn_hybrid_jepa_visual_prior_sweep.py`、`src/kd_sensing/diagnostics/jepa_visual_analysis.py` 或窄 helper。
- 配置：`configs/diagnostics/cnn_hybrid_jepa_visual_prior_sweep_manifest.yaml`、必要的 `configs/fusion/experiments/jepa_image_gps/` 派生配置或 generator 生成规则。
- 测试：GPSQueryPool/token readout shape 与 metadata focused tests、配置加载 characterization、visual architecture sweep manifest tests、诊断 evidence synthetic tests。
- 验证：`conda run -n kd_mm_beam pytest tests/test_gps_conditioned_jepa.py tests/test_jepa_visual_architecture_sweep.py tests/test_config_load_characterization.py -q`，必要时追加 `tests/test_jepa_visual_analysis.py`。
