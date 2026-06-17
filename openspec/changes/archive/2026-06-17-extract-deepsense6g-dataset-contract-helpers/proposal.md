## Why

`DeepSense6GDataset` 当前同时处理 CSV audit、模态字段校验、GPS feature mode、beam target source、cache path、soft target、LiDAR/CSI/GPS 读取和 metadata 拼装。它承担太多数据契约细节，导致后续修改数据字段、BeamBench GPS Direct 或 2604 BEV XY 相关逻辑时，Codex 需要在一个超长类里定位纯配置/校验逻辑，误伤样本语义的风险较高。

## What Changes

- 从 `DeepSense6GDataset` 中抽出低风险、无状态或近似无状态的 dataset contract helper。
- 首批 helper 覆盖 GPS feature mode normalization、beam target source normalization、必要列校验、scene calibration/GPS angle offset、cache path resolution 和 sample metadata parsing。
- `DeepSense6GDataset` 继续负责 orchestration、实际资源读取、`__getitem__` 和对外样本结构。
- 新 helper 必须保持轻量，不导入 torch dataset、训练循环、模型或真实数据。
- 不改变 CSV schema、target label、sample id、metadata 字段、cache 内容或训练/评估数值。

## Capabilities

### New Capabilities

### Modified Capabilities

- `dataset-runtime-contracts`: 明确当前保留 dataset 可将契约 normalization/validation/helper 从 dataset 类中拆出，同时保持 flat sample 和 target contract 不变。
- `project-health-guardrails`: 热点预算和架构边界需要鼓励 dataset contract helper 拆分，并防止 helper 又回流到超长 dataset 类。

## Impact

- 主要影响 `src/kd_sensing/data/datasets/deepsense6g.py` 和新增 `src/kd_sensing/data/datasets/deepsense6g_contract.py`、`deepsense6g_gps.py`、`deepsense6g_columns.py` 或等价 helper 模块。
- 影响 DeepSense6G dataset focused tests、BeamBench Image AE+GPS 配置 characterization、architecture boundary hotspot budget。
- 不读取真实 `dataset/`，测试应使用 synthetic dataframe、tmp path 或现有 unit fixtures。
