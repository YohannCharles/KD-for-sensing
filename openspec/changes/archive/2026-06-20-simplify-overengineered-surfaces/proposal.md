## Why

最近的全仓库 over-engineering 审计发现多处只有测试引用、没有当前入口消费、或由文档/配置重复撑起来的维护面。现在用一个独立 OpenSpec change 先收敛契约，避免实施时把删除、右尺寸化和入口迁移混成不可回滚的大改。

## What Changes

- 删除或收敛无当前调用方的源码表面：`communication_state_features`、未接入的 LiDAR pillar encoder 原型、未消费的 dataset runtime adapter 框架，以及重复的 `OutputRegistry`。
- 收窄 JEPA GPS shortcut benchmark facade：保留公开 runner/API，private helper 不再由 facade 重新导出，测试改为直接覆盖窄模块。
- **BREAKING**：退役 `scripts/*.py` thin alias，当前推荐入口收敛到 `pyproject.toml` 声明的 `kd-sensing-*` console scripts；README、AGENTS、docs、维护索引和架构测试同步改口。
- 将 CSI hardening 的重复实体 YAML 矩阵收敛为 base config + 轻量 overlay/recipe 表，同时保持现有 A/B/C/D/E 组逻辑配置可解析。
- 收缩架构边界测试中的 prose mirror：测试只验证机器可读治理表、路径存在性、AST/import 边界和关键生命周期，不再逐字断言 README/docs/OpenSpec 长段文案。
- 删除未使用 dev 依赖 `thop` 和 `pytorch-model-summary`。
- 更新维护索引、inventory、OpenSpec specs 和最小验证命令，让删除项被明确分类为 retired、merged、base+overlay 或 no-current-surface，而不是靠隐式缺失解释。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `project-surface-cleanup`: 增加低价值孤立模块、重复 helper、重复实验配置和 dev 依赖删除的收敛契约。
- `project-health-guardrails`: 将架构边界测试从 prose mirror 收缩为机器可读治理和结构扫描，并保留无运行副作用边界。
- `maintainer-context-index`: 更新 entrypoint、hotspot、merge-candidate、dependency 和 remediation wave metadata，以描述本次删减后的支持面。
- `project-architecture`: 将当前入口契约从 `scripts/*.py` thin alias 迁移到 package console scripts，并禁止新增脚本兼容包装。
- `dataset-runtime-contracts`: 允许以现有 dataset 实现和轻量 row/metadata helper 满足 runtime contract，不要求保留未接入的通用 `RuntimeDataset`/adapter framework。
- `lidar-preprocessing`: 明确当前 LiDAR 支持面是 BEV 伪图像与质量摘要；未接入的 pillar encoder 原型不属于当前必须保留 surface。
- `csi-hardening-experiment-matrix`: 允许 hardening matrix 由 base config + overlays/recipe 生成或解析，取代多份重复完整 YAML。
- `jepa-gps-shortcut-benchmark`: 明确 facade 只承载公开 runner/API，helper 实现与测试归属窄模块。

## Impact

- 受影响源码：`src/kd_sensing/data/dataset_runtime.py`、`src/kd_sensing/diagnostics/communication_state_features.py`、`src/kd_sensing/models/lidar_pillar_encoder.py`、JEPA benchmark facade/窄模块、诊断 artifact registry helper。
- 受影响入口：`scripts/train.py`、`scripts/evaluate.py`、`scripts/preprocess.py`、BeamBench 相关 thin aliases 以及对应文档入口；package console scripts 保留。
- 受影响配置：`configs/csi/hardening_matrix/` 与 `configs/fusion/csi_hardening_matrix/` 的重复实体 YAML 矩阵。
- 受影响测试：架构边界测试、CLI help smoke、CSI hardening config load tests、JEPA shortcut benchmark tests、相关 orphan module tests。
- 依赖影响：删除 dev extra 中未使用的 `thop` 与 `pytorch-model-summary`；runtime dependencies 不变。
- 本 change 不删除 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或历史权重；所有验证继续避免读取真实本地数据。
