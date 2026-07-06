## Why

当前仓库跟踪 267 个 YAML，其中 Scene31、RBMA/KD/BTAPA、JEPA image+GPS 和多类 local/manual experiment config 占据大量源码表面。缺失模态主线已经完成第一轮入口收口后，下一步需要把实验配置族收缩为“当前证据需要、可复现实验需要、诊断需要、或可由 generator/manifest 重建”的小而可审计表面。

本 change 目标是缩小实验配置族，而不是删最多 YAML：保留 claim/evidence 和当前复现路径，删除或转为生成产物的重复实体 YAML，确保 docs、tests、config doctor 和 result provenance 一致。

## What Changes

- 审计并分类配置族：
  - `configs/scene31/`
  - `configs/fusion/experiments/rbma_missing_workflow/`
  - `configs/fusion/experiments/rbma_missing_workflow_strong_encoders/`
  - `configs/fusion/experiments/jepa_image_gps/`
  - 与这些族直接相关的 generator、manifest、README/docs 和 claim provenance。
- 将每个配置标记为 canonical/current、paper/workflow reproduction、claim/evidence input、diagnostics manifest、local/manual overlay、generated/recipe-backed、historical 或 delete-candidate。
- **BREAKING**：删除可由 generator/template/manifest 无损重建且无 current claim/docs/spec/test 依赖的实体 YAML；不新增 virtual alias 或旧路径兼容 wrapper。
- 对仍保留的实体 YAML 记录保留理由、owner、输出边界、真实数据/本地 checkpoint 依赖和删除触发条件。
- 更新 generator focused tests，使保留的 generator/manifest 能覆盖 run name、seed、epoch、sampler、loss weight、missing pattern、dataset split、output boundary 和关键 overrides。
- 更新 docs/experiment matrix/mainline catalog/result claims，使 current 文档指向保留实体 YAML、generator/manifest 输入或明确 local/manual 状态，不指向已删除配置。

## Capabilities

### New Capabilities

- 无。本 change 只收缩现有配置支持面。

### Modified Capabilities

- `canonical-config-resolution`：加强 generated/recipe-backed config family 的等价验证、删除边界和 config doctor 分类要求。
- `project-surface-cleanup`：增加实验配置族收缩的删除证据、保留理由和本地产物保护边界。
- `mainline-experiment-documentation`：要求实验矩阵、主线目录和 claim provenance 在配置删除/生成化后继续指向真实 current/reproduction/local-manual 入口。

## Impact

- 影响范围：`configs/scene31/`、`configs/fusion/experiments/rbma_missing_workflow*`、`configs/fusion/experiments/jepa_image_gps/`、相关 generator scripts、config focused tests、`docs/project_surface_inventory.md`、`docs/mainline_model_catalog.md`、`docs/experiment_matrix.md`、`docs/result_claims_registry.md`。
- 不影响范围：runtime outputs、logs、checkpoint、dataset、本地 cache、训练数学语义、配置加载 migration guards 和 retired-route 拒绝语义。
- 兼容性：部分历史或 generated 实体 YAML 会从源码树删除；复跑路径应通过保留的 generator/manifest/base config 或保留的 evidence YAML 表达。
- 验证：OpenSpec strict validate、config load characterization、generator focused tests、architecture boundary、surface/config doctor 和引用路径一致性检查。
