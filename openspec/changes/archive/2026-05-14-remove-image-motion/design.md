## Context

当前仓库同时存在 RGB/ImageNet image 路径、legacy `motion_mask` profile、image motion cache、预处理入口和 `motion_cnn` encoder。`image_motion` 相关字段还被 cache policy、run metadata、diagnostics、OpenSpec specs 和 README 引用，导致 image 模态主路径需要持续绕过或兼容一个适用面较窄的方法。

本变更是有意的破坏性删除：用户会重新跑实验，因此不保留旧配置、旧 checkpoint、旧 cache 或旧模型注册名的运行兼容。历史 `outputs/` 只是实验产物，不在本次清理范围内。

## Goals / Non-Goals

**Goals:**

- 删除所有 `image_motion_*` 配置字段、cache 目录解析、cache key、metadata、预热命令和文档说明。
- 删除 `motion_mask` profile 和依赖单通道 motion mask 的 image encoder/model 分支。
- 让 image 模态只有 RGB/ImageNet 输入契约；包含 image 的训练、评估、fusion 和诊断路径都使用同一语义。
- 更新 specs、测试和文档，确保仓库源码中不再有可运行的 image motion 路径或兼容承诺。

**Non-Goals:**

- 不删除、迁移或重写 `outputs/` 下历史实验结果。
- 不提供旧 checkpoint 到新 RGB encoder 的自动迁移。
- 不为旧配置提供 deprecation warning 或 fallback。
- 不改变 LiDAR BEV cache、beam label cache、radar/GPS/mmWave/LiDAR 数据处理能力。

## Decisions

1. 直接删除而不是保留禁用开关。

   - 决策：移除 `image_motion_use_cache`、`image_motion_write_cache`、`image_motion_cache_dir`、`motion_mask` profile、`motion_cnn` 注册名和对应测试。
   - 理由：用户明确要求“不保留任何兼容”，保留开关会继续扩大配置矩阵和测试面。
   - 备选：保留 profile 但默认关闭。该方案仍要求维护 dataset、cache policy、batch padding、encoder 校验和 checkpoint 兼容逻辑，不符合目标。

2. image 模态契约统一为 RGB/ImageNet。

   - 决策：`resolve_image_profile` 相关逻辑应收敛为固定 RGB profile，或删除 profile 选择层，只保留必要的 RGB image loader/normalization 契约。
   - 理由：后续 ResNet-18 image architecture 可以成为唯一 image 主路径，避免 motion mask 和 RGB 两套 shape、通道和时间长度语义并存。
   - 备选：保留 `image_profile: rgb_imagenet` 字段作为显式配置。实现时可以短期保留该字段，但不得接受 `motion_mask` 或其它旧兼容值。

3. cache policy 只处理仍支持的 cache 类型。

   - 决策：自动 cache policy 不再解析 image cache；`data.cache.image.*` 和 dataset 低层 `image_motion_*` 字段删除。
   - 理由：RGB image 路径直接从原始帧读取和标准化，不产生 image motion cache artifact。
   - 备选：把 image cache policy 解释为 RGB frame cache。该能力不在本需求内，会引入新的 artifact 语义。

4. 诊断显示真实 RGB image 输入。

   - 决策：manifest 继续基于 Dataset 实际返回张量生成记录，但 image 的 processed 表示从 motion mask 改为 RGB/ImageNet tensor 或可视化派生物。
   - 理由：诊断应反映训练输入；删除 motion mask 后不能再引用旧 processed mask。
   - 备选：继续展示 raw image 并忽略 processed image。该方案会削弱“训练输入一致”的诊断价值。

5. Active OpenSpec change 必须同步清理。

   - 决策：实现时需要更新或重写 `openspec/changes/add-resnet18-image-architecture` 中关于 legacy motion 兼容、motion profile 和 `motion_cnn` adapter 的任务与 specs。
   - 理由：该 active change 当前明确要求保留 motion mask 兼容，和本变更冲突。
   - 备选：忽略 active change。后续 archive 或 apply 时会重新引入被删除能力。

## Risks / Trade-offs

- [Risk] 旧 image-only 或 image+radar 配置直接失效。→ Mitigation：配置解析阶段对 `motion_mask`、`motion_cnn`、`image_motion_*` 给出清晰错误，文档列出需要改用 RGB/ImageNet 配置。
- [Risk] 测试中大量 fixture 依赖 motion mask shape。→ Mitigation：先删除低价值 cache 单测，再用最小 RGB dataset/model smoke test 覆盖 image 主路径。
- [Risk] 文档或 OpenSpec 中残留 image motion 说明。→ Mitigation：实现任务包含全仓 `rg` 检查，允许 archive 目录保留历史记录，但当前 specs、README、configs、src、tests 不得残留可执行引用。
- [Risk] 旧 checkpoint 路径仍被默认配置引用。→ Mitigation：移除或重命名默认 image/fusion checkpoint registry 中的 motion 分支引用，要求重新训练。

## Migration Plan

1. 删除 image motion 数据转换、cache、preprocessor 和 CLI 类型。
2. 删除 dataset/config/cache policy/run metadata/batch 准备中的 `image_motion_*` 和 `motion_mask` 分支。
3. 删除 motion image encoder 和 modular model 中的 motion adapter 默认路径。
4. 更新所有 configs、README、docs、diagnostics 和 tests。
5. 更新 `add-resnet18-image-architecture` active change，确保它不再提出 motion 兼容要求。
6. 运行 `conda run -n kd_mm_beam pytest ...` 和 `openspec validate remove-image-motion --strict`。

Rollback 策略：本变更不提供运行时回滚。若确实需要恢复，只能从版本控制恢复删除的代码和 specs。

## Open Questions

无。用户已经明确接受破坏性删除和重新跑实验。
