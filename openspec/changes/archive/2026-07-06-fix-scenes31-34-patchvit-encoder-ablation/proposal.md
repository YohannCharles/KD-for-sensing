## Why

上一轮 Scene31-34 encoder 对照误用了 `tinyvit_5m_scratch_rgb`，而用户实际需要的是仓库已有的轻量 ViT 风格编码器：一层卷积 patch/stem 加一层 Transformer。需要在不停止现有 TinyViT 长跑的前提下，补上正确 PatchViT 对照组并启动对应预训练与 downstream。

## What Changes

- 新增可注册的 `lightweight_patchvit_frame` encoder，复用已有 `patch_vit` visual token encoder，并将 token mean-pool 为模块化模型需要的 `[B, T, D]` 帧级特征。
- 新增 Scene31-34 PatchViT encoder ablation 生成脚本，产出 image/lidar 单模态预训练配置、普通 downstream 配置和 PatchViT+JEPA downstream 配置。
- 新增 focused tests，覆盖 encoder 构建/forward 语义、manifest run 名、PatchViT encoder 类型、checkpoint path 和 JEPA flag。
- 训练和日志继续写入 ignored `outputs/` root；现有 TinyViT-5M 训练不被停止或覆盖。

## Capabilities

### New Capabilities
- `scene31-34-patchvit-encoder-ablation`: Scene31-34 主 workflow 的轻量 PatchViT encoder 对照组，本地生成配置并按 pretrain/downstream 两阶段并行执行。

### Modified Capabilities
- `component-registry`: 默认组件导入后应能发现 `lightweight_patchvit_frame` / `patchvit_frame` encoder。
- `scenes31-34-main-missing-modality-workflow`: 增补正确的 PatchViT encoder ablation local/manual workflow，区别于已有 TinyViT-5M 对照。

## Impact

- 影响模型注册与配置生成：`src/kd_sensing/models/jepa.py`、`scripts/generate_scenes31_34_patchvit_ablation.py`、focused tests。
- 不改变训练循环、dataset contract 或现有 TinyViT 运行产物。
- 新生成 checkpoint、日志和 config 仍是本地 runtime artifacts，不进入源码提交。
