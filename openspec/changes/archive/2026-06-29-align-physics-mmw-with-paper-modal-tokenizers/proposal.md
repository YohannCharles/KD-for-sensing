## Why

当前 `pinn_multimodal_beam` 已具备路径参数头、可微信道合成器和物理监督损失，但多模态感知侧仍主要依赖每个模态的统计量小编码器，难以支撑“多模态感知 - 路径推断 - 窄带阵列信道重构 - 波束选择”的可解释链路。

本变更将物理一致 MMW baseline 的模态编码方式对齐到 arXiv:2603.29796 风格的“模态专用 tokenizer + 模态/位置嵌入 + 共享 Transformer”结构，同时按本项目约束使用已有 `jepa_context_image` 作为图像编码器，不使用 GPS context。

## What Changes

- 将 `pinn_multimodal_beam` 的多模态前端从统一统计编码器升级为可配置的模态 tokenizer。
- 图像模态使用项目已有 `jepa_context_image` 轻量 ViT 风格编码器，要求通过 checkpoint 预训练或显式标记为非正式 smoke/debug；图像分支不得使用 GPS query/context。
- Radar、LiDAR、GPS、RF/CSI 分支采用论文风格的轻量 tokenizer：2D CNN 或 Linear + LayerNorm，将各模态映射到统一 token 维度。
- 引入共享 Transformer fusion core，对模态 token 加入 modality embedding、time/position embedding 和可选局部位置 embedding。
- 保留现有路径参数头、可微信道合成器、direct/physics/hybrid logits 和 physics-informed loss bundle；只把前端感知和 fusion 结构升级。
- 明确无线输入边界：完整窄带阵列 CSI 仍默认只作为 `csi_target` 监督；模型输入使用 sparse pilot / sparse antenna / RF scan 等受限观测；`oracle_full` 只作为上界。
- 增加对应配置、metadata 和 focused smoke，区分 paper-style tokenizer baseline、oracle upper-bound 和非正式 debug。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `physics-informed-mmw-beam-baseline`: 增加 paper-style 模态 tokenizer、共享 Transformer fusion、JEPA image tokenizer 和受限无线输入到物理链路的需求。
- `model-architecture-extension-contract`: 细化 `pinn_multimodal_beam` whole-model exception 的可组合前端要求、metadata 和测试要求。
- `experiment-workflow`: 增加 paper-style physics MMW 配置、预训练 checkpoint 约束、oracle/debug 标记和实验分层要求。

## Impact

- 主要影响 `src/kd_sensing/models/pinn_multimodal_beam.py`、现有 encoder registry、物理 helper、配置文件和 `tests/test_physics_informed_mmw.py`。
- 需要复用 `src/kd_sensing/models/jepa.py` 中的 `jepa_context_image`，并配置为 mean pooling 或其它不依赖 GPS context 的 pooler。
- 不新增根目录训练脚本，不复制训练循环，不读取真实 `dataset/` 作为测试依赖。
- 不声明完整宽带 CSI 重构；本变更的物理监督目标限定为 MMW 数据当前可提供的窄带阵列信道 `[T, Nsc, Nant, 2]`。
