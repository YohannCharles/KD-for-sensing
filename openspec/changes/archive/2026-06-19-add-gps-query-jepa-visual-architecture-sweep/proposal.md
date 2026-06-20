## Why

当前 GPS-query JEPA 下游主线使用轻量 `VisualPatchTokenEncoder`：224x224 输入、非重叠 patch16、浅层 Transformer、再经 GPS-query 或 predictive pooler 压成帧级特征。已有同口径结果显示 JEPA GPS-query 尚未稳定超过 `Image ResNet+GPS`，因此需要一次受控的视觉架构 sweep，系统性判断瓶颈来自 patch 粒度、局部归纳偏置、pooling 方式、时序上下文还是 GPS shortcut。

用户希望把候选架构一并跑完后择优保留；本变更将这些候选收敛成可配置、可比较、可诊断的实验矩阵，而不是提前把结论固定为 CNN+Transformer。

## What Changes

- 新增 GPS-query JEPA visual architecture sweep 能力，覆盖所有项目内可实现、同口径可跑的实用候选族：
  - baseline：现有 patch16 JEPA mean / GPS-query / Predictive GPS-query++。
  - patch/token 粒度：patch16、patch14、patch8、小分辨率/高分辨率组合和 `max_tokens` 约束。
  - tokenizer 改造：重叠 patch embedding、stacked 3x3 conv stem、compact convolutional tokenizer。
  - 局部归纳偏置：LocalViT-style depthwise FFN、CvT-style convolutional QKV/projection、条件位置编码或相对位置偏置。
  - CNN token backbone：ResNet18/ResNet34 或 ConvNeXt-like stage feature map tokens，经投影后接 GPS-query pooler。
  - 多尺度视觉 token：layer3/layer4 或 low/high resolution tokens 合并后接 GPS-query pooler。
  - pooling/head ablation：mean、GPS-query K=2/K=4、content+GPS residual、Predictive GPS-query++、可选 K-token 保留给 fusion core。
  - 非 Transformer 对照：ConvNeXt/ResNet 帧级 embedding、SE/dynamic-conv-like lightweight CNN gate、纯 GPS 和 Image ResNet+GPS anchor。
- 新增派生配置族，要求所有候选复用相同 split、label space、metric profile、history window、seed、GPS feature mode、训练 recipe 关键字段和输出目录边界。
- 新增 architecture sweep manifest，记录每个候选的架构类别、token grid、token_count、pooler、参数量/FLOPs 近似、checkpoint reuse/freeze policy、比较组和运行命令。
- 新增诊断输出要求：attention/activation peakiness、token entropy、branch/gate weights、相邻 beam error、Top-1/3/5、DBA、P0-P5 或 strict condition metrics。
- 不恢复旧 KD、HiST/Hist、Top8 selector、camera residual、root-level 旧脚本或独立训练循环；新增能力优先作为 `modular_sequence` 的 encoder/pooler/core component baseline 或配置派生实现。
- 不把本地训练产物、checkpoint、log、cache 或 sweep 结果纳入源码变更。

## Capabilities

### New Capabilities

- `jepa-visual-architecture-sweep`: 定义 GPS-query JEPA visual encoder/pooler 架构候选矩阵、配置派生、manifest、比较口径、诊断字段和保留/淘汰规则。

### Modified Capabilities

- `gps-conditioned-jepa-pretraining`: 扩展 JEPA visual encoder 配置契约，使 Stage 1 可 opt-in 使用 patch/conv/overlap/local-attention/tokenizer variants，并记录 tokenizer metadata；默认 patch16 行为保持兼容。
- `jepa-downstream-extensibility`: 扩展 supervised downstream reuse，使 `jepa_context_image` 可消费新 visual token encoder 输出、CNN feature-map tokens、多尺度 tokens 或 K-token pooler 输出，同时保持默认 `[B,T,D]` 契约兼容。
- `configurable-multimodal-fusion`: 增加 architecture sweep 派生配置与严格可比性要求，确保新增候选通过 config/component baseline 表达，不新增 whole-model 例外或旧入口。

## Impact

- 受影响模型：`src/kd_sensing/models/jepa.py`、`src/kd_sensing/models/jepa_downstream.py`、`src/kd_sensing/models/image_encoders.py` 和相关 encoder/pooler/core registry。
- 受影响配置：`configs/pretraining/` 中 GPS-conditioned JEPA visual encoder variants，`configs/fusion/experiments/jepa_image_gps/` 中 supervised downstream sweep configs 和 manifest。
- 受影响诊断：JEPA visual analysis、GPS-query attention diagnostics、strict benchmark/claim gate 表格和 architecture sweep summary。
- 受影响测试：配置加载、registry build、synthetic forward shape、checkpoint compatibility、pooler metadata、architecture boundary 和 focused diagnostic tests。
- 运行产物：训练结果、logits、attention map、summary CSV/JSON/PNG 和 checkpoint 写入 ignored `outputs/analysis/jepa_visual_architecture_sweep/` 或配置指定 ignored 目录。
