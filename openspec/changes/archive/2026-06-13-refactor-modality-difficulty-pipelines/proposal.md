## Why

后续研究重点会转向提高 GPS 学习难度（noise、延迟、间隔采样、missing/stale）和 image 输入难度，以放大 JEPA 视觉表征相对 GPS shortcut 的优势。当前 CSI/LiDAR 的输入退化、JEPA GPS shortcut benchmark 的 GPS/image perturbation、训练/评估配置解析分散在不同模块里，继续堆新难度会让 dataset、diagnostics runner 和模型分支互相耦合。

## What Changes

- 新增统一的模态难度管线能力：用配置化 profile/condition 描述 GPS 与 image 等输入扰动，并通过可注册 batch transform 在训练、评估和 benchmark 中复用。
- 将 GPS noisy、cumulative drift、missing/dropout、temporal delay、low-rate/stride、random async、GPS distractor，以及 image fog/rain、night、occlusion、motion blur 等难度统一成可组合 suite/operator。
- 明确难度管线只改变输入模态或其 mask/metadata，不移动当前 beam label、power target、sample id 或 split 语义。
- 为所有难度条件写出 replay metadata、warnings、source index/timestamp、valid/stale/dropout mask 和 resolved profile digest，支持复现实验和论文图审计。
- 让 JEPA GPS shortcut benchmark 消费同一套难度管线，保留当前 manifest 语义和输出表格，但避免 runner 内部继续维护专属 perturbation 实现。
- 为 supervised train/evaluate 支持可选难度 profile，使 clean training、mild async training、GPS/image dropout training 与 evaluation-only benchmark 共用同一份配置和确定性种子规则。
- 不新增旧入口、根目录脚本或兼容聚合层；所有运行入口继续位于 `src/kd_sensing` 包内 CLI、现有训练/评估入口或 OpenSpec 声明的 workflow。

## Capabilities

### New Capabilities
- `modality-difficulty-pipeline`: 统一描述、注册、解析和应用 GPS/image 等模态输入难度 profile/operator 的运行时能力，覆盖训练、评估和 benchmark 的 shared batch transform、metadata、determinism 和测试契约。

### Modified Capabilities
- `jepa-gps-shortcut-benchmark`: benchmark manifest 与 runner 需要改为引用统一难度 profile/operator，并保持现有 perturbation suite、Scenario C preset、指标和产物兼容。
- `dataset-runtime-contracts`: runtime dataset/dataloader metadata 需要记录 resolved difficulty profiles、作用 split/stage、输入 mask 与 replay metadata，并约束 target 不随输入难度移动。
- `component-registry`: 组件注册表需要新增或扩展难度 operator/profile 注册边界，同时保持 registry 轻量导入。
- `canonical-config-resolution`: 配置加载、overlay 和 validation 需要解析训练/评估难度 profile，拒绝未知 operator、非法 split/stage 和会移动 target 的配置。
- `modality-contracts`: 模态契约需要明确难度 profile 复用 canonical modality keys，不新增伪模态名称，并为 mask/metadata 字段提供中心化语义。

## Impact

- 主要代码影响：`src/kd_sensing/data/` 或新的窄模块用于 difficulty transforms，`src/kd_sensing/engine/data_factory.py`、`src/kd_sensing/engine/evaluation_pass.py`、训练 batch path、`src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py`、`src/kd_sensing/config/validation.py`、`src/kd_sensing/registries.py` 和相关 CLI/config examples。
- 测试影响：新增 synthetic batch determinism、shape/target preservation、Scenario C no-future-GPS、profile validation、registry import-lightness、benchmark manifest compatibility，以及训练/评估 smoke/focused tests。
- 产物影响：新增 metadata、warnings 和 profile digest 写入 ignored `outputs/` 或 manifest 指定目录；不提交真实数据、训练输出、cache 或 checkpoint。
- 兼容性：现有 JEPA shortcut benchmark manifest 的 suite type 与 Scenario C canonical preset 应继续可解析；内部实现可迁移到统一管线。
