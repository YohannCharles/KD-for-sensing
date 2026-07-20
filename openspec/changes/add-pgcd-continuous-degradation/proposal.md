## Why

现有纯 missing-modality 筛选中，样本级动态权重未稳定超过 learned global prior，二值可用性也无法提供连续质量信号。需要在 MMW 的自然天气与在线传感器退化下，用 64-beam topology prototype evidence 衡量实际决策漂移，并以严格的 dynamic-vs-global-mean 替换实验判断动态可靠性是否具有真实价值。

## What Changes

- 为 MMW 四模态 `image/radar/gps/lidar` 增加确定性、在线、传感器特定的 L0-L4 连续退化；L4 与现有 missing availability 语义对齐，且不修改原始数据或划分。
- 复用当前 block feature、64-beam prototype logits 与 topology distance，构造 clean/corrupted 双 view 的 topology drift、task degradation 和组合质量 target；clean teacher 仅 stop-gradient，不引入独立 teacher。
- 增加轻量质量估计与非负 reliability 调制，在 learned block prior 上实现 masked dynamic fusion，并支持 C0-C7 八个受控方向。
- 增加 seen/unseen、severity、weather、stale-frame、历史 missing 和 dynamic replacement 诊断，以及质量相关性、单调性和梯度一致性输出。
- 对配置和 batch 增加 channel/CSI/path/channel-gain 输入与 target 的 fail-fast 防泄漏检查；唯一通信监督保持为最优 beam index 与不依赖信道文件的 64-beam topology。
- 提供单 seed、inner/development、claim-ineligible 的预检、评测和 GPU 0-7 launcher；实现和验证阶段不自动启动八个完整训练任务，也不启动 multi-seed 或 outer test。

## Capabilities

### New Capabilities

- `pgcd-continuous-degradation`: 定义 MMW 四传感器连续退化、prototype-guided 质量学习、可靠性感知融合、固定快筛与诊断协议。

### Modified Capabilities

- `t2-baseline-surface`: 将 PGCD 作为 active T2 inner/development 研究任务纳入可追溯 current surface，同时保持四模态、canonical recipe 和本地产物边界。
- `u-mask-beam-jepa`: 允许 active PGCD 分支复用 block prototype evidence，并增加受控 clean/corrupted 双 view、质量监督与可靠性融合 payload。

## Impact

变更将涉及 `src/kd_sensing/data/` 的在线退化与 batch 契约、`src/kd_sensing/models/` 和 `src/kd_sensing/losses/` 的 PGCD 分支、`configs/mmw/` 的 tracked recipe、`scripts/` 的本地 launcher/evaluator，以及相应测试。实现不新增第三方依赖，不新增 public CLI，不读取 channel/path 文件，不改变 MMW train/validation/test split，也不将 `dataset/`、`outputs/`、日志或 checkpoint 纳入源码。
