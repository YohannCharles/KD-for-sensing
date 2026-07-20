## Why

现有 MMW T2 的 64-beam 原型拓扑损失已有效，但时间池化后的 confidence/prototype-center Router 在联合缺失下接近静态先验，无法利用逐时间块原型证据。需要在不复制模型和训练循环的前提下，用统一的时间块缺失协议快速验证完整到缺失视图的一致性与反事实贡献监督是否能改善硬缺失场景。

## What Changes

- 在 T2 内新增 opt-in 的 PCER 时间块融合路径：复用 `[B,T,M,D]` 编码特征和现有 `[64,D]` prototype bank，产生 `[B,T,M,64]` 块级原型证据。
- 新增可配置、可复现的六类时间块 availability mask 与三阶段 curriculum；训练和固定评测共享样本身份、seed 和 mask 语义。
- 新增完整视图到缺失视图的 prototype-logit KL，以及基于缓存块证据向量化计算的 leave-one-block-out 拓扑贡献 Router 监督。
- 新增 A0 静态、A1 旧 Router、A2 一致性静态、A3 完整 PCER 四组 MMW 15-domain quick-validation 配置生成、GPU4--7 启动、固定缺失评测、Router 诊断和汇总。
- 保持 canonical `configs/mmw/t2.yaml`、旧 Router、默认 40-epoch outer evidence 和已有 checkpoint 行为不变；新产物仅写入 ignored `outputs/quick_pcer_validation/`。

## Capabilities

### New Capabilities

- `pcer-temporal-block-fusion`: 定义六类时间块缺失、逐块波束原型证据、完整/缺失一致性、反事实 Router、固定评测与 quick-validation 证据边界。

### Modified Capabilities

- `u-mask-beam-jepa`: 增加 PCER opt-in forward payload 和块级 mask/fusion 契约，同时保持 current 默认路径数值兼容。
- `training-evaluation-runtime`: 增加 PCER curriculum、同 backbone 完整视图监督、四组 quick-validation 与确定性逐样本 mask 评测。

## Impact

- 主要影响 `src/kd_sensing/data/temporal_missing*`、`src/kd_sensing/models/u_mask_beam_jepa.py`、`src/kd_sensing/losses/u_mask_beam_jepa*`、MMW quick-validation 本地 helper 与聚焦测试。
- 不新增第三方依赖，不修改 `dataset/`、canonical recipe、历史 checkpoint 或正式 claim；训练日志、resolved config、checkpoint、mask cache 和汇总均为本地产物。
