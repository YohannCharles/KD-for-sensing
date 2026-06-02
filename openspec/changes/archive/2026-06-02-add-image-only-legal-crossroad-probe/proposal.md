## Why

现有 v8/v9 快速验证同时受多模态融合、target oracle 字段误判、support outlier、prototype 粒度和 source prior collapse 影响，且合法汇总中 `eligible_run_count=0`。本变更先建立一个干净的 image-only 合法 few-shot 适配实验平台，用最小可控路径判断 image backbone 加小输出头是否能稳定迁移。

## What Changes

- 新增 `image_only_legal_crossroad_probe` 配置与运行脚本，统一运行 I0 source-only、I1 target linear probe、I2 V8 target prior head、I3 V9 sector proto 四个 image-only 对照。
- 新增 image-only 数据路径约束：batch 只向模型、loss、adaptation 和 eval 暴露 image、beam、scene、sample_id、split 等合法字段，不使用 GPS、LiDAR、radar、mmWave、CSI、channel、path 或 beam_power。
- 新增或收敛 HiST-Beam image-only variant 行为，复用 image encoder/backbone，默认 `identity` fusion，支持 frozen backbone + small head target adaptation。
- 将现有 A2、V8 A3 target prior head、V9 sector proto 搬到合法 image-only 协议下，不新增 v10、新 beam-level prototype 主线、pseudo-label self-training 或 image backbone full fine-tuning。
- 新增 image backbone feature cache，用于 source/head probe 与 target support/test evaluation；cache 不保存禁用模态或 path/radio 字段，且 adaptation 不得读取 target_test cache labels。
- 修复 eligibility checker：依据实际 consumed fields 和 split metadata 判断合法性，而不是依据原始数据文件中是否存在 path/radio/channel/beam_power 字段。
- 所有 run 写出 prediction histogram、confusion-by-true-beam、run summary、eligibility 结果，并汇总到 `combined_summary.csv` 与可选 histogram summary。

## Capabilities

### New Capabilities

- `image-only-legal-crossroad-probe`: 定义 image-only 合法 crossroad probe 的配置、四模式运行矩阵、feature cache、诊断产物、汇总和成功标准。

### Modified Capabilities

- `modality-aware-data-loading`: 明确 image-only probe 下 dataset/collate/batch preparation 只暴露启用模态和合法标签字段，禁用字段存在于原始数据中不等于被模型使用。
- `hist-beam-cross-scene-adaptation`: 增加 image-only source-only、target linear probe、V8 target prior head、V9 sector prototype 的模型输出、冻结策略、训练参数和诊断契约。
- `mmw-cross-scene-adaptation-protocol`: 强化 quick validation eligibility audit，要求基于实际 target-side oracle usage 和 strict split eligibility 给出可审计结论。

## Impact

- 受影响配置与脚本：`configs/hist_beam/`、`scripts/` 中新增 image-only probe 配置和运行入口。
- 受影响代码：MMW/HiST-Beam dataset 与 collate/batch preparation、HiST-Beam 模型注册和 forward 输出、target adaptation、feature cache、evaluation metrics/diagnostics、eligibility checker 与 summary writer。
- 受影响产物：每个 run 的 `metrics.json`、prediction 文件、`prediction_hist.json`、`confusion_by_true_beam.json`、eligibility metadata，以及总目录下的 `combined_summary.csv`。
- 不引入新的外部依赖，不删除现有多模态路径，不改变旧 v7/v8/v9 多模态实验入口的默认行为。
