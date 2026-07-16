## Why

历史 H4 筛选显示优化器、权重衰减和学习率日程的组合具有开发价值，但现有 MMW launcher 会静默重写这些字段，无法将 H4 作为可审计的主线训练配方。与此同时，T2 的容量、编码器、可靠性融合、时序汇聚、GPS 鲁棒性和辅助目标尚未在同一固定协议下做可归因的设计筛选。

本 change 将先建立可追溯的 H4 训练 profile，再以分阶段、单因素且不消费正式测试集的方式筛选有限的 T2 设计候选，为后续独立多 seed 证据冻结一个主线配置。

## What Changes

- 新增显式 `umask_h4_v1` 训练 profile：固定 AdamW、`lr=5e-4`、`weight_decay=3e-4`、40-epoch cosine warm restart 日程，并要求新的 T2/S1 主线 launcher、resolved config、checkpoint 和汇总共同记录其指纹。
- 修复 MMW all-weather config builder 对 optimizer、weight decay 和 scheduler 的静默覆盖；保持 T2/S1 的 matched U-Mask profile 可显式选择，且不改变 AMBER-Full/RMBP-MM 的基线 recipe。
- 新增仅服务开发筛选的 T2 设计矩阵：在 H4 基线上分阶段评估模型宽度、保留 encoder 选择、融合、时序汇聚、GPS 输入扰动/编码，以及 BPA、CMA 和一个预注册辅助目标候选。
- 将可选 fusion 与 temporal pooling 实现为受限、显式配置分支，并保留默认 T2 行为；所有新候选通过单 step smoke、固定 inner validation、40 epoch 和完整 provenance 才可进入训练。
- 增加 profile/候选 fingerprint、训练完成性和 identity/mask 一致性校验；筛选输出永久标记为 development-only，不更新论文 claim 或替换进行中的 BPA/CMA 正式消融。

## Capabilities

### New Capabilities

- `mmw-t2-h4-design-screening`: 定义 H4 主线 profile 与 MMW T2 分阶段设计筛选的配置、并行、选择和证据边界。

### Modified Capabilities

- `u-mask-beam-jepa`: 将 T2/S1 模型契约扩展为受控的 fusion、时序汇聚、GPS encoder 和辅助目标实验分支，同时保持默认 T2 行为。
- `training-evaluation-runtime`: 要求 MMW 训练和评估记录并校验训练 profile、结构候选和 recipe 指纹，禁止不同 profile 的结果混合。
- `canonical-config-resolution`: 要求 tracked MMW recipe 与 tracked launcher 的显式训练 profile 能在干净 clone 中解析，且 profile 不隐式改变无关 baseline 或 legacy H0 protocol。

## Impact

- 影响 `configs/mmw/`、`scripts/launch_mmw_all_weather_matrix.py`、新的 T2 design-screening launcher 和相关 MMW runtime tests。
- 影响 `src/kd_sensing/models/u_mask_beam_jepa.py`、GPS encoder/loss 配置及其模型/损失测试。
- 新生成 YAML、split、日志、checkpoint、指标和图表仅写入 ignored `outputs/`；不新增第三方依赖，也不改变现有 BPA/CMA formal ablation 的 H0 行。
