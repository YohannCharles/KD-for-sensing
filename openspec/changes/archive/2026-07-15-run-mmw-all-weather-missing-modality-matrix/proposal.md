## Why

MMW 已被指定为新的主数据集，但当前 15 组 Town03 H5/P1 产物尚不能直接支撑公平主实验：所有 split 都因单步 beam 类别序列重复被标为 strict-ineligible，rainy/foggy readiness 仍为 pending，四传感器主配置所需 radar maps 也未物化，同时现有训练入口没有显式的跨 condition/scenario pooled-domain 契约。需要先收口数据与评估协议，再比较 S1、T2 和两个缺失模态 baseline，避免把数据泄漏、缺失输入或 domain 数量偏置误当作模型收益。

## What Changes

- 将 P1 `future_label_sequence_reuse` 保留为分布诊断，但只有 frame/window/group/guard-band 等结构性重叠才使 split strict-ineligible；禁止把正常的 beam 类别重复直接判作数据泄漏。
- 通过公开 MMW preparation/radar utility 为 sunny、rainy、foggy 各 5 个 Town03 场景生成可审计的严格 split、radar maps、带 radar 列的 CSV、metadata、sanity 和 availability。
- 为 MMW 增加显式 all-weather domain 列表和 pooled `ConcatDataset` 训练契约，按 scene-weather domain 等权采样，并在 run metadata 中记录每个 domain 的 condition、scenario、split 和样本数。
- 增加 S1、T2、AMBER-Full、RMBP-MM 的 GPU0-3 seed1 launcher、固定 whole-modality/temporal-missing evaluation 和 weather/scene macro summary；四方法共享输入模态、split、缺失采样、训练预算和 checkpoint 规则。
- 增加 T2、AMBER-Full、RMBP-MM 的配对融合表征诊断：复用固定 checkpoint、验证样本和 `modality_frame` mask cache，在各方法 clean 表征上独立拟合 PCA，并用原始特征空间的漂移与邻近 beam 指标约束可视化解释；进一步用原始 64 维余弦 Gram matrix、固定 kNN-Isomap、循环距离衰减和 signed feature-shift 图展示 PCA 丢失的高维循环邻接。
- 第一轮不增加天气标签输入或天气专用模型模块；T2 负责 temporal superset consistency，现有 reliability fusion 负责天气引起的模态质量差异。只有首轮 router/calibration 诊断确认失败模式后才另提模型 change。
- DeepSense seeds2/3 和其 PPT 结果回填按用户决定停止；GPU4-7 的 LG/CLS seed1 候选继续运行，但不作为 MMW 主结论的前置条件。

## Capabilities

### New Capabilities

- `mmw-all-weather-missing-modality-matrix`: 定义 5 场景 × 3 天气 pooled 训练、domain-balanced sampling、四方法公平矩阵、固定缺失评估、weather/scene macro 与晋级门禁。

### Modified Capabilities

- `mmw-town10-dataset-preparation`: 修正 strict split eligibility 对单步类别序列重复的解释，并要求 H5/P1 派生产物可被 readiness writer 审计。
- `dataset-runtime-contracts`: 增加显式 MMW domain 列表构建 pooled train/validation/test dataset 的配置与 provenance 契约。
- `mmw-sensor-assisted-beam-prediction`: 将 all-weather 主矩阵约束为 image、GPS、LiDAR、radar 四传感器输入，并增加按天气/场景与 missing pattern 的 eligibility 和汇总要求。

## Impact

- 数据准备与诊断：`src/kd_sensing/data/mmw/`、`src/kd_sensing/preprocessing/mmw_radar.py`、condition/scenario 本地 metadata。
- 数据运行时：`src/kd_sensing/engine/data_factory.py` 及 focused domain-balanced sampler/provenance helper。
- 本地实验入口：新增或窄扩展 `scripts/` 下 MMW all-weather launcher、evaluator 和 summary；生成配置、日志、cache、checkpoint 与报告全部留在 ignored `outputs/` 或 `dataset/MMW/`。
- 本地表征诊断：新增 local/manual 脚本，分片特征、PCA/高维拓扑图片、CSV/JSON summary 全部留在 ignored `outputs/analysis/`，不新增公共 CLI、第三方依赖或训练时行为。
- 配置：复用现有 U-Mask、AMBER-Full、RMBP-MM 模型与编码器，不新增依赖、不新增天气网络模块、不恢复任何 retired Hist/HiST 路由。
