## Context

DeepSense6G 是当前多条 workflow 的共享数据底座，包括 Image+GPS JEPA query-pool、BeamBench AE+GPS Direct、BEV-Fusion 2604、GPS/LiDAR BGAM 和 difficulty pipeline。直接大拆 dataset class 风险很高，因为 `__getitem__` 和真实路径读取与多个 current workflow 耦合。

## Goals / Non-Goals

**Goals:**

- 先抽出不读取大资源、不改样本输出的 contract helper。
- 让新增/修改 GPS feature mode、beam target source、column guard 和 cache path 规则时无需编辑超长 dataset class 主体。
- 降低 `DeepSense6GDataset.__init__` 和 class 行数预算。
- 为 helper 提供 synthetic tests，覆盖错误信息和兼容行为。

**Non-Goals:**

- 不重写 `__getitem__`。
- 不替换为完整 RuntimeDataset/adapter 架构。
- 不改变现有 CSV columns、flat sample keys、target semantics 或 cache file layout。
- 不读取真实 `dataset/` 作为验证依据。

## Decisions

### Decision 1: 抽 pure/near-pure helper，而不是先抽资源读取

首批 helper 只处理输入配置、路径、列名和 metadata parsing。image/GPS/LiDAR/CSI 实际读取仍留在 dataset class 或现有 transform helper 中。这样改动更容易用 synthetic tests 验证。

### Decision 2: helper 模块按契约职责命名

建议模块：

- `deepsense6g_contract.py`: beam target source、enabled modalities、common validation。
- `deepsense6g_gps_contract.py`: GPS feature mode、angle offset、scene calibration、GPS BEV XY source。
- `deepsense6g_columns.py`: required column sets 和 error message builder。
- `deepsense6g_cache_paths.py`: image/LiDAR/cache path resolution 的纯路径逻辑。

模块名可在实现中调整，但职责必须清晰，不能引入 facade 聚合层。

### Decision 3: 数据类继续 orchestration

`DeepSense6GDataset` 应变薄，但不要求一次性降到很小。它可以继续组合 dataframe、helper、reader 和 cache policy，只要新 contract 规则不再继续堆进类体。

## Risks / Trade-offs

- [Risk] 抽 helper 时改变错误消息或默认值。  
  → Mitigation: 为当前默认 `paper_distance_angle`、`beam_target_source=current/future`、GPS BEV XY source 和 missing columns 添加 characterization tests。

- [Risk] helper 变成新的聚合 facade。  
  → Mitigation: helper 按职责拆分，内部调用不通过大 facade 回流。

- [Risk] dataset 行数下降不明显。  
  → Mitigation: 首批目标是减少高风险契约逻辑集中度；行数预算可以小幅下降，不追求一次性大瘦身。

- [Risk] 多 workflow 共享语义被误改。  
  → Mitigation: 跑配置 characterization、BeamBench/JEP A focused tests 和 architecture boundary。
