## Context

当前包已收敛为 MMW 的四模态 T2/baseline 运行面，batch 和模型均固定消费 `image`、`radar_ra`、`radar_da`、`gps`、`lidar` 与 `target_beam`。历史 DeepSense6G 实现同时承担 CSI、毫米波原始输入、软标签、缓存、蒸馏和多条数据协议，删除它不能直接恢复为当前主线。

本变更仅恢复与现有四模态契约重合的 DeepSense6G 数据路径。Scene31–34 的 CSV 已包含按时间编号的 camera、radar、GPS、基站 GPS、LiDAR 与 future beam 路径；future beam 文件是 64 维功率向量，可直接转换为硬标签。

## Goals / Non-Goals

**Goals:**

- 使 `data.dataset.type: deepsense6g` 与现有 `mmw` 通过同一 data factory、batch、模型、训练和评估链路运行。
- 仅支持 Scene31–34、显式 train/test CSV、四模态输入和 64 类未来波束 `argmax` 标签。
- 保持 MMW 的四个 canonical baseline recipe、专属 evidence helper 和 active MMW OpenSpec 变更不变。
- 为 DeepSense6G 提供独立 T2 recipe、严格校验和小型合成数据测试。

**Non-Goals:**

- 不恢复 CSI、毫米波原始输入、输入 beam、软标签、蒸馏、缓存、descriptor、数据预处理 CLI、历史 split 名称或场景别名。
- 不为 DeepSense6G 伪造 S1、AMBER-Full 或 RMBP-MM 比较矩阵。
- 不迁移或重新生成本地数据、输出、checkpoint 或历史实验结论。

## Decisions

### 1. 以独立的严格 dataset 类接入共享工厂

新增 `DeepSense6GDataset`，由现有 registry 和 data factory 根据 `data.dataset.type` 显式构建。该类直接读取当前 CSV 和资源文件，并输出已经存在的 batch 字段；模型、batch 和 temporal mask 不增加数据集分支。

选择独立类而不把 CSV 解析塞入 MMW 样本模块，因为两者标签和字段语义不同。也不恢复历史万能数据类，因为它会重新引入已退役的输入协议。

### 2. 固定 Scene31–34 与未来波束硬标签

配置中的 `scene` 必须是整数 31、32、33 或 34。默认根目录为 `dataset/DeepSense6G/scenario{scene}`；train/test 分别使用显式配置名或该目录的标准 CSV 名称，validation 仅在显式提供 `val_csv_name` 时构建。每个时间步的 `future_beamN` 文件必须是有限的 64 维向量，标签为其 `argmax`。

这避免历史 route、cross-scene、目标切换和软标签分支。直接读取 future beam 比另建预处理产物更少代码，也不会增加 CLI 或本地产物依赖。

### 3. 四模态和 GPS 语义完全复用当前契约

DeepSense6G 使用既有 image、radar、GPS 与 LiDAR transform。雷达仅使用 RA/DA map，GPS 使用当前相对极坐标三维特征；训练 split 拟合的 GPS scaler 注入 validation/test split。dataset 必须按当前顺序输出时间序列字段、硬标签、稳定 sample ID 和 dataset/scene metadata。

这样 T2/baseline batch prep 和模型无需泛化。拒绝缺失列、错误 shape、非有限标签和不支持场景，优先在加载边界失败。

### 4. 配置表面保持最小

新增 `configs/deepsense6g/_base.yaml` 与 `t2.yaml`。四个 MMW recipe 继续保留，并且 DeepSense6G 不配置未验证的 baseline 复现实验。配置校验只接受 `mmw` 或 `deepsense6g`，且两者都必须声明当前四模态集合；DeepSense6G 额外校验场景和 64 类雷达/标签契约。

### 5. 不提供兼容层

历史 DeepSense6G 文件名、场景别名、cache、输入 beam、CSI/mmWave 及旧入口一律不映射。用户如需历史研究资料，只能从 retired-route 文档与 git 历史追溯，而不能通过当前 runtime 重新激活。

## Risks / Trade-offs

- [本地 CSV 的资源路径缺失或格式损坏] → dataset 在构建和取样时给出带 scene/split/列名的明确错误，测试覆盖最小有效与无效样本。
- [DeepSense6G 与 MMW 标签语义混淆] → 使用独立 dataset 类和 `dataset_family` metadata，不复用 MMW label parser。
- [双数据集配置继续膨胀] → 只添加 T2 recipe，不添加旧 CLI、缓存或 baseline 矩阵；架构测试限制当前 surface。
- [真实数据加载比合成 smoke 慢] → 不在版本控制中加入数据；测试用临时小样本验证契约，完整训练由本地数据执行。

## Migration Plan

1. 添加规格、独立 dataset、registry/factory/validation 接入和 DeepSense6G T2 配置。
2. 用临时样本验证 CSV 到四模态 batch、GPS scaler 和未来 beam 标签，再运行现有全量校验。
3. 更新 README、导航和 retired-route 说明，将 DeepSense6G 从“退役数据族”改为“重写后的受限主线”。

无需数据迁移或兼容开关。回滚只需移除本变更新增的 dataset/config 与相应双数据集条件，不影响 MMW 数据与模型路径。

## Open Questions

- 无。Scene31–34、四模态输入、64 类未来波束标签和仅提供 T2 recipe 已作为本变更边界确定。
