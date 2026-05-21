对，你说得对。**s008 不能严谨地叫 future beam prediction / 波束预测**，更准确应该叫：

```text
sensing-aided beam selection
当前多模态快照 → 当前最优 beam pair
```

Raymobtime/CAVIAR 论文明确说 s008 是 **snapshot dataset**，每个 episode 只有 **1 个 scene**；而做 beam 的论文也把 s008/s009 这类数据定义为 **beam selection task**：从 32×8=256 个 Tx/Rx beam pair 中选择使接收 SNR 最强的最优 beam pair。([arXiv][1])

所以我之前方案里的 “future beam prediction / beam prediction” 表述要改成 **current beam selection / 当前波束选择**。下面是修改后的 Codex 方案。

---

## Codex 任务：Raymobtime s008 多模态多任务 beam selection 框架

### 0. 核心修正

Raymobtime s008 是 snapshot 数据，不是时序轨迹数据。

```text
s008:
episode_i
  └── scene_0
        ├── vehicle_0
        ├── vehicle_1
        └── ...
```

因此第一版任务不要做：

```text
历史窗口 → 未来 beam
future beam prediction
beam tracking
LOS/NLOS transition prediction
```

第一版只做：

```text
当前多模态快照 → 当前最优 beam
当前多模态快照 → 当前 LOS/NLOS
当前多模态快照 → 当前 link quality
```

任务名称统一改为：

```text
T1: current beam selection
T2: LOS/NLOS classification
T3: link quality regression
```

---

# 一、数据现状

已有数据：

```text
Raymobtime_s008/
├── baseline_data/
│   ├── beam_output/
│   ├── coord_input/
│   ├── lidar_input/
│   └── image_v2_input/
└── raw_data/
    ├── CoordVehiclesRxPerScene_s008.csv
    └── ray_tracing_data_s008_carrier60GHz.zip
```

s008 的 beam 任务应该按 **beam selection** 实现。也就是：输入当前时刻的 coord/image/lidar/ray feature，输出当前样本的最优 beam pair 或 beam class。

---

# 二、总体目标

构建一个 Raymobtime s008 的 **多模态多任务 beam selection** 框架，用于研究：

```text
snapshot setting 下，不同任务是否偏好不同模态，
以及多任务训练是否会导致 task-dependent modality imbalance。
```

输入模态：

```text
coord: baseline_data/coord_input
image: baseline_data/image_v2_input
lidar: baseline_data/lidar_input
ray: raw_data/ray_tracing_data_s008_carrier60GHz.zip 提取的 path-level feature
```

输出任务：

```text
T1: beam selection
    标签来自 baseline_data/beam_output
    指标 top-1 / top-3 / top-5 accuracy

T2: LOS/NLOS classification
    标签来自 raw_data/CoordVehiclesRxPerScene_s008.csv 的 LOS 列
    指标 accuracy / F1 / AUC

T3: link quality regression
    标签来自 ray-tracing received power
    指标 MAE / RMSE / R2
```

注意：代码里所有命名都避免使用 `future_beam`、`beam_prediction_horizon`、`seq_len`、`num_pred` 这类时序预测概念。

---

# 三、目录结构

```text
project_root/
├── configs/
│   └── raymobtime_s008_multitask_selection.yaml
├── scripts/
│   ├── audit_s008_files.py
│   ├── build_s008_index.py
│   ├── extract_s008_ray_features.py
│   ├── build_s008_cache.py
│   ├── train_s008_multitask_selection.py
│   └── analyze_s008_modality_imbalance.py
├── src/
│   ├── data/
│   │   ├── raymobtime_s008_dataset.py
│   │   └── raymobtime_s008_utils.py
│   ├── models/
│   │   ├── encoders.py
│   │   ├── fusion_multitask.py
│   │   └── losses.py
│   ├── engine/
│   │   ├── trainer_multitask.py
│   │   └── metrics.py
│   └── analysis/
│       ├── gate_analysis.py
│       ├── gradient_analysis.py
│       └── modality_ablation.py
└── outputs/
    └── raymobtime_s008/
```

---

# 四、数据审计

实现：

```bash
python scripts/audit_s008_files.py --data_root /path/to/Raymobtime_s008
```

检查：

```text
baseline_data/beam_output
baseline_data/coord_input
baseline_data/lidar_input
baseline_data/image_v2_input
raw_data/CoordVehiclesRxPerScene_s008.csv
raw_data/ray_tracing_data_s008_carrier60GHz.zip
```

对每个 `.npz` 输出：

```text
file path
npz keys
shape
dtype
min / max / mean
```

对 CSV 输出：

```text
总行数
Val=V 数量
Val=I 数量
LOS=1 数量
LOS=0 数量
EpisodeID 唯一数量
SceneID 唯一数量
VehicleArrayID 唯一数量
x/y/z 范围
```

输出文件：

```text
outputs/raymobtime_s008/audit/audit_summary.json
outputs/raymobtime_s008/audit/npz_shapes.csv
outputs/raymobtime_s008/audit/csv_summary.json
```

---

# 五、构建 snapshot 样本索引

实现：

```bash
python scripts/build_s008_index.py \
  --data_root /path/to/Raymobtime_s008 \
  --out_dir outputs/raymobtime_s008/cache
```

从：

```text
raw_data/CoordVehiclesRxPerScene_s008.csv
```

读取并标准化列：

```text
Val
EpisodeID
SceneID
VehicleArrayID
VehicleName
x
y
z
LOS
```

只保留：

```text
Val == "V"
```

构建：

```text
sample_id = f"e{EpisodeID}_s{SceneID}_v{VehicleArrayID}"
```

注意：**这里不是时间序列索引**，而是 snapshot index。

每行代表：

```text
一个 episode 的唯一 scene 中的一个 receiver 样本
```

输出：

```text
outputs/raymobtime_s008/cache/index_train.csv
outputs/raymobtime_s008/cache/index_val.csv
outputs/raymobtime_s008/cache/index_test.csv
outputs/raymobtime_s008/cache/index_all_valid.csv
```

每个 index 至少包含：

```text
sample_id
split
valid_index
EpisodeID
SceneID
VehicleArrayID
VehicleName
x
y
z
LOS
coord_row_in_split
```

---

# 六、beam selection 标签处理

在：

```text
scripts/build_s008_cache.py
```

中读取：

```text
baseline_data/beam_output/
```

自动识别 train/val/test 文件。

支持三种 beam 输出格式：

```text
[N]         -> 已经是 beam class
[N, 2]      -> [tx_beam, rx_beam]
[N, Tx, Rx] -> beam score / gain matrix
```

统一输出：

```text
beam_label: [N]
beam_tx: [N]
beam_rx: [N]
num_beam_classes
num_tx_beams
num_rx_beams
```

如果是 `[N, 2]`：

```python
beam_class = tx_beam * num_rx_beams + rx_beam
```

如果是 `[N, Tx, Rx]`：

```python
flat_idx = argmax(scores.reshape(N, -1), axis=1)
tx = flat_idx // Rx
rx = flat_idx % Rx
beam_class = flat_idx
```

保存：

```text
outputs/raymobtime_s008/cache/labels_train.npz
outputs/raymobtime_s008/cache/labels_val.npz
outputs/raymobtime_s008/cache/labels_test.npz
```

其中包含：

```text
beam_label
beam_tx
beam_rx
los_label
sample_id
valid_index
```

---

# 七、ray-tracing 特征提取

实现：

```bash
python scripts/extract_s008_ray_features.py \
  --data_root /path/to/Raymobtime_s008 \
  --index_dir outputs/raymobtime_s008/cache \
  --out_dir outputs/raymobtime_s008/cache
```

从：

```text
raw_data/ray_tracing_data_s008_carrier60GHz.zip
```

提取每个 snapshot sample 的 path-level 特征。

每个样本由：

```text
EpisodeID
SceneID
VehicleArrayID
```

定位。

提取特征：

```text
max_power_dbm
mean_power_dbm_valid
sum_power_linear
num_valid_rays
min_toa
mean_toa
strongest_ray_toa
strongest_ray_elev_aod
strongest_ray_az_aod
strongest_ray_elev_aoa
strongest_ray_az_aoa
strongest_ray_phase
power_spread_db
toa_spread
```

保存两套特征：

```text
ray_features_no_los:
    不包含 LOS flag，用作模型输入

ray_features_with_los:
    包含 LOS flag，只用于 audit / 校验，不作为 LOS 分类输入
```

link quality label：

```text
link_power_max_dbm
link_power_sum_dbm
```

保存：

```text
outputs/raymobtime_s008/cache/ray_features_train.npz
outputs/raymobtime_s008/cache/ray_features_val.npz
outputs/raymobtime_s008/cache/ray_features_test.npz
```

---

# 八、Dataset 类

实现：

```python
RaymobtimeS008SnapshotDataset
```

不要叫 sequence dataset。

初始化参数：

```python
data_root
cache_dir
split: train / val / test
modalities: ["coord", "image", "lidar", "ray"]
tasks: ["beam_selection", "los", "link"]
normalize: bool = True
```

`__getitem__` 返回：

```python
{
    "inputs": {
        "coord": Tensor,
        "image": Tensor,
        "lidar": Tensor,
        "ray": Tensor,
    },
    "targets": {
        "beam_selection": LongTensor scalar,
        "los": FloatTensor scalar,
        "link": FloatTensor scalar,
    },
    "meta": {
        "sample_id": str,
        "EpisodeID": int,
        "SceneID": int,
        "VehicleArrayID": int,
        "valid_index": int,
        "split": str,
    }
}
```

不要返回：

```text
history
future
t
t+1
seq_len
horizon
```

---

# 九、模型设计

实现两个 snapshot 模型。

## 1. SimpleConcatMultiTaskSelection

```text
coord encoder  -> [B, D]
image encoder  -> [B, D]
lidar encoder  -> [B, D]
ray encoder    -> [B, D]

concat -> [B, K*D]
projection -> [B, D]

beam_selection_head -> [B, num_beam_classes]
los_head            -> [B, 1]
link_head           -> [B, 1]
```

## 2. TaskAwareGatedMultiTaskSelection

对每个任务单独 gate：

```python
features = stack([feat_coord, feat_image, feat_lidar, feat_ray], dim=1)
# [B, K, D]

for task in ["beam_selection", "los", "link"]:
    gate_logits = gate_net_task(features)  # [B, K]
    gate = softmax(gate_logits, dim=1)
    fused_task = sum(gate[:, k] * features[:, k])
    pred_task = head_task(fused_task)
```

保存 gates：

```python
outputs["gates"] = {
    "beam_selection": [B, K],
    "los": [B, K],
    "link": [B, K],
}
```

---

# 十、Loss

```python
L = w_beam * CE(beam_logits, beam_label)
  + w_los  * BCEWithLogits(los_logit, los_label)
  + w_link * SmoothL1(link_pred, link_label)
```

默认：

```yaml
loss_weights:
  beam_selection: 1.0
  los: 0.5
  link: 0.2
```

评估：

```text
beam_top1
beam_top3
beam_top5
los_acc
los_f1
los_auc
link_mae
link_rmse
link_r2
```

---

# 十一、实验矩阵

## 1. 单模态多任务

```text
coord only
image only
lidar only
ray only
```

## 2. 多模态融合

```text
coord + image
coord + lidar
coord + ray
image + lidar
coord + image + lidar
coord + image + lidar + ray
```

每组跑：

```text
simple_concat
task_aware_gated
```

## 3. 感知-only 与 ray-enhanced 分开

必须分开汇报：

```text
sensing-only:
    coord + image + lidar

sensing + ray:
    coord + image + lidar + ray
```

原因：ray-tracing 特征本身离通信标签源很近，可能过强。不要让它掩盖 image/lidar/coord 的价值。

## 4. test-time modality drop

对 full model 做：

```text
drop coord
drop image
drop lidar
drop ray
```

输出：

```text
Δbeam_top1
Δlos_f1
Δlink_mae
```

---

# 十二、模态失衡分析

实现：

```bash
python scripts/analyze_s008_modality_imbalance.py \
  --exp_dir outputs/raymobtime_s008/experiments/debug_full
```

输出：

```text
single_modality_task_performance.csv
gate_mean_by_task.csv
gate_mean_by_task_and_los_bucket.csv
drop_modality_delta.csv
beam_metrics_by_los_bucket.csv
grad_norms_by_task_modality.csv
```

重点分析：

```text
1. beam selection 是否主要依赖 coord/ray
2. LOS/NLOS 是否更依赖 image/lidar/ray
3. link quality 是否主要依赖 ray
4. simple concat 是否被某个强模态主导
5. task-aware gate 是否能让不同任务选择不同模态
6. NLOS bucket 下 image/lidar 是否比全局指标更有价值
```

---

# 十三、必须删除/避免的内容

请删除旧方案中的这些概念：

```text
future beam prediction
beam tracking
history window
future horizon
seq_len
num_pred
GRU / temporal transformer
scene t -> scene t+1
LOS transition prediction
beam switch prediction
```

s008 当前只做：

```text
snapshot-based current beam selection
```

---

# 十四、验收命令

```bash
python scripts/audit_s008_files.py --data_root /path/to/Raymobtime_s008

python scripts/build_s008_index.py \
  --data_root /path/to/Raymobtime_s008 \
  --out_dir outputs/raymobtime_s008/cache

python scripts/extract_s008_ray_features.py \
  --data_root /path/to/Raymobtime_s008 \
  --index_dir outputs/raymobtime_s008/cache \
  --out_dir outputs/raymobtime_s008/cache

python scripts/build_s008_cache.py \
  --data_root /path/to/Raymobtime_s008 \
  --cache_dir outputs/raymobtime_s008/cache

python scripts/train_s008_multitask_selection.py \
  --config configs/raymobtime_s008_multitask_selection.yaml \
  --experiment_name debug_full \
  --max_train_batches 5 \
  --max_val_batches 2

python scripts/analyze_s008_modality_imbalance.py \
  --exp_dir outputs/raymobtime_s008/experiments/debug_full
```

---

# 十五、论文表述也要改

不要写：

```text
beam prediction on Raymobtime s008
future beam prediction
sequential beam prediction
```

应该写：

```text
sensing-aided mmWave beam selection on Raymobtime s008
snapshot-based beam selection
current optimal beam-pair classification
```

更准确的问题定义是：

> 给定当前场景快照中的 coord / image / LiDAR / ray-tracing features，选择当前最优 Tx/Rx beam pair，并同时预测 LOS/NLOS 和 link quality。

这样才和 s008 数据结构一致。

[1]: https://arxiv.org/pdf/2106.05377?utm_source=chatgpt.com "arXiv:2106.05377v1 [eess.SP] 9 Jun 2021"
