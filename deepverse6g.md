下面这份可以直接发给 Codex。核心目标是：**新增一套 DeepVerse6G-DT31 多模态多任务数据管线**，用于研究 **task-dependent modality imbalance**，不要再沿用 DeepSense 的强标签源捷径。

DeepVerse6G 的 Python 仓库说明它是用于生成多模态 sensing + communication 数据的 generator，可通过 `pip install deepverse` 或源码安装；官方也说明 DT31 是 DeepSense Scenario 31 的 digital twin。([GitHub][1]) DeepVerse 文档里，数据生成需要设置 `dataset_folder` 和 `scenario`，然后通过 `Dataset(param_manager)` 触发生成；通信和 radar 是实时生成，camera/LiDAR 等传感器数据则从 scenario 文件加载。([DeepVerse 6G][2])

---

# 可以直接发给 Codex 的开发方案

```text
请在当前项目中新增 DeepVerse6G-DT31 多模态多任务数据管线，不要破坏现有 DeepSense6G 代码。目标是支持 camera + LiDAR + radar + weak wireless + noisy position 输入，同时生成 future beam / future blockage / future trajectory 三个任务标签，用于研究多模态多任务模态失衡。

背景：
DeepVerse6G 不是直接提供可训练的统一 npz/h5 数据，而是需要用 python deepverse generator + config 参数从 scenario 文件生成 dataset object。请实现一个可复现的数据生成、标签派生、缓存、训练数据加载和 baseline 评估流程。

核心原则：
1. channel / LoS_status / clean mobility location / bounding_box 只用于生成标签或 oracle 分析，默认不要作为主输入直接喂模型。
2. 主输入使用 camera、LiDAR、radar、past weak wireless features、noisy position。
3. 任务标签：
   - future beam label：由 ray-tracing channel 经过预定义 beam codebook 计算 beam gain，然后 argmax 得到。
   - future blockage label：由 comm_sample.LoS_status 派生，LoS 为 0，NLoS/no-path 为 1。
   - future trajectory label：由 mobility ground-truth location 派生，预测未来 K 个 [x, y] 坐标。
4. 训练划分尽量按 UE/object id 或 trajectory id 划分，不要纯随机 frame 划分，避免相邻时间泄漏。

请按下面结构实现。
```

---

# 1. 新增目录结构

```text
src/kd_sensing/
  data/
    deepverse/
      __init__.py
      config.py
      generator.py
      label_builder.py
      codebook.py
      preprocess_camera.py
      preprocess_lidar.py
      preprocess_radar.py
      dataset.py
      collate.py
      split.py
      sanity_check.py

  models/
    deepverse/
      __init__.py
      encoders.py
      multitask_model.py
      fusion.py
      losses.py
      metrics.py

scripts/
  deepverse/
    generate_dt31_cache.py
    build_dt31_manifest.py
    train_deepverse_multitask.py
    eval_deepverse_multitask.py
    run_deepverse_sanity.py

configs/
  deepverse/
    dt31_generation.yaml
    dt31_multitask.yaml
    dt31_experiments.yaml

outputs/
  deepverse_dt31/
    cache/
    manifests/
    checkpoints/
    reports/
```

---

# 2. DeepVerse DT31 数据生成方案

## 2.1 目标

实现：

```bash
python scripts/deepverse/generate_dt31_cache.py \
  --scenario-root /path/to/DeepVerse/scenarios \
  --scenario DT31 \
  --config-m /path/to/DeepVerse/scenarios/DT31/param/config.m \
  --output-root dataset/deepverse_dt31/cache \
  --seq-len 8 \
  --pred-horizon 3 \
  --beam-codebook-size 64 \
  --modalities camera,lidar,radar,past_wireless,noisy_position \
  --position-noise-std 1.0
```

要求生成：

```text
dataset/deepverse_dt31/cache/
  metadata.json
  samples.parquet 或 samples.csv
  labels.npz
  radar_features.npz
  weak_wireless.npz
  noisy_position.npz
  camera_index.json
  lidar_index.json
  split.json
```

其中 `samples.csv/parquet` 每一行是一个训练 sample：

```text
sample_id
scenario
ue_id
bs_id
t_anchor
history_indices
future_indices

camera_paths
lidar_paths
radar_paths

past_beam_indices
past_topk_beam_powers
past_max_power
past_beam_entropy

noisy_position_history
clean_position_history

label_beam_future
label_blockage_future
label_trajectory_future

los_status_future
beam_gain_future
valid_mask
split
```

---

## 2.2 DeepVerse generator 代码要求

新增 `src/kd_sensing/data/deepverse/generator.py`：

```python
class DeepVerseDT31Generator:
    def __init__(
        self,
        scenario_root: str,
        scenario: str = "DT31",
        config_m: str | None = None,
        scenes: list[int] | None = None,
        enable_camera: bool = True,
        enable_lidar: bool = True,
        enable_radar: bool = True,
        enable_comm: bool = True,
        enable_position: bool = True,
    ):
        ...

    def load_dataset(self):
        """
        使用 deepverse.ParameterManager 和 deepverse.Dataset 生成 dataset object。
        如果 config_m 不存在，尝试使用 scenario_root/scenario/param/config.m。
        生成前修改 param_manager.params:
          dataset_folder = scenario_root
          scenario = scenario
          camera = enable_camera
          lidar = enable_lidar
          radar['enable'] = enable_radar
          comm['enable'] = enable_comm
          position = enable_position
          scenes = scenes if provided
        """
        ...

    def get_dataset(self):
        ...
```

实现时注意：

1. `deepverse` 可能来自 `pip install deepverse`，也可能来自本地源码安装；导入失败时给出明确报错。
2. DeepVerse 官方示例中用法是：

   ```python
   from deepverse import ParameterManager
   from deepverse import Dataset
   param_manager = ParameterManager(config_path)
   param_manager.params['dataset_folder'] = r'D:\DeepVerse\scenarios'
   param_manager.params['scenario'] = 'Town01-Carla'
   dataset = Dataset(param_manager)
   ```

   请按这个模式实现。
3. 如果 DT31 参数字段和示例字段不完全一致，不要崩溃；先打印 `param_manager.params.keys()`，再按存在字段修改。
4. 保存最终参数到：

   ```text
   dataset/deepverse_dt31/cache/used_generation_params.json
   ```

DeepVerse 文档里给的参数示例包含 `camera=True`、`lidar=True`、`position=True`、`comm.enable=True`、`comm.generate_OFDM_channels=1`、`comm.bs_antenna.shape=[32,1]`、`radar.enable=False/True`、`scenes=[...]` 等字段；请用“存在则修改，不存在则跳过”的稳健写法。([DeepVerse 6G][2])

---

# 3. 标签生成方案

新增 `src/kd_sensing/data/deepverse/label_builder.py`。

## 3.1 Beam label

输入：

```python
comm_sample = dataset.get_sample("comm-ue", index=t, bs_idx=bs_idx, ue_idx=ue_idx)
H = comm_sample.coeffs
```

DeepVerse 文档示例显示 `comm_sample.coeffs.shape` 类似 `(32, 1, 2)`，并且 `comm_sample.LoS_status` 可以访问。([DeepVerse 6G][2])

实现：

```python
beam_gain_vector = compute_beam_gain(H, codebook)
beam_label = int(np.argmax(beam_gain_vector))
```

新增 `src/kd_sensing/data/deepverse/codebook.py`：

```python
def make_ula_dft_codebook(num_ant: int, num_beams: int = 64) -> np.ndarray:
    """
    返回 shape [num_ant, num_beams] 的复数 codebook。
    默认使用 DFT/steering codebook。
    每个 beam vector L2 normalize。
    """

def compute_beam_gain(H: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """
    H 可能是 [N_bs_ant, N_ue_ant, N_sc] 或 [N_rx, N_tx, N_sc]。
    针对 DeepVerse 示例，优先假设 H[:, 0, k] 是 BS antenna 维度。
    对每个 beam f_b:
        gain_b = mean_k |f_b^H h_k|^2
    返回 [num_beams] float32。
    """
```

注意加 shape 检查：

```python
if H.shape[0] == codebook.shape[0]:
    h = H[:, 0, k]
elif H.shape[1] == codebook.shape[0]:
    h = H[0, :, k]
else:
    raise ValueError(...)
```

同时保存：

```text
beam_label_t
beam_gain_vector_t
max_beam_power_t
topk_beam_indices_t
topk_beam_powers_t
beam_entropy_t
```

这些后面用于 past weak wireless input。

---

## 3.2 Blockage label

DeepVerse 的 `LoS_status` 推荐这样映射：

```python
def los_to_blockage(los_status: int) -> int:
    # DeepVerse docs/comment convention:
    # 1: LoS
    # 0: NLoS
    # -1: no path
    return 0 if los_status == 1 else 1
```

标签：

```text
blockage_label = 0  # LoS / unblocked
blockage_label = 1  # NLoS or no-path / blocked
```

同时保留原始：

```text
los_status
```

用于分析。

---

## 3.3 Trajectory label

从 mobility 读取：

```python
mobility = dataset.get_sample("mobility", ue_idx=ue_idx)
mobility_info = mobility.get_all_samples()
locations = np.asarray(mobility_info["location"])  # [T, 3]
times = np.asarray(mobility_info["time"])
```

DeepVerse mobility 文档说明 `get_all_samples()` 返回 `time/location/angle/speed/acceleration/bounding_box/tx_height/slope`，其中 `location` 是 3D Cartesian 坐标，`bounding_box` 是 box corners。([DeepVerse 6G][3])

生成：

```python
position_2d_t = locations[t, :2]
trajectory_future = locations[t+1:t+K+1, :2]
```

注意：

1. 不能直接假设 mobility array index 等于全局 time index。
2. 要建立 `time -> local_idx` 映射。
3. 一个 sample 只有当 `history_indices` 和 `future_indices` 都存在时才有效。
4. 如果某个 UE 进入/离开场景导致窗口不完整，跳过该 sample。

---

# 4. 输入模态生成方案

## 4.1 Camera 输入

新增 `preprocess_camera.py`：

```python
def get_camera_paths(dataset, t: int, camera_ids: list[int]) -> list[str]:
    """
    调用 dataset.get_sample('cam', index=t, device_index=camera_id)
    返回图片路径。
    如果文件不存在，返回空并记录 warning。
    """

def preprocess_camera_image(path, image_size=(224, 224)):
    """
    训练时可在 Dataset 中实时读取；
    cache 阶段只保存路径，不强制保存图像 tensor。
    """
```

主实验先用：

```text
camera_id = [1]
```

多 camera 后续再扩展。

---

## 4.2 LiDAR 输入

新增 `preprocess_lidar.py`：

推荐先做轻量 BEV，不要直接 PointNet：

```python
def pcd_to_bev(
    pcd_path: str,
    x_range=(-80, 80),
    y_range=(-80, 80),
    z_range=(-3, 5),
    resolution=0.5,
) -> np.ndarray:
    """
    输出 [C, H, W] BEV:
      C0: occupancy
      C1: max_height
      C2: density
    """
```

cache 策略：

```text
dataset/deepverse_dt31/cache/lidar_bev/{sample_time}_{device_id}.npz
```

如果暂时不想预处理点云，就先保存 pcd 路径，Dataset 读取时再处理。

---

## 4.3 Radar 输入

新增 `preprocess_radar.py`：

DeepVerse 文档里 `dataset.get_sample('radar', index=..., bs_idx=..., ue_idx=...)` 可以拿到 `radar_sample.coeffs`、`radar_sample.waveform`、antenna 信息，示例中 FMCW 参数包括 chirps、samples/chirp、slope、sampling 等。([DeepVerse 6G][2])

先实现两个 fallback：

```python
def extract_radar_feature(radar_sample) -> np.ndarray:
    """
    优先使用 radar_sample.coeffs 生成低维统计特征：
      abs mean/std/max
      phase diff mean/std
      path count if available
    后续再扩展 range-Doppler map。
    """

def radar_to_tensor(radar_sample) -> np.ndarray:
    """
    如果 radar_sample 已有可用 raw/FMCW tensor，则返回。
    否则返回 extract_radar_feature。
    """
```

原因：DT31 的 radar 数据字段可能和示例不完全一致，先做稳健版。

---

## 4.4 Weak wireless 输入

不要把完整 `H` 输入模型。只缓存低维历史无线特征：

```python
weak_wireless_t = [
    beam_label_t / (B-1),
    max_beam_power_t,
    top1_power_t - top2_power_t,
    beam_entropy_t,
    los_binary_t  # 可选，默认不要作为主输入；若用，要单独做 ablation
]
```

建议默认不输入历史 `LoS_status`，因为它和 blockage 标签太近。可以配置：

```yaml
use_past_los_as_input: false
```

历史窗口：

```python
weak_wireless_history = weak_wireless[t-seq_len+1:t+1]
```

---

## 4.5 Noisy position 输入

不要输入 clean location。生成 noisy position：

```python
noisy_pos = clean_pos + np.random.normal(0, sigma, size=clean_pos.shape)
```

配置：

```yaml
position_noise_std: 1.0
position_dropout_prob: 0.1
```

保存：

```text
noisy_position_history: [seq_len, 2]
clean_position_history: [seq_len, 2]  # 只用于 oracle/分析，不作为默认输入
```

---

# 5. 样本构建逻辑

新增 `build_dt31_manifest.py` 或在 `generate_dt31_cache.py` 中调用。

伪代码：

```python
for ue_id in all_ue_ids:
    mobility = dataset.get_sample("mobility", ue_idx=ue_id)
    info = mobility.get_all_samples()
    times = np.array(info["time"])
    locations = np.array(info["location"])

    valid_times = sorted(times)

    for t_anchor in valid_times:
        hist = [t_anchor - seq_len + 1, ..., t_anchor]
        fut = [t_anchor + 1, ..., t_anchor + pred_horizon]

        if any time not available in mobility:
            continue

        if any comm sample unavailable for future:
            continue

        generate labels:
          beam labels for fut
          blockage labels for fut
          trajectory fut from mobility

        generate inputs:
          camera paths for hist
          lidar paths for hist
          radar feature/path for hist
          weak wireless hist from already computed beam/power at hist
          noisy position hist

        append sample row
```

默认任务目标：

```text
beam_label = future t+1 的 beam label
blockage_label = future t+1 的 blockage label
trajectory = t+1:t+K 的 2D trajectory
```

也可以保存多 horizon beam/blockage：

```text
beam_labels_future: [K]
blockage_labels_future: [K]
```

训练时先用 `t+1`，后续再做多 horizon。

---

# 6. Dataset 和 collate

新增 `src/kd_sensing/data/deepverse/dataset.py`：

```python
class DeepVerseMultitaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_path: str,
        cache_root: str,
        split: str,
        modalities: list[str],
        seq_len: int,
        pred_horizon: int,
        image_size: tuple[int, int] = (224, 224),
        lidar_bev: bool = True,
        radar_mode: str = "feature",
        use_clean_position: bool = False,
        use_full_channel: bool = False,
    ):
        ...

    def __getitem__(self, idx):
        return {
            "inputs": {
                "camera": Tensor[T, C, H, W] or None,
                "lidar": Tensor[T, C, H, W] or None,
                "radar": Tensor[T, D] or Tensor[T, C, H, W] or None,
                "weak_wireless": Tensor[T, Dw] or None,
                "noisy_position": Tensor[T, 2] or None,
            },
            "labels": {
                "beam": LongTensor[],
                "blockage": FloatTensor[],
                "trajectory": FloatTensor[K, 2],
            },
            "meta": {
                "sample_id": str,
                "ue_id": int,
                "t_anchor": int,
                "los_status": int,
            }
        }
```

新增 `collate.py`：

```python
def deepverse_collate(batch):
    """
    支持部分模态缺失。
    返回 modality_mask:
      camera/lidar/radar/weak_wireless/noisy_position
    """
```

---

# 7. 模型 baseline

新增 `src/kd_sensing/models/deepverse/multitask_model.py`。

## 7.1 Baseline 1：Late Fusion Multitask

```python
class DeepVerseLateFusionMTL(nn.Module):
    def __init__(self, modalities, d_model=128, num_beams=64, pred_horizon=3):
        ...
```

各模态 encoder：

| 模态             | Encoder                      |
| -------------- | ---------------------------- |
| camera         | ResNet18 frame encoder + GRU |
| LiDAR BEV      | small CNN + GRU              |
| radar feature  | MLP + GRU                    |
| weak wireless  | MLP + GRU                    |
| noisy position | MLP + GRU                    |

融合：

```python
z_m = encoder_m(x_m)  # [B, D]
z = concat([z_m])
z = fusion_mlp(z)
```

任务头：

```python
beam_head: Linear(D, num_beams)
blockage_head: Linear(D, 1)
trajectory_head: Linear(D, pred_horizon * 2)
```

损失：

```python
L = lambda_beam * CE(beam_logits, beam_label)
  + lambda_blockage * BCEWithLogits(blockage_logit, blockage_label)
  + lambda_traj * SmoothL1(traj_pred, traj_label)
```

默认：

```yaml
lambda_beam: 1.0
lambda_blockage: 1.0
lambda_traj: 0.5
```

---

## 7.2 Baseline 2：Task-aware Gated Fusion

新增：

```python
class TaskAwareGatedFusion(nn.Module):
    """
    每个 task 单独生成 modality gate。
    gate_{task,m} = softmax(MLP([z_m, task_embedding]))
    z_task = sum_m gate_{task,m} * z_m
    """
```

输出：

```python
beam_logits = beam_head(z_beam)
blockage_logit = blockage_head(z_blockage)
trajectory_pred = trajectory_head(z_traj)
```

保存 gate：

```python
return {
    "beam_logits": ...,
    "blockage_logits": ...,
    "trajectory": ...,
    "gates": {
        "beam": [B, M],
        "blockage": [B, M],
        "trajectory": [B, M],
    }
}
```

这个是你论文主线最重要的 baseline，因为它可以分析：

```text
beam task 是否更依赖 weak wireless / position
blockage task 是否更依赖 LiDAR / radar / camera
trajectory task 是否更依赖 radar / LiDAR / position
```

---

## 7.3 Baseline 3：Source-dominance diagnosis

实现单模态和组合模态训练：

```text
camera only
lidar only
radar only
weak wireless only
noisy position only
camera+lidar+radar
weak wireless+noisy position
all modalities
oracle: full channel + clean position
```

目的不是追求最好，而是证明：

```text
不同 task 的优势模态不同；
强状态模态会主导某些任务；
感知模态在 blockage / trajectory / degraded setting 下更有价值。
```

---

# 8. 模态失衡分析指标

新增 `metrics.py` 和 report 生成。

必须输出：

## 8.1 任务性能

```text
beam:
  top1_acc
  top3_acc
  top5_acc

blockage:
  accuracy
  F1
  balanced_accuracy
  AUROC if possible

trajectory:
  ADE
  FDE
  RMSE
```

## 8.2 模态贡献

对 task-aware gated model 输出：

```text
mean_gate_by_task.csv
sample_gate_by_task.csv
gate_vs_los_status.csv
gate_vs_position_noise.csv
```

## 8.3 模态消融

训练后做 inference-time ablation：

```python
for each modality m:
    mask modality m
    evaluate performance drop
```

输出：

```text
ablation_drop_by_task.csv
```

格式：

```text
task, removed_modality, delta_metric
beam, weak_wireless, -0.18 top1
beam, lidar, -0.03 top1
blockage, lidar, -0.12 F1
trajectory, radar, +0.08 ADE
```

## 8.4 强模态退化实验

实现：

```text
position noise std = 0, 0.5, 1, 3, 5 meters
weak wireless dropout = 0, 0.1, 0.3, 0.5
camera dropout = 0, 0.3
lidar dropout = 0, 0.3
radar dropout = 0, 0.3
```

输出：

```text
robustness_grid.csv
```

---

# 9. DT31 额外处理方案

因为 DeepVerse6G DT31 不能像普通数据集一样直接读取，需要先通过参数生成。请额外实现一个专门处理 DT31 的脚本。

新增：

```text
scripts/deepverse/generate_dt31_cache.py
```

功能：

```bash
python scripts/deepverse/generate_dt31_cache.py \
  --scenario-root /root/datasets/DeepVerse/scenarios \
  --scenario DT31 \
  --config-m /root/datasets/DeepVerse/scenarios/DT31/param/config.m \
  --output-root /root/projects/KD-for-sensing/dataset/deepverse_dt31/cache \
  --scenes all \
  --seq-len 8 \
  --pred-horizon 3 \
  --beam-codebook-size 64 \
  --enable-camera \
  --enable-lidar \
  --enable-radar \
  --enable-comm \
  --enable-position \
  --position-noise-std 1.0 \
  --train-ratio 0.8 \
  --val-ratio 0.2 \
  --split-by sample
```

脚本流程：

```python
def main():
    args = parse_args()

    set_seed(args.seed)

    generator = DeepVerseDT31Generator(
        scenario_root=args.scenario_root,
        scenario=args.scenario,
        config_m=args.config_m,
        scenes=parse_scenes(args.scenes),
        enable_camera=args.enable_camera,
        enable_lidar=args.enable_lidar,
        enable_radar=args.enable_radar,
        enable_comm=args.enable_comm,
        enable_position=args.enable_position,
    )

    dataset = generator.load_dataset()

    label_builder = DeepVerseLabelBuilder(
        dataset=dataset,
        num_beams=args.beam_codebook_size,
        seq_len=args.seq_len,
        pred_horizon=args.pred_horizon,
        position_noise_std=args.position_noise_std,
        output_root=args.output_root,
    )

    manifest = label_builder.build_manifest()

    split = make_split(
        manifest,
        split_by=args.split_by,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    save manifest, labels, radar features, weak wireless, noisy positions, split, metadata

    run_sanity_checks()
```

注意事项：

1. 首次生成可能慢，因为 DeepVerse 会实时生成 wireless/radar data。
2. 默认启用 radar；如果用户显式关闭 radar，应在 metadata 中移除 `radar` 默认输入。
3. 如果某些 camera_id 或 lidar device 不存在，自动降级为可用 device，并写入 metadata。
4. 如果 `dataset.get_sample('comm-ue', ...)` 某个 UE/time 不存在，跳过该样本。
5. 所有跳过原因要统计：

   ```text
   skipped_no_mobility
   skipped_no_comm
   skipped_no_camera
   skipped_no_lidar
   skipped_short_window
   skipped_nan_label
   ```

---

# 10. 配置文件示例

新增 `configs/deepverse/dt31_generation.yaml`：

```yaml
scenario:
  root: /root/datasets/DeepVerse/scenarios
  name: DT31
  config_m: /root/datasets/DeepVerse/scenarios/DT31/param/config.m
  scenes: all

generation:
  enable_camera: true
  enable_lidar: true
  enable_radar: true
  enable_comm: true
  enable_position: true

sequence:
  seq_len: 8
  pred_horizon: 3
  target_horizon_for_beam: 1
  target_horizon_for_blockage: 1

beam:
  num_beams: 64
  codebook_type: ula_dft
  topk: 5

position:
  use_noisy_position: true
  noise_std: 1.0
  dropout_prob: 0.1
  use_clean_position_as_input: false

wireless:
  use_full_channel_as_input: false
  use_past_los_as_input: false
  weak_features:
    - beam_index
    - max_power
    - top1_minus_top2
    - entropy

modalities:
  camera:
    enabled: true
    camera_ids: [1]
    image_size: [224, 224]
  lidar:
    enabled: true
    mode: bev
    x_range: [-80, 80]
    y_range: [-80, 80]
    z_range: [-3, 5]
    resolution: 0.5
  radar:
    enabled: true
    mode: feature

split:
  split_by: sample
  train_ratio: 0.8
  val_ratio: 0.2
  test_ratio: 0.0
  seed: 42

output:
  root: dataset/deepverse_dt31/cache
```

新增 `configs/deepverse/dt31_multitask.yaml`：

```yaml
data:
  manifest: dataset/deepverse_dt31/cache/samples.parquet
  cache_root: dataset/deepverse_dt31/cache
  modalities:
    - camera
    - lidar
    - radar
    - weak_wireless
    - noisy_position
  seq_len: 8
  pred_horizon: 3

model:
  name: task_aware_gated_mtl
  d_model: 128
  num_beams: 64
  dropout: 0.1

loss:
  beam: 1.0
  blockage: 1.0
  trajectory: 0.5

train:
  batch_size: 16
  epochs: 50
  lr: 0.0003
  weight_decay: 0.0001
  num_workers: 4
  seed: 42

eval:
  save_gates: true
  save_ablation: true
```

---

# 11. 训练脚本

新增：

```bash
python scripts/deepverse/train_deepverse_multitask.py \
  --config configs/deepverse/dt31_multitask.yaml \
  --exp-name dt31_all_task_aware_gate
```

训练脚本要求：

1. 支持选择模态：

   ```bash
   --modalities camera,lidar,radar
   --modalities weak_wireless,noisy_position
   --modalities camera,lidar,radar,weak_wireless,noisy_position
   ```
2. 支持模型：

   ```bash
   --model late_fusion
   --model task_aware_gate
   ```
3. 保存：

   ```text
   checkpoints/best.pt
   reports/metrics.json
   reports/metrics_by_los.csv
   reports/gates_by_task.csv
   reports/ablation_drop_by_task.csv
   ```

---

# 12. Sanity check 必须实现

新增：

```bash
python scripts/deepverse/run_deepverse_sanity.py \
  --cache-root dataset/deepverse_dt31/cache
```

检查内容：

```text
1. manifest 是否非空
2. train/val/test 是否无 sample_id 重叠
3. 默认 split_by=sample 时 train/val 是否都非空；显式 split_by=ue 时 train/val/test 是否无 ue_id 重叠
4. beam label 是否在 [0, num_beams-1]
5. blockage label 是否只有 0/1
6. trajectory shape 是否为 [K, 2]
7. camera/lidar/radar 路径是否存在
8. weak wireless 是否没有 NaN/Inf
9. noisy position 与 clean position 是否不同
10. LoS/blockage 类别是否极端不平衡；如果正负样本比例小于 5%，给 warning
```

输出：

```text
dataset/deepverse_dt31/cache/sanity_report.json
```

---

# 13. 最小可运行版本优先级

请按这个顺序实现，不要一开始做太复杂。

## Phase 1：先跑通 DT31 数据生成

只需要：

```text
comm channel
LoS_status
mobility location
camera path
lidar path
```

输出 manifest + labels。

## Phase 2：跑通模型训练

先用：

```text
weak_wireless + noisy_position
```

和：

```text
camera + lidar
```

Radar 在 Phase 1 先用低维统计特征，不直接做复杂 range-Doppler。

## Phase 3：加入 radar range-Doppler 表示

在低维 radar 特征可跑通后，再扩展 range-Doppler map。

## Phase 4：加入 task-aware gated fusion

输出不同 task 的 gate，可视化模态偏好。

## Phase 5：做模态失衡实验

包括：

```text
单模态
模态组合
强模态退化
inference-time modality ablation
task gate analysis
```

---

# 14. 实验矩阵

实现 `configs/deepverse/dt31_experiments.yaml`：

```yaml
experiments:
  - name: camera_only
    modalities: [camera]

  - name: lidar_only
    modalities: [lidar]

  - name: radar_only
    modalities: [radar]

  - name: weak_wireless_only
    modalities: [weak_wireless]

  - name: noisy_position_only
    modalities: [noisy_position]

  - name: sensing_only
    modalities: [camera, lidar, radar]

  - name: strong_state_only
    modalities: [weak_wireless, noisy_position]

  - name: all_modalities_late_fusion
    model: late_fusion
    modalities: [camera, lidar, radar, weak_wireless, noisy_position]

  - name: all_modalities_task_gate
    model: task_aware_gate
    modalities: [camera, lidar, radar, weak_wireless, noisy_position]

robustness:
  position_noise_std: [0.0, 0.5, 1.0, 3.0, 5.0]
  weak_wireless_dropout: [0.0, 0.1, 0.3, 0.5]
```

---

# 15. 你这篇论文要看的核心结果

最终报告里至少输出这 4 张表：

## 表 1：单模态/模态组合性能

```text
modality setting | beam top1 | beam top3 | blockage F1 | trajectory ADE | average score
```

用于证明：

```text
不同任务优势模态不同。
```

## 表 2：强模态退化实验

```text
noise/dropout setting | method | beam top1 | blockage F1 | trajectory ADE
```

用于证明：

```text
强状态模态退化时，感知模态有补偿价值。
```

## 表 3：模态消融 drop

```text
removed modality | delta beam | delta blockage | delta trajectory
```

用于证明：

```text
不同任务依赖不同模态，存在 task-dependent modality imbalance。
```

## 表 4：task gate 平均权重

```text
task | camera | lidar | radar | weak wireless | noisy position
beam
blockage
trajectory
```

用于支撑你的方法解释性。

---

# 16. 关键实现注意事项

1. 不要默认把 `full channel coeffs` 作为输入。它只能用于生成 beam label 和 oracle。
2. 不要默认把 `LoS_status` 作为输入。它只能用于生成 blockage label。
3. 不要默认把 `clean location` 作为输入。它只能用于 trajectory label 或 oracle。
4. `bounding_box` 暂时只保存，后续可以生成 geometric blockage / nearest blocker distance。
5. split 必须优先按 UE/object id 做，否则时间序列泄漏会让结果虚高。
6. radar 字段可能不稳定，先实现 feature fallback。
7. 所有 DeepVerse API 调用都要 try/except，并记录失败原因。
8. 所有生成结果要可缓存，避免每次训练重新调用 generator。
9. 先保证最小版本跑通，不要一开始做复杂 BEV/range-Doppler。

---

# 17. 最终预期命令

```bash
# 1. 生成 DT31 cache
python scripts/deepverse/generate_dt31_cache.py \
  --config configs/deepverse/dt31_generation.yaml

# 2. 检查数据
python scripts/deepverse/run_deepverse_sanity.py \
  --cache-root dataset/deepverse_dt31/cache

# 3. 训练 late fusion baseline
python scripts/deepverse/train_deepverse_multitask.py \
  --config configs/deepverse/dt31_multitask.yaml \
  --model late_fusion \
  --modalities camera,lidar,radar,weak_wireless,noisy_position \
  --exp-name dt31_late_fusion_all

# 4. 训练 task-aware gate 方法
python scripts/deepverse/train_deepverse_multitask.py \
  --config configs/deepverse/dt31_multitask.yaml \
  --model task_aware_gate \
  --modalities camera,lidar,radar,weak_wireless,noisy_position \
  --exp-name dt31_task_gate_all

# 5. 批量跑实验矩阵
python scripts/deepverse/eval_deepverse_multitask.py \
  --experiments configs/deepverse/dt31_experiments.yaml
```

---

# 你自己要把握的主线

你换成 DeepVerse 后，主线不要写成“我做了一个更复杂的多模态融合模型”，而要写成：

> DeepVerse6G-DT31 allows us to decouple supervision sources from input modalities. We derive beam labels from ray-tracing channels, blockage labels from LoS status, and trajectory labels from mobility ground truth, while using camera, LiDAR, radar, weak wireless history, and noisy position as model inputs. This enables controlled study of task-dependent modality imbalance in ISAC multitask learning.

中文就是：

> DeepVerse6G-DT31 可以把标签生成真值和模型输入模态分开。我们用 ray-tracing channel 生成 beam 标签，用 LoS status 生成 blockage 标签，用 mobility ground truth 生成 trajectory 标签，但模型输入只使用 camera、LiDAR、radar、弱化历史无线特征和带噪位置。这样可以更干净地研究 ISAC 多任务中的任务依赖型模态失衡。

[1]: https://github.com/wireless-intelligence-lab/DeepVerse6G-python "GitHub - wireless-intelligence-lab/DeepVerse6G-python · GitHub"
[2]: https://deepverse6g.net/html/examples/getting_started.html "Starting with DeepVerse6G - DeepVerse6G documentation"
[3]: https://deepverse6g.net/html/examples/mobility.html "Mobility Data - DeepVerse6G documentation"
