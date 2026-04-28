我看了公开实现后，结论很明确：**别人的程序基本没有在 Dataset 初始化阶段把全量 LiDAR BEV 读进内存再 `concatenate` 算 mean/std。** 你现在这个 37GB RSS 的问题，主要是你自己的 `lidar_normalize` 设计导致的，不是 DeepSense-6G 常规做法。

## 1. TII 多模态方案是“按需读 + 局部归一化”

TII 的公开代码里，`CARLA_Data.__init__` 主要读 CSV、保存配置、做 GPS normalization；LiDAR 并没有在初始化阶段全量加载。它在 `__getitem__` 里为每个样本的 5 个时间帧构造路径，然后每帧现场 `o3d.io.read_point_cloud(...)` 读取点云，再调用 `lidar_to_histogram_features(...)` 转成 BEV histogram。([GitHub][1])

它的 LiDAR BEV normalization 也不是全局 mean/std，而是**固定规则归一化**：把点云的 x-y 坐标投到 256×256 网格，用 `np.histogramdd` 统计每个 cell 的点数，然后把大于 5 的 cell 截断到 5，再除以 5。也就是说输出天然在 `[0,1]` 范围内。([GitHub][1])

它还用了 scenario-specific FoV。比如 scenario33 使用 `xbins = [-50, 0]`、`ybins = [-12, 7]`，这是按场景裁剪 BEV，而不是把所有原始点云无脑 rasterize。([GitHub][1])

更关键的是，TII 最佳提交的命令里是：

```bash
--filtered 0 --custom_FoV_lidar 1
```

也就是**最佳配置没有启用 filtered LiDAR，但启用了 custom FoV LiDAR**。([GitHub][2])

所以你的短期命令：

```bash
-o data.dataset.lidar_normalize=false
```

和 TII 的思路是接近的：**只要你的 BEV 本身已经是 density/height/intensity 的合理范围，先不做全局 z-score 是完全可以接受的。**

## 2. TII 训练时也没有预缓存全量 LiDAR

TII 训练代码只是创建 `CARLA_Data`、`ConcatDataset`、`random_split`，然后交给 `DataLoader`。训练 loader 用 `num_workers=8, pin_memory=True`，但没有看到初始化阶段对全量 LiDAR 做一次 `cat/stack` 的 normalizer pass。([GitHub][3])

这说明它们依赖的是：

```text
Dataset 只保存路径/小数组
DataLoader batch 时读取 LiDAR
每个样本局部转 BEV
BEV 本身用固定规则归一化到合理范围
```

而不是：

```text
初始化 Dataset
读完整个训练集 LiDAR
拼成巨大 tensor
算 mean/std
再进入训练
```

你现在的卡住点正好是第二种。

## 3. Scenario 8 的 LiDAR-only 代码也是按需读取

另一个公开实现 `acyiobs/lidar_beam_tracking` 用的是 Scenario 8。它不是 2D BEV，而是 8 帧 × 216 angle-bin 的 LiDAR 序列。它在 `create_samples` 里只把 CSV 里的序列路径和 beam label 列表存起来；真正的 LiDAR 数据是在 `__getitem__` 里逐帧 `loadmat(...)` 读取，然后除以 10，填到 `torch.zeros((seq_len, 216))`。([GitHub][4])

训练时也只是：

```python
train_loader = DataLoader(DataFeed(...), batch_size=32, shuffle=True)
```

没有全量 LiDAR cache，也没有全局 concatenate。([GitHub][5])

所以第二个实现也支持同一个判断：**别人是按样本懒加载，normalization 用简单固定缩放，不在训练前拼全量数据。**

## 4. 你现在最合理的改法

短期：继续用这个跑起来。

```bash
conda run --no-capture-output -n kd_mm_beam python -u scripts/train.py \
  --config configs/lidar/teacher_no_kd.yaml \
  -o data.dataset.lidar_normalize=false
```

长期：不要恢复“全量 cache + concatenate 算 normalizer”。应该改成**流式统计一次 mean/std，然后保存到文件**。大致逻辑如下：

```python
@torch.no_grad()
def compute_lidar_stats_streaming(dataset, save_path):
    # dataset[i]["lidar"] shape: [T, C, H, W]
    n = 0
    sum_c = None
    sumsq_c = None

    for i in tqdm(range(len(dataset)), desc="Compute LiDAR stats"):
        x = dataset[i]["lidar"].float()  # [T, C, H, W]
        # 按 C 统计，合并 T/H/W
        x = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)

        if sum_c is None:
            sum_c = torch.zeros(x.shape[0], dtype=torch.float64)
            sumsq_c = torch.zeros(x.shape[0], dtype=torch.float64)

        sum_c += x.double().sum(dim=1)
        sumsq_c += (x.double() ** 2).sum(dim=1)
        n += x.shape[1]

    mean = sum_c / n
    var = sumsq_c / n - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-12))

    torch.save({"mean": mean.float(), "std": std.float(), "n": n}, save_path)
```

然后 Dataset 里只加载这个小文件：

```python
stats = torch.load("lidar_stats.pt")
self.lidar_mean = stats["mean"].view(1, -1, 1, 1)
self.lidar_std = stats["std"].view(1, -1, 1, 1)
```

`__getitem__` 里：

```python
x = (x - self.lidar_mean) / (self.lidar_std + 1e-6)
```

## 5. 我的建议排序

当前项目已按这个方向修改：LiDAR-only 和包含 LiDAR 的 fusion 配置默认使用：

```yaml
data:
  dataset:
    lidar_normalize: false
```

因为公开 DeepSense-6G LiDAR 实现基本都是局部归一化/固定缩放，而不是全局 z-score。

如果需要全局通道 mean/std，现在使用显式 streaming stats 配置：

```yaml
data:
  dataset:
    lidar_normalization:
      enabled: true
      mode: streaming_stats
      stats_path: outputs/cache/lidar_stats_train.npz
```

训练命令建议使用无捕获输出，避免 tqdm 被 `conda run` 缓冲：

```bash
conda run --no-capture-output -n kd_mm_beam python -u scripts/train.py \
  --config configs/lidar/teacher_no_kd.yaml
```

不要再做：

```python
all_lidar = torch.cat([...])
mean = all_lidar.mean(...)
```

你看到的 37GB RSS 不是训练慢，而是初始化阶段把训练集 LiDAR 当成一个巨大 tensor 处理了。公开实现的共同点是：**LiDAR 大数据不进 Dataset init，只在 batch 里进内存。**

[1]: https://github.com/ITU-AI-ML-in-5G-Challenge/DeepSense6G_TII/blob/main/data2_seq.py "DeepSense6G_TII/data2_seq.py at main · ITU-AI-ML-in-5G-Challenge/DeepSense6G_TII · GitHub"
[2]: https://github.com/ITU-AI-ML-in-5G-Challenge/DeepSense6G_TII "GitHub - ITU-AI-ML-in-5G-Challenge/DeepSense6G_TII: The repository contains code, report and presentation for the solution of Team TII for ITU AI/ML in 5G Grand Challenge 2022: ML5G-PS-011: Multi Modal Beam Prediction: Towards Generalization · GitHub"
[3]: https://github.com/ITU-AI-ML-in-5G-Challenge/DeepSense6G_TII/blob/main/train2_seq.py "DeepSense6G_TII/train2_seq.py at main · ITU-AI-ML-in-5G-Challenge/DeepSense6G_TII · GitHub"
[4]: https://github.com/acyiobs/lidar_beam_tracking/blob/main/data_feed.py "lidar_beam_tracking/data_feed.py at main · acyiobs/lidar_beam_tracking · GitHub"
[5]: https://github.com/acyiobs/lidar_beam_tracking/blob/main/train_model.py "lidar_beam_tracking/train_model.py at main · acyiobs/lidar_beam_tracking · GitHub"
