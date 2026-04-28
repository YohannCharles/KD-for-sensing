结论先说：**如果你做的是 DeepSense-6G 场景31–34，尤其是场景33这种多模态波束预测，LiDAR 最推荐的预处理是“ROI/FoV裁剪 + 背景过滤 + BEV伪图像化 + 轻量几何增强”。**
如果你做的是早期那种 **LiDAR-only / 2D LiDAR future beam prediction**，则更推荐 **极坐标 angle-bin/range 向量 + GRU 时序建模**，不要强行转 BEV。

## 1. DeepSense-6G 里 LiDAR 常见预处理方式

### 方法一：Angle-bin / 极坐标距离向量

这是 LiDAR-aided future beam prediction 那篇最典型的处理。原始 LiDAR 是 360° 点云，每个样本包含角度和距离，预处理后变成 **216 个角度 bin**，角度分辨率约 0.97°；因为波束码本有 64 个波束，相邻波束约 2.8°，所以这个 LiDAR 角度分辨率已经足够支持波束预测。作者还用历史窗口 (W=8) 来预测当前和未来波束。

这种方法的优点是非常轻量，和“波束本质上主要和角度有关”高度匹配。它在该工作中 current beam 和 first future beam 的 Top-5 accuracy 分别达到 95.6% 和 95.0%，而且只需要约 10.4% 的 beam training overhead。

但它更适合 **2D LiDAR / 单模态 / 轻量时序模型**。如果你现在用的是场景33的 3D point cloud，多模态融合里直接用这种一维角度向量会损失很多空间结构信息。

---

### 方法二：3D histogram / voxel grid

早期 LiDAR beam selection 论文把点云 (C) 转成 3D histogram：先把点云从相对坐标转换到绝对位置，然后在固定覆盖区域里均匀量化成网格，每个 bin 存储落入该区域的点数；同时会丢弃离车辆太远的点。([users.ece.utexas.edu][1])

这个方向本质上是 **体素化 / 占据栅格化**。优点是保留了 3D 空间结构，比一维 angle-bin 信息更多；缺点是维度较大，稀疏，训练开销更高，而且如果直接用 3D CNN 或 PointNet 类模型，在 DeepSense-6G 这种样本规模下容易过拟合。该论文还会去除低高度点，例如 (z<0.1m) 的地面反射点，并限制最大距离 (d_{max}=25m)。([users.ece.utexas.edu][1])

---

### 方法三：BEV 伪图像化

这是我最建议你优先采用的方式。TII 的 DeepSense6G 多模态 Transformer 方案就是把 LiDAR 点云转成 **Bird’s-Eye View，BEV**：先在 ROI 内把点云离散化到 2D 网格，再把 **height、intensity、density** 分别映射到 RGB 三个通道，形成类似图像的 LiDAR BEV。作者认为 BEV 可以保留点云基本结构和深度信息，同时比 PointNet 更省计算，也方便后续 CNN/ResNet 提特征并与图像、雷达特征融合。([arXiv][2])

对场景31–34这种多模态任务，论文还提到每个样本包含 5 个时刻的 camera / LiDAR / radar，以及 2 个时刻的 GPS，用于预测最优波束。因此 BEV 的好处是：LiDAR、camera、radar 都可以变成 2D map，然后用类似 ResNet 的编码器统一处理，再做时序/跨模态融合。([arXiv][2])

---

### 方法四：背景过滤 + FoV 对齐

TII 方案还对 LiDAR 做了两个很关键的处理。第一是 **background filtering**：用每个场景所有帧的 moving average 估计静态背景，然后从每帧点云中减去静态物体点，保留移动车辆附近区域。第二是 **FoV calibration**：裁剪 BEV 投影，让 LiDAR 的视场和图像视场尽量一致，这样 CNN 和 Transformer 更容易学习图像-LiDAR之间的对应关系。([arXiv][2])

这个对你现在的多模态融合很重要。因为你之前一直遇到“某些模态效果弱、多模态不一定提升”的问题，LiDAR 如果没有和 camera / radar / GPS 在空间上对齐，很容易变成噪声模态。

---

### 方法五：安全数据增强

LiDAR 点云增强一般不要乱旋转、乱平移，因为波束标签和几何方向强相关。TII 方案使用的增强比较安全：随机下采样约 10%，给点云加入小的 3D 位置扰动；此外还做了多模态水平翻转，并把波束标签同步变换为 (65-\text{原beam index})。([arXiv][2])

这类增强适合你当前任务，但要注意：**水平翻转必须同时翻 LiDAR、图像、雷达，并同步改 beam label**，否则标签会错。

## 2. 哪种最好？

对你现在的任务，我建议排序如下：

| 任务情况                                   | 最推荐预处理                                     | 原因                                          |
| -------------------------------------- | ------------------------------------------ | ------------------------------------------- |
| DeepSense-6G 场景33，多模态融合，预测当前/下一时刻 beam | **BEV + ROI/FoV裁剪 + 背景过滤 + 时序建模**          | 最适合和 image/radar 一起用 CNN/Transformer/GRU 融合 |
| LiDAR 单模态，轻量 future beam prediction    | **angle-bin/range vector + GRU**           | 和波束角度强相关，参数少，实时性好                           |
| 想研究 LiDAR 本身的 3D 几何贡献                  | **BEV 与 PointPillars/PointNet 做 ablation** | 可以证明 BEV 是否损失 3D 信息                         |
| 小样本、泛化到新场景                             | **不要上复杂 PointNet++/SparseConv 作为主模型**      | 容易过拟合，且不一定比 BEV 稳                           |

所以，**场景33下最优先尝试 BEV，不是 PointNet，也不是直接喂 raw point cloud。**

## 3. 给你当前项目的推荐版本

我建议你让 Codex 按这个流程改：

```text
LiDAR preprocessing for DeepSense-6G Scenario 33:

1. Read raw .pcd point cloud.
2. Remove invalid points: NaN / inf / extremely sparse outliers.
3. Apply ROI crop around road and BS-visible region.
4. Optional: remove ground points with small z value, but do not blindly remove all static structures.
5. Build per-scenario static background map using temporal average / occupancy frequency.
6. Subtract stable background points, but keep possible LoS/NLoS blocking objects and vehicles.
7. Convert point cloud to BEV pseudo-image:
   - channel 1: max height per grid cell
   - channel 2: max or mean intensity per grid cell
   - channel 3: log-normalized point density per grid cell
8. Resize BEV to 224×224 or 256×256.
9. Normalize each channel using training-set statistics.
10. For temporal input, keep 5 LiDAR BEV frames as sequence:
    shape = [T, C, H, W], T=5, C=3.
11. Use shared LiDAR CNN/ResNet encoder for each frame.
12. Feed temporal LiDAR features to GRU/Transformer.
13. Training augmentation only:
    - random point dropout/downsampling around 10%
    - small 3D jitter
    - optional horizontal flip with beam label = 65 - beam_label
```

有一个细节要注意：TII 的 best run 里命令参数显示 `filtered=0`、`custom_FoV_lidar=1`，也就是说**他们最终最优配置未必用了背景过滤，但用了自定义 FoV LiDAR 裁剪**。这说明背景过滤不是绝对必做，建议你把它作为 ablation：`BEV`、`BEV+FoV`、`BEV+FoV+background filtering` 分别跑。([GitHub][3])

## 4. 我的最终建议

你现在如果是做 **多模态波束预测/下一时刻 beam prediction**，就不要直接 raw point cloud，也不要先上复杂 PointNet。最稳的是：

**LiDAR → ROI/FoV crop → BEV(height/intensity/density) → ResNet18/轻量CNN → GRU/Temporal Transformer → fusion。**

然后做三个 ablation：

1. `LiDAR-1D angle-bin + GRU`
2. `LiDAR-BEV + ResNet18 + GRU`
3. `LiDAR-BEV + FoV crop + safe augmentation + ResNet18 + GRU`

如果第 2/3 比第 1 强，说明 3D 空间结构确实有用；如果第 1 反而更强，说明你的任务主要依赖角度/位置，LiDAR 的复杂空间信息可能在当前标注和模型下没有被有效利用。

[1]: https://users.ece.utexas.edu/~shakkott/Pubs/LIDAR.pdf "LIDAR Data for Deep Learning-Based mmWave Beam-Selection"
[2]: https://arxiv.org/html/2309.11811 "MULTIMODAL TRANSFORMERS FOR WIRELESS COMMUNICATIONS: A CASE STUDY IN BEAM PREDICTION"
[3]: https://github.com/ITU-AI-ML-in-5G-Challenge/DeepSense6G_TII "GitHub - ITU-AI-ML-in-5G-Challenge/DeepSense6G_TII: The repository contains code, report and presentation for the solution of Team TII for ITU AI/ML in 5G Grand Challenge 2022: ML5G-PS-011: Multi Modal Beam Prediction: Towards Generalization · GitHub"
