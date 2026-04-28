一般对 **DeepSense-6G 的 GPS/position 数据**，最常见、也最靠谱的预处理不是直接用经纬度，而是：

> **经纬度 lat/lon → UTM/XY 米制坐标 → 减去 BS 坐标得到相对位置 → 归一化/标准化 → 加入时间差分特征。**

这也是我最建议你现在用的版本。

DeepSense-6G 的相关任务里，GPS 通常作为 UE 位置输入，用来预测最优 beam；官方 position-only baseline 里会读取 `unit2_loc_1`、`unit2_loc_2` 两个 UE 位置样本和 `unit1_loc` 基站位置，然后把 lat/lon 转成 UTM 的 XY 坐标，再做 **UE-BS 相对坐标**，最后 min-max 归一化后送进 GRU。([GitHub][1]) 2022 多模态 beam prediction challenge 也明确是“5 个 camera/LiDAR/radar 样本 + 2 个 GPS 样本”来预测最优 beam。([GitHub][2])

## 1. 常见 GPS 预处理方法

### 方法 A：原始经纬度直接 min-max

就是直接把 `[lat, lon]` 做 min-max 到 `[0,1]`。这个最简单，但我不推荐。因为经纬度不是等距坐标，纬度和经度的物理尺度不一样，而且模型学到的是“地球坐标系下的数值关系”，不是“UE 相对 BS 的几何关系”。

适合做最低级 baseline，不适合作为你论文里的主方法。

### 方法 B：经纬度转 UTM/XY，再归一化

这一步把 GPS 的经纬度转成以“米”为单位的平面坐标。相关代码里 `utm.from_latlon()` 就是这么做的。Position-Aided Beam Prediction 的开源代码里也把 lat/lon 转成 cartesian/XY 坐标，并提供了多种 normalization 方式，包括原始 lat/lon、UTM cartesian、相对坐标、极坐标等。([GitHub][3])

这个比方法 A 明显合理，因为毫米波 beam 本质上和空间几何位置有关，用米制坐标更符合物理意义。

### 方法 C：UTM/XY 后减去 BS 坐标，得到相对坐标

这是最常用、也最推荐的基础方案：

[
x_{rel}=x_{UE}-x_{BS},\quad y_{rel}=y_{UE}-y_{BS}
]

DeepSense 官方 position-only baseline 采用的就是这个思路：先把 UE 和 BS 的 lat/lon 都转成 XY，然后计算 `pos_diff = pos_ue_cart - pos_bs_cart`，再做 min-max 归一化。([GitHub][1])

这个方法比“绝对坐标”更适合 beam prediction，因为 beam 主要由 UE 相对于 BS 的方向、距离、遮挡决定，而不是 UE 在全球地图上的绝对经纬度。

### 方法 D：相对坐标转极坐标：距离 + 角度

也就是：

[
d=\sqrt{x_{rel}^2+y_{rel}^2},\quad \theta=\arctan2(y_{rel},x_{rel})
]

这个有通信味儿，因为 beam 和角度强相关，路径损耗和距离强相关。Position-Aided Beam Prediction 的代码里也提供了这种 polar-coordinate normalization：先转 cartesian，再算相对 BS 的距离和角度。([GitHub][3])

但注意：角度有 (-\pi) 到 (\pi) 的跳变问题，所以最好不要直接用 `theta`，而是用：

[
\sin\theta,\quad \cos\theta
]

这样模型更稳定。

### 方法 E：GPS 轨迹平滑 / 去噪

GPS 会有跳点、漂移、系统误差。相关验证论文指出，DeepSense 里的地理位置数据会受到干扰、卫星数量和几何、接收机质量等因素影响，并且在 Scenario 9 中能看到明显的 run-level 系统偏移。([IFIP Open Digital Library][4])

常见做法是：

使用移动平均、移动中值、Kalman filter 或基于速度阈值的 outlier removal。但我建议你先别上来就复杂滤波，因为你只有 2 个 GPS 历史点时，强滤波意义不大，反而可能抹掉“下一时刻运动趋势”。

### 方法 F：量化 / 网格化 GPS

有些工作会把连续坐标映射到离散 bin 或 lookup table。2025 的 DeepSense GPS beamforming 分析提到，已有工作会把坐标量化到离散值。([IFIP Open Digital Library][4])

这个适合 KNN、lookup table、轻量部署，但不太适合你现在的深度多模态融合模型。深度模型最好保留连续坐标，再由 MLP/GRU/Transformer 自己学习。

## 2. 哪种最好？

对你现在这个任务，我建议：

> **主方案：BS-relative UTM Cartesian + train-only 标准化/归一化 + 时间差分特征。**
> 也就是不要只输入两个 GPS 点，而是输入“当前位置 + 运动趋势 + 几何角度”。

具体输入可以这样设计：

[
[x_t,y_t,\Delta x,\Delta y,v,d,\sin\theta,\cos\theta]
]

其中：

[
\Delta x=x_t-x_{t-1},\quad \Delta y=y_t-y_{t-1}
]

[
v=\sqrt{\Delta x^2+\Delta y^2}
]

[
d=\sqrt{x_t^2+y_t^2},\quad \theta=\arctan2(y_t,x_t)
]

这样比只喂 `[lat, lon]` 或 `[x, y]` 更适合 **预测下一时刻 beam**。因为下一时刻 beam 不只取决于当前位置，还取决于 UE 往哪个方向运动。

## 3. 你论文里建议做的 ablation

你可以把 GPS 预处理分成下面几组实验：

| 名称                | GPS 输入                      | 作用           |
| ----------------- | --------------------------- | ------------ |
| GPS-Raw           | `[lat, lon]` min-max        | 最弱 baseline  |
| GPS-UTM           | `[x, y]` UTM 坐标             | 验证米制坐标是否有效   |
| GPS-Rel           | `[x_rel, y_rel]`            | 验证相对 BS 几何信息 |
| GPS-Rel-Polar     | `[d, sinθ, cosθ]`           | 验证角度/距离先验    |
| GPS-Motion        | `[x,y,Δx,Δy,v,d,sinθ,cosθ]` | 推荐主方法        |
| GPS-Motion-Smooth | GPS-Motion + 简单平滑           | 验证去噪是否有用     |

我觉得你最终主模型应该用 **GPS-Motion**，而不是纯 GPS-Rel-Polar。原因是：beam 与角度关系强，但下一时刻预测还需要运动方向；只用距离角度会丢掉短时速度信息。

## 4. 一个很关键的坑：不要用全数据算 min-max

DeepSense baseline 代码里是先对所有样本算 `pos_min` / `pos_max`，再 split train/val/test。([GitHub][1]) 这对复现实验没问题，但如果你写论文，最好改成：

> **只用训练集计算 scaler，然后应用到 val/test。**

否则严格来说有轻微数据泄漏。尤其你现在做“预测下一时刻 beam”，如果 val/test 的位置范围提前参与归一化，会让实验不够严谨。

推荐这样：

训练集：

[
\mu_x,\sigma_x,\mu_y,\sigma_y
]

验证/测试集直接用训练集的均值方差：

[
x'=\frac{x-\mu_{train}}{\sigma_{train}}
]

如果位置有明显异常值，可以用 **RobustScaler**，即中位数和 IQR，而不是均值方差。

## 5. 给你直接发 Codex 的修改建议

你可以让 Codex 这样改：

```text
请修改 DeepSense-6G GPS/position preprocessing：

1. 不要直接使用原始 lat/lon。
2. 对 UE 和 BS 的 GPS 经纬度使用 utm.from_latlon 转换为米制 XY 坐标。
3. 计算 UE 相对 BS 的坐标：
   x_rel = x_ue - x_bs
   y_rel = y_ue - y_bs
4. 对每个样本的两个 GPS 历史点，构造以下特征：
   x_t, y_t,
   dx = x_t - x_{t-1},
   dy = y_t - y_{t-1},
   speed = sqrt(dx^2 + dy^2),
   dist = sqrt(x_t^2 + y_t^2),
   sin_theta = sin(arctan2(y_t, x_t)),
   cos_theta = cos(arctan2(y_t, x_t))
5. scaler 只能在 train split 上 fit，然后用于 val/test，避免数据泄漏。
6. GPS encoder 使用小型 MLP：
   Linear(8, 64) + LayerNorm + GELU + Dropout(0.1)
   Linear(64, d_model)
7. 保留 ablation：
   GPS-Raw, GPS-UTM, GPS-Rel, GPS-Rel-Polar, GPS-Motion。
```

## 6. 我的结论

如果你只是想复现官方 baseline，用：

> **UTM + UE-BS 相对坐标 + min-max normalization**

如果你要写论文、做更合理的下一时刻 beam prediction，用：

> **UTM + UE-BS 相对坐标 + 训练集标准化 + Δx/Δy/speed/d/sinθ/cosθ**

这应该是你目前最稳的 GPS 预处理方案。它既符合 DeepSense 现有代码习惯，又比官方 position-only baseline 更适合“未来 beam prediction”。

[1]: https://github.com/DeepSense6G/Multi-Modal-Beam-Prediction-Challenge-2022-Baseline/blob/main/position_only_baseline.py "Multi-Modal-Beam-Prediction-Challenge-2022-Baseline/position_only_baseline.py at main · DeepSense6G/Multi-Modal-Beam-Prediction-Challenge-2022-Baseline · GitHub"
[2]: https://github.com/ITU-AI-ML-in-5G-Challenge/DeepSense6G_TII "GitHub - ITU-AI-ML-in-5G-Challenge/DeepSense6G_TII: The repository contains code, report and presentation for the solution of Team TII for ITU AI/ML in 5G Grand Challenge 2022: ML5G-PS-011: Multi Modal Beam Prediction: Towards Generalization · GitHub"
[3]: https://raw.githubusercontent.com/jmoraispk/Position-Beam-Prediction/main/train_test_func.py "raw.githubusercontent.com"
[4]: https://opendl.ifip-tc6.org/db/conf/wmnc/wmnc2025/1571178178.pdf "Challenges of Predictive Beamforming Using Geographical Positioning: Insights from the DeepSense Dataset"
