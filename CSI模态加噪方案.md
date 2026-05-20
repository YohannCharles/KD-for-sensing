对 MMW 这个数据集，我觉得最适合的思路不是“随便给 CSI 加噪”，而是：

> **MMW 的 channel 是 Sionna ray tracing 生成的高质量多径真值，因此它天然信息上限很高；你要做的是把这个“完美 CSI / path-level channel”退化成更接近真实估计 CSI 的形式，使它仍然有用，但更难学。**

MMW 页面说明它包含 7 类模态，channel 模态由 Sionna ray tracing 生成，包含 **complex channel gain、path delay、AoD、AoA** 等多径属性；并且所有模态按六位 frame number 同步对齐。([乐量][1]) CAV 侧的 `_paths.npy` 表示从 RSU 到 CAV 的 V2I 下行信道数据。([乐量][2]) 所以你可以直接围绕 **多径、角度、时延、阵列、时间同步** 这几类做退化。

---

## 1. 复增益噪声：对 complex gain 加信道估计误差

MMW 里最核心的是 multipath complex gain (a)。页面给出的 complex gain 维度包含 Rx 天线、Tx 天线、paths 和 time steps。([乐量][2])

你可以对每条路径的复增益加噪：

```text
a_noisy = a + n
n ~ CN(0, σ²)
```

推荐等级：

| 等级     | SNR   |
| ------ | ----- |
| mild   | 20 dB |
| medium | 10 dB |
| hard   | 5 dB  |
| severe | 0 dB  |

这是最标准、最容易解释的方式。它模拟真实系统里的 **信道估计误差、低 SNR、接收噪声**。

但是我不建议只做这个，因为 MMW 的 path-level CSI 太干净，只加 AWGN 可能不足以形成你想要的“高上限但难学”。

---

## 2. 多径 path dropout：随机丢失部分反射/散射路径

MMW 的 channel 是多径形式，包括多条 path 的 gain、delay、AoA、AoD。([乐量][2]) 真实系统里，弱路径不一定都能被准确估计出来，所以可以随机丢掉一部分 path：

```text
a[:, :, :, :, :, dropped_paths, :] = 0
```

推荐：

| 等级     | path dropout |
| ------ | ------------ |
| mild   | 10%          |
| medium | 20%          |
| hard   | 30%          |
| severe | 50%          |

更真实的做法是：**优先丢弱路径，而不是完全随机丢路径**。

```text
按 |a| 从小到大排序，随机/按比例丢掉弱反射路径
```

这比随机丢更符合实际，因为低功率 path 更容易估计失败。

对你的研究来说，这个很合适：CSI 仍然保留主路径和部分几何传播信息，但高维细节被破坏，学习难度明显上升。

---

## 3. LOS / dominant path attenuation：削弱最强路径

如果只丢弱路径，模型可能仍然很容易依赖最强路径。你可以设计一个更难的版本：**不删除最强路径，而是衰减最强路径**。

```text
a_dominant = γ * a_dominant
γ ∈ {0.7, 0.5, 0.3}
```

这个模拟遮挡、半遮挡、车辆阻挡、人体/建筑物导致的主径衰落。

注意：不要直接把 strongest path 置零。直接删除最强 path 可能会让 CSI 从“难学”变成“标签信息严重缺失”。建议先做：

```text
γ = 0.5
```

也就是主径功率衰减一半左右。

---

## 4. delay 噪声 / delay 量化：破坏时延结构

MMW 明确提供 multipath delay (\tau)。([乐量][2]) 你可以对 delay 做扰动：

```text
τ_noisy = τ + ε
ε ~ N(0, σ_τ²)
```

推荐：

| 等级     | delay noise |
| ------ | ----------- |
| mild   | 0.1 ns      |
| medium | 0.5 ns      |
| hard   | 1 ns        |
| severe | 2 ns        |

也可以做 delay quantization：

```text
τ_noisy = round(τ / Δτ) * Δτ
```

例如：

```text
Δτ ∈ {0.5 ns, 1 ns, 2 ns}
```

这个非常适合 MMW，因为它的 CSI 是 ray tracing path-level 数据，delay 是干净的几何真值。你加 delay noise 后，CSI 仍有传播结构，但从“完美几何真值”变成“估计不准的真实 CSI”。

---

## 5. AoA / AoD 角度扰动：破坏方向信息

MMW 提供 AoA 和 AoD，包括 local coordinate 和 global coordinate 下的角度信息。([乐量][2]) 如果你的 beam label 和方向强相关，那么角度信息会非常强。为了让 CSI 难学，可以给 AoA/AoD 加角度误差：

```text
θ_noisy = θ + ε_θ
φ_noisy = φ + ε_φ
ε ~ N(0, σ_angle²)
```

推荐：

| 等级     | angle noise |
| ------ | ----------- |
| mild   | 1°          |
| medium | 3°          |
| hard   | 5°          |
| severe | 10°         |

我建议主实验用：

```text
AoA/AoD noise = 3° 或 5°
```

这很容易解释为阵列校准误差、角度估计误差、传感器姿态误差。

---

## 6. 天线阵列退化：mask 部分 Tx/Rx antenna

MMW 的 complex gain 维度里包含 Rx antenna 和 Tx antenna 维度。([乐量][2]) 因此可以做 antenna masking：

```text
a[:, :, masked_rx_ant, :, :, :, :] = 0
a[:, :, :, :, masked_tx_ant, :, :] = 0
```

推荐：

| 阵列规模        | mask 数量    |
| ----------- | ---------- |
| 4 antennas  | mask 1 个   |
| 8 antennas  | mask 1–2 个 |
| 16 antennas | mask 2–4 个 |

这个模拟 **RF chain 失效、阵列部分不可用、硬件增益不一致**。

比起普通 AWGN，antenna masking 更能破坏 CSI 的空间结构，因此更适合作为“难学 CSI”的构造。

---

## 7. 阵列相位误差：每根天线加固定相位偏移

这个比直接加噪更真实。对每个 antenna 加一个 calibration error：

```text
a_noisy[..., ant, ...] = a[..., ant, ...] * exp(j * δ_ant)
δ_ant ~ N(0, σ_phase²)
```

推荐：

| 等级     | phase error |
| ------ | ----------- |
| mild   | 5°          |
| medium | 10°         |
| hard   | 20°         |
| severe | 30°         |

这个会破坏 beamforming 相关信息，但不会完全抹掉 channel。非常适合你要的“上限高但难学”。

---

## 8. 时间错位：CSI 与其他模态不同步

MMW 的一个特点是所有模态按 frame number 同步对齐。([乐量][2]) 这对多模态学习很友好，但真实系统中 CSI、camera、LiDAR、GNSS 不一定严格同步。

你可以故意让 CSI 使用错位帧：

```text
当前样本使用 t 时刻的 image/GNSS/LiDAR
但 CSI 使用 t-1 或 t+1
```

推荐：

| 等级     | shift         |
| ------ | ------------- |
| mild   | ±1 frame      |
| medium | ±2 frames     |
| hard   | 随机 0–3 frames |
| severe | 随机 0–5 frames |

MMW 采样率是 100 Hz。([乐量][1]) 所以 ±1 frame 大约就是 10 ms 的错位。这个很适合做未来 beam prediction，因为它会破坏 CSI 的短期时序一致性。

---

## 9. 子载波域退化：如果你把 path 转成 OFDM CSI

如果你后续把 MMW 的 path-level channel 合成为 OFDM CSI：

```text
H[f] = Σ_m a_m exp(-j 2π f τ_m)
```

那么可以做：

```text
subcarrier dropout
subcarrier band masking
frequency-selective noise
```

例如：

```text
随机 mask 20% subcarriers
或者 mask 连续 8/16 个 subcarriers
```

这会让 CSI 更像真实 OFDM 估计结果，也方便用 CNN/Transformer 编码。

---

# 我最推荐的 5 种 MMW-CSI 退化方式

对于 MMW，我建议优先做这 5 个，而不是所有都做：

| 推荐度  | 方式                              | 为什么适合 MMW                          |
| ---- | ------------------------------- | ---------------------------------- |
| 必做   | complex gain AWGN               | 最基础的信道估计误差                         |
| 必做   | path dropout                    | MMW 是 path-level channel，非常适合做多径缺失 |
| 必做   | AoA/AoD angle noise             | MMW 明确提供角度信息，beam prediction 强相关   |
| 强烈推荐 | antenna phase calibration error | 破坏阵列结构，但保留 CSI 信息                  |
| 强烈推荐 | CSI temporal shift              | MMW 是 100 Hz 且严格同步，故意错位很有意义        |

---

# 建议你设计 3 个 CSI 质量等级

## Clean CSI

```text
不退化
```

用于证明 CSI 的信息上限。

## Medium Degraded CSI

```text
complex gain SNR = 10 dB
path dropout = 20%
AoA/AoD noise = 3°
delay noise = 0.5 ns
antenna phase error = 10°
CSI temporal shift = ±1 frame
```

这是我最推荐的主实验设置。

## Hard Degraded CSI

```text
complex gain SNR = 5 dB
path dropout = 30%
dominant path attenuation γ = 0.5
AoA/AoD noise = 5°
delay noise = 1 ns
antenna phase error = 20°
CSI temporal shift = ±2 frames
```

这个用于鲁棒性实验。

不要一开始就用：

```text
SNR = 0 dB
path dropout = 50%
AoA noise = 10°
dominant path removed
```

这可能会把 CSI 直接变成弱信息模态，而不是“高上限但难学模态”。

---

# 更适合论文叙事的构造方式

你可以把 MMW 里的模态分成两类：

| 类型      | MMW 中的模态                    | 特点                            |
| ------- | --------------------------- | ----------------------------- |
| 易学低维模态  | GNSS、IMU、bounding box、粗粒度位置 | 低维、收敛快、容易支配梯度                 |
| 高上限难学模态 | path-level CSI / raw CSI    | 包含多径、角度、时延、阵列信息，但维度高、噪声敏感、建模难 |

G2D 里对模态失衡的描述正好支持这个逻辑：模态失衡是某些模态主导优化，导致其他模态欠利用；而且这种问题和模态收敛速度差异有关。

所以你的实验可以这样设计：

```text
GNSS-only
Clean CSI-only
Degraded CSI-only
GNSS + Clean CSI
GNSS + Degraded CSI
GNSS + Degraded CSI + CSI-prioritized / G2D-style training
```

你希望观察到：

```text
Clean CSI-only 最终上限高
Degraded CSI-only 前期难学、收敛慢
GNSS-only 收敛快但上限有限
GNSS + Degraded CSI 普通联合训练时，模型偏向 GNSS
CSI-prioritized / distillation 后，Degraded CSI 的贡献被释放
```

---

# 最推荐你使用的最终方案

如果只选一个主方案，我建议：

```text
MMW-CSI-Medium-Degradation:

1. complex gain AWGN: SNR = 10 dB
2. path dropout: 20%，优先丢弱路径
3. AoA/AoD noise: 3°
4. delay noise: 0.5 ns
5. antenna phase error: 10°
6. temporal shift: 随机 {-1, 0, +1} frame
```

这个方案最符合 MMW 的数据结构，因为它直接作用在：

```text
complex gain a
delay τ
AoA / AoD
antenna dimension
frame synchronization
```

也就是说，它不是泛泛地“给 CSI 加噪”，而是把 MMW 的 **ray-tracing perfect path-level channel** 退化成更接近真实系统的 **estimated imperfect CSI**。这更容易支撑你的论文主张：**CSI 本身信息上限高，但由于估计误差、多径缺失、阵列误差和时间错位，在联合训练中会变成高潜力难优化模态，容易被 GNSS/IMU/位置类易学模态压制。**

[1]: https://le-liang.github.io/mmw/ "Multimodal-Wireless Dataset"
[2]: https://le-liang.github.io/mmw/data-collection.html "Data Collection - Multimodal-Wireless"
