可以。你这张图里的加噪方式本质上不是“直接对 CSI 加 AWGN”，而是模拟 **上行导频估计误差**：

[
\bar{\mathbf{r}}_{kl}
=====================

# \frac{1}{\bar{p}N_u}\bar{\mathbf{Y}}_l\bar{\mathbf{s}}_k

\mathbf{h}_{kl}
+
\frac{1}{\bar{p}N_u}\bar{\mathbf{N}}\bar{\mathbf{s}}_k
]

也就是说，模型输入不再是 clean CSI：

```text
h
```

而是 AP 通过 pilot correlation 得到的 noisy channel estimate：

```text
h_hat = h + estimation_noise
```

下面给你一版新的 Codex 实现方案。

---

# Codex 实现任务：Pilot-based Noisy Dual-View CSI Encoder

## 1. 核心改动

原来的 noisy CSI 是：

```text
h_noisy = h + AWGN
```

现在改成论文里的 uplink pilot estimation：

```text
Y = h s^H + N
h_hat = 1 / (p_bar * N_u) * Y s
      = h + 1 / (p_bar * N_u) * N s
```

如果不同用户 pilot 正交，那么其他用户项会被消掉，所以对于目标用户，可以直接使用等价形式：

```text
h_hat = h + e
e ~ CN(0, sigma_p^2 / (p_bar * N_u))
```

其中：

```text
p_bar      = 每个 pilot 的发射功率
N_u        = pilot sequence length
sigma_p^2  = AP 接收端 pilot 噪声方差
```

所以你不需要真的构造完整的 `Y`，直接加等价的 channel estimation noise 就可以。

---

# 2. 输入输出定义

默认 raw CSI 输入：

```python
x_csi: [B, T, Nsc, Nant, 2]
```

含义：

```text
B     = batch size
T     = 历史时间窗口
Nsc   = subcarrier 数量
Nant  = AP/BS antenna 数量
2     = real / imag
```

内部转成 complex：

```python
h: [B, T, Nsc, Nant]
```

经过 pilot-based channel estimation：

```python
h_hat: [B, T, Nsc, Nant]
```

然后进入 dual-view encoder：

```text
h_hat
├── frequency-domain view: h_hat
└── delay-domain view: IFFT(h_hat, dim=subcarrier)
```

最终输出：

```python
z_csi: [B, T, d_model]
```

默认：

```text
d_model = 64
```

---

# 3. Pilot-based CSI 加噪模块

## 3.1 推荐实现：等价估计噪声版本

因为正交 pilot correlation 后有：

[
\hat h = h + e
]

其中：

[
e \sim \mathcal{CN}
\left(
0,
\frac{\sigma_p^2}{\bar p N_u}
\right)
]

所以直接实现：

```python
class PilotCSIChannelEstimator(nn.Module):
    def __init__(
        self,
        pilot_len=16,
        pilot_power=1.0,
        noise_var=None,
        est_snr_db=None,
        train_snr_min_db=None,
        train_snr_max_db=None,
        normalize_by_train_rms=True,
        train_rms=1.0,
        eps=1e-8,
    ):
        super().__init__()
        self.pilot_len = pilot_len
        self.pilot_power = pilot_power
        self.noise_var = noise_var
        self.est_snr_db = est_snr_db
        self.train_snr_min_db = train_snr_min_db
        self.train_snr_max_db = train_snr_max_db
        self.normalize_by_train_rms = normalize_by_train_rms
        self.register_buffer("train_rms", torch.tensor(float(train_rms)))
        self.eps = eps

    def forward(self, h, snr_db=None):
        """
        h: complex tensor [B, T, Nsc, Nant]
        return:
            h_hat: complex tensor [B, T, Nsc, Nant]
            aux: dict
        """

        # 1. fixed dataset-level normalization
        # 不建议在加噪后做 per-sample RMS normalization，
        # 否则会弱化 pilot SNR 的物理含义。
        if self.normalize_by_train_rms:
            h = h / (self.train_rms + self.eps)

        # 2. decide effective estimation SNR
        if self.training and self.train_snr_min_db is not None:
            # random SNR per batch
            low = self.train_snr_min_db
            high = self.train_snr_max_db
            snr_db_tensor = torch.empty(
                h.shape[0], 1, 1, 1,
                device=h.device,
                dtype=h.real.dtype,
            ).uniform_(low, high)
        else:
            if snr_db is not None:
                snr_db_tensor = torch.tensor(
                    float(snr_db),
                    device=h.device,
                    dtype=h.real.dtype,
                )
            elif self.est_snr_db is not None:
                snr_db_tensor = torch.tensor(
                    float(self.est_snr_db),
                    device=h.device,
                    dtype=h.real.dtype,
                )
            else:
                snr_db_tensor = None

        # 3. compute post-correlation estimation noise variance
        if snr_db_tensor is not None:
            # 如果 h 已经用 train_rms 归一化，则平均信道功率约为 1
            # CE-SNR = E|h|^2 / sigma_e^2
            # sigma_e^2 = E|h|^2 / 10^(SNR/10)
            signal_power = h.abs().pow(2).mean(dim=(1, 2, 3), keepdim=True)
            snr_linear = torch.pow(10.0, snr_db_tensor / 10.0)
            sigma_e2 = signal_power / (snr_linear + self.eps)

            # 等价到 pilot noise:
            # sigma_e2 = sigma_p^2 / (pilot_power * pilot_len)
        else:
            # physical mode:
            # sigma_e2 = sigma_p^2 / (p_bar * N_u)
            if self.noise_var is None:
                return h, {
                    "sigma_e2": torch.zeros((), device=h.device),
                    "snr_db": None,
                }

            sigma_e2 = self.noise_var / (
                self.pilot_power * self.pilot_len + self.eps
            )

        # 4. complex Gaussian estimation noise
        noise_real = torch.randn_like(h.real)
        noise_imag = torch.randn_like(h.imag)
        noise = torch.complex(noise_real, noise_imag)

        # CN(0, sigma_e2) means real/imag each has variance sigma_e2 / 2
        noise = noise * torch.sqrt(sigma_e2 / 2.0)

        h_hat = h + noise

        aux = {
            "sigma_e2": sigma_e2.detach(),
            "snr_db": snr_db_tensor.detach() if snr_db_tensor is not None else None,
        }

        return h_hat, aux
```

---

# 4. 两种加噪配置

## 4.1 物理参数模式

如果你知道 pilot 噪声方差：

```yaml
csi_estimation:
  mode: physical
  pilot_len: 16
  pilot_power: 1.0
  noise_var: 0.01
```

对应：

[
\sigma_e^2 = \frac{\sigma_p^2}{\bar p N_u}
]

也就是：

```text
pilot_len 越长，CSI 估计越准
pilot_power 越大，CSI 估计越准
noise_var 越大，CSI 估计越差
```

---

## 4.2 推荐实验模式：estimation SNR 模式

如果你只是想构造 noisy CSI 模态，推荐用这个：

```yaml
csi_estimation:
  mode: est_snr
  pilot_len: 16
  pilot_power: 1.0
  train_snr_min_db: 0
  train_snr_max_db: 30
  test_snr_db: [0, 5, 10, 20, 30]
```

训练时：

```text
SNR ~ Uniform(0, 30) dB
```

测试时分别固定：

```text
0 / 5 / 10 / 20 / 30 dB
```

这样更适合做消融：

```text
clean CSI
noisy CSI, 30 dB
noisy CSI, 20 dB
noisy CSI, 10 dB
noisy CSI, 5 dB
noisy CSI, 0 dB
```

---

# 5. Dual-view CSI Encoder 新版整体结构

```text
raw clean CSI h
→ train_rms normalization
→ pilot-based channel estimation
→ h_hat = h + e
→ frequency-domain view
→ delay-domain view
→ CNN tokenizer for each view
→ symmetric fusion
→ GRU temporal encoder
→ [B, T, 64]
```

注意顺序：

```text
先做 pilot-based estimation noise
再做 frequency/delay dual-view
```

不要写成：

```text
先 IFFT，再分别加噪
```

因为实际是 AP 先通过 pilot 得到 noisy CSI estimate，然后你再对这个 noisy estimate 做频域/时延域特征提取。

---

# 6. Frequency view

```python
def make_frequency_view(h_hat):
    """
    h_hat: complex [B, T, Nsc, Nant]
    return: real tensor [B, T, 2, Nant, Nsc]
    """
    x = torch.stack([h_hat.real, h_hat.imag], dim=2)
    # [B, T, 2, Nsc, Nant]

    x = x.permute(0, 1, 2, 4, 3).contiguous()
    # [B, T, 2, Nant, Nsc]

    return x
```

---

# 7. Delay view

```python
def make_delay_view(h_hat, delay_taps=None):
    """
    h_hat: complex [B, T, Nsc, Nant]
    return: real tensor [B, T, 2, Nant, L_delay]
    """

    h_delay = torch.fft.ifft(h_hat, dim=2, norm="ortho")
    # [B, T, Nsc, Nant]

    if delay_taps is not None:
        h_delay = h_delay[:, :, :delay_taps, :]

    x = torch.stack([h_delay.real, h_delay.imag], dim=2)
    # [B, T, 2, L_delay, Nant]

    x = x.permute(0, 1, 2, 4, 3).contiguous()
    # [B, T, 2, Nant, L_delay]

    return x
```

默认：

```yaml
delay_taps: 32
```

如果你的 `Nsc < 32`，就用：

```python
delay_taps = min(32, Nsc)
```

---

# 8. CNN tokenizer

频域和时延域使用同构分支，参数不共享：

```python
class CSIViewTokenizer(nn.Module):
    def __init__(self, d_model=64, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: [B*T, 2, Nant, Nsc_or_delay]
        return: [B*T, d_model]
        """
        x = self.net(x)
        x = self.proj(x)
        return x
```

---

# 9. 双视角融合

推荐默认使用 symmetric gate：

```python
class SymmetricViewFusion(nn.Module):
    def __init__(self, d_model=64, dropout=0.1):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, z_freq, z_delay):
        """
        z_freq:  [B, T, D]
        z_delay: [B, T, D]
        """

        gate_input = torch.cat(
            [
                z_freq,
                z_delay,
                torch.abs(z_freq - z_delay),
                z_freq * z_delay,
            ],
            dim=-1,
        )

        gate_logits = self.gate(gate_input)
        gate = torch.softmax(gate_logits, dim=-1)

        z = gate[..., 0:1] * z_freq + gate[..., 1:2] * z_delay
        z = self.out(z)

        return z, gate
```

建议同时支持：

```text
mean
concat
symmetric_gate
```

默认：

```yaml
view_fusion: symmetric_gate
```

---

# 10. Temporal encoder

默认用 1 层 GRU：

```python
class CSITemporalGRU(nn.Module):
    def __init__(self, d_model=64, num_layers=1, dropout=0.1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        z, _ = self.gru(x)
        z = self.norm(z)
        return z
```

---

# 11. 完整 Encoder 类

Codex 按下面结构实现：

```python
class PilotDualViewCSIEncoder(nn.Module):
    def __init__(
        self,
        d_model=64,
        delay_taps=32,
        pilot_len=16,
        pilot_power=1.0,
        noise_var=None,
        est_snr_db=None,
        train_snr_min_db=0,
        train_snr_max_db=30,
        train_rms=1.0,
        view_fusion="symmetric_gate",
        temporal_type="gru",
        output_mode="sequence",
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.delay_taps = delay_taps
        self.output_mode = output_mode
        self.view_fusion_type = view_fusion

        self.estimator = PilotCSIChannelEstimator(
            pilot_len=pilot_len,
            pilot_power=pilot_power,
            noise_var=noise_var,
            est_snr_db=est_snr_db,
            train_snr_min_db=train_snr_min_db,
            train_snr_max_db=train_snr_max_db,
            train_rms=train_rms,
            normalize_by_train_rms=True,
        )

        self.freq_tokenizer = CSIViewTokenizer(
            d_model=d_model,
            dropout=dropout,
        )

        self.delay_tokenizer = CSIViewTokenizer(
            d_model=d_model,
            dropout=dropout,
        )

        if view_fusion == "symmetric_gate":
            self.view_fusion = SymmetricViewFusion(
                d_model=d_model,
                dropout=dropout,
            )
        elif view_fusion == "mean":
            self.view_fusion = None
        elif view_fusion == "concat":
            self.view_fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Unknown view_fusion: {view_fusion}")

        self.temporal = CSITemporalGRU(
            d_model=d_model,
            num_layers=1,
            dropout=dropout,
        )

    def to_complex(self, x):
        if torch.is_complex(x):
            return x

        if x.shape[-1] != 2:
            raise ValueError(
                f"Expected real/imag CSI with last dim=2, got {x.shape}"
            )

        return torch.complex(x[..., 0], x[..., 1])

    def make_frequency_view(self, h_hat):
        x = torch.stack([h_hat.real, h_hat.imag], dim=2)
        x = x.permute(0, 1, 2, 4, 3).contiguous()
        return x

    def make_delay_view(self, h_hat):
        B, T, Nsc, Nant = h_hat.shape
        h_delay = torch.fft.ifft(h_hat, dim=2, norm="ortho")

        if self.delay_taps is not None:
            taps = min(self.delay_taps, Nsc)
            h_delay = h_delay[:, :, :taps, :]

        x = torch.stack([h_delay.real, h_delay.imag], dim=2)
        x = x.permute(0, 1, 2, 4, 3).contiguous()
        return x

    def fuse_views(self, z_freq, z_delay):
        if self.view_fusion_type == "mean":
            gate = None
            z = 0.5 * (z_freq + z_delay)
            return z, gate

        if self.view_fusion_type == "concat":
            gate = None
            z = self.view_fusion(torch.cat([z_freq, z_delay], dim=-1))
            return z, gate

        if self.view_fusion_type == "symmetric_gate":
            return self.view_fusion(z_freq, z_delay)

        raise ValueError(f"Unknown view_fusion: {self.view_fusion_type}")

    def forward(self, x, snr_db=None, return_aux=False):
        """
        x:
            [B, T, Nsc, Nant, 2] real/imag
            or complex [B, T, Nsc, Nant]

        return:
            z: [B, T, D] if output_mode == sequence
            z: [B, D]    if output_mode == last
        """

        h = self.to_complex(x)
        B, T, Nsc, Nant = h.shape

        # pilot-based channel estimation
        h_hat, est_aux = self.estimator(h, snr_db=snr_db)

        # dual views
        freq_view = self.make_frequency_view(h_hat)
        delay_view = self.make_delay_view(h_hat)

        _, _, _, _, F_freq = freq_view.shape
        _, _, _, _, F_delay = delay_view.shape

        freq_view = freq_view.reshape(B * T, 2, Nant, F_freq)
        delay_view = delay_view.reshape(B * T, 2, Nant, F_delay)

        # tokenize each view
        z_freq = self.freq_tokenizer(freq_view).reshape(B, T, self.d_model)
        z_delay = self.delay_tokenizer(delay_view).reshape(B, T, self.d_model)

        # view fusion
        z_view, gate = self.fuse_views(z_freq, z_delay)

        # temporal modeling
        z = self.temporal(z_view)

        if self.output_mode == "last":
            z_out = z[:, -1]
        else:
            z_out = z

        if return_aux:
            aux = {
                "h_hat": h_hat,
                "z_freq": z_freq,
                "z_delay": z_delay,
                "view_gate": gate,
                "estimation": est_aux,
            }
            return z_out, aux

        return z_out
```

---

# 12. 推荐配置

```yaml
model:
  csi_encoder:
    name: pilot_dual_view_raw_csi
    d_model: 64
    output_mode: sequence

    pilot_estimation:
      pilot_len: 16
      pilot_power: 1.0
      mode: est_snr
      train_snr_min_db: 0
      train_snr_max_db: 30
      test_snr_db: [0, 5, 10, 20, 30]
      train_rms: ${computed_from_train_set}

    dual_view:
      frequency_view: true
      delay_view: true
      delay_taps: 32
      view_fusion: symmetric_gate

    tokenizer:
      type: cnn
      channels: [16, 32, 64]
      dropout: 0.1

    temporal:
      type: gru
      num_layers: 1
      dropout: 0.1
```

---

# 13. train_rms 计算方式

在训练集上预先计算：

```python
def compute_train_csi_rms(train_loader):
    total_power = 0.0
    total_count = 0

    for batch in train_loader:
        x = batch["csi"]

        if not torch.is_complex(x):
            h = torch.complex(x[..., 0], x[..., 1])
        else:
            h = x

        total_power += h.abs().pow(2).sum().item()
        total_count += h.numel()

    rms = (total_power / total_count) ** 0.5
    return rms
```

保存到 config：

```yaml
train_rms: 0.XXXXX
```

不要每个样本单独 RMS normalize 后再加噪，否则 pilot SNR 的意义会变弱。推荐：

```text
clean CSI
→ 用训练集全局 RMS 归一化
→ pilot-based estimation noise
→ dual-view encoder
```

---

# 14. 标签生成建议

你的标签仍然应该用 clean channel / clean beam power 生成：

```text
label = argmax clean received beam power
```

输入使用：

```text
h_hat = h + pilot estimation noise
```

这样实验含义最清楚：

```text
noisy CSI 是由上行导频估计误差导致的弱化模态；
标签代表理想情况下未来最优 beam；
模型需要从 noisy CSI 中恢复对 beam prediction 有用的信息。
```

不要用 noisy CSI 重新生成 label。否则 label 自身也会被噪声污染，任务含义会变成“预测 noisy observation 的最大 beam”，不利于论文叙事。

---

# 15. 必做消融

至少跑这些：

```text
A. clean CSI, frequency only
B. clean CSI, delay only
C. clean CSI, frequency + delay
D. pilot noisy CSI, frequency only
E. pilot noisy CSI, delay only
F. pilot noisy CSI, frequency + delay
```

SNR 消融：

```text
0 dB
5 dB
10 dB
20 dB
30 dB
clean
```

pilot 参数消融：

```text
N_u = 4, 8, 16, 32
p_bar = 0.25, 0.5, 1.0, 2.0
```

因为：

[
\sigma_e^2 = \frac{\sigma_p^2}{\bar p N_u}
]

所以你可以验证：

```text
pilot_len 越大 → CSI 估计更准 → beam prediction 更好
pilot_power 越大 → CSI 估计更准 → beam prediction 更好
```

---

# 16. 和中期融合对接

其他模态如果是：

```python
z_image  # [B, T, 64]
z_lidar  # [B, T, 64]
z_radar  # [B, T, 64]
z_gps    # [B, T, 64]
```

那么 CSI 输出：

```python
z_csi = csi_encoder(x_csi)  # [B, T, 64]
```

融合输入：

```python
features = {
    "image": z_image,
    "lidar": z_lidar,
    "radar": z_radar,
    "gps": z_gps,
    "csi": z_csi,
}
```

CSI encoder 只负责：

```text
raw CSI → pilot noisy CSI estimate → CSI feature
```

不要在 CSI encoder 内部做跨模态融合。

---

# 17. 最终建议版本

你现在最适合实现：

```text
Raw clean CSI
→ train-set global RMS normalization
→ pilot-based channel estimation:
      h_hat = h + e,
      e ~ CN(0, sigma_p^2 / (p_bar * N_u))
→ frequency-domain view: real/imag(h_hat)
→ delay-domain view: real/imag(IFFT(h_hat))
→ two identical CNN tokenizers
→ symmetric gated view fusion
→ 1-layer GRU
→ [B, T, 64]
→ 中期融合
```

这版比普通 AWGN 更符合通信系统叙事，因为 noisy CSI 不是凭空加噪，而是来自 **uplink pilot estimation error**。你的论文可以写成：

```text
We model the noisy CSI modality as an uplink pilot-based channel estimate in a TDD system. 
With orthogonal pilot sequences, the AP isolates the target user's channel by correlating 
the received pilot matrix with the known pilot sequence. The resulting CSI estimate is 
h_hat = h + e, where e follows a complex Gaussian distribution with variance 
sigma_p^2 / (p_bar N_u). This noisy CSI estimate is then encoded through a dual-view 
frequency-delay encoder and fused with other sensing modalities.
```
