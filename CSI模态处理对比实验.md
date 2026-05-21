可以。你现在不要再单点调参了，建议做一个**控制变量矩阵**：一组只改 CSI 预处理，一组只改 CSI 编码器，一组做组合，然后用同一个分析脚本判断哪种最接近目标曲线：

```text
目标曲线：
CSI final ceiling 基本不掉
CSI E50 / E80 / E90 明显变慢
与 easy modality 融合时，普通 joint 更容易被 easy modality 主导
加入 G2D/SMP/CSI-prioritized 后，CSI 贡献恢复
```

G2D 的依据也正好是这个逻辑：它把模态失衡定义为某些模态主导优化、其他模态被欠利用，并指出这种现象与不同模态收敛速度差异有关；它还用 unimodal teacher confidence 判断哪个模态在训练中更 dominant，再用 SMP 给低置信模态专门训练阶段。

下面这份可以直接发给 Codex。

---

# Codex 任务：构建 CSI hard-to-learn 控制变量实验矩阵

## 目标

当前项目 CSI encoder 为：

```text
csi_batch [B,T,Nsc,Nant,2]
→ _as_complex_csi
→ train_rms normalization
→ PilotCSIChannelEstimator
→ frequency view / delay view
→ CSIViewTokenizer
→ view fusion
→ internal GRU
→ modular_sequence outer GRU
→ beam_head
```

现在要构建一批控制变量实验，目标是找到一种 CSI 设置，使其满足：

```text
1. CSI-only final accuracy / ADBA 与 clean CSI teacher 接近
2. CSI-only 收敛速度明显变慢
3. GPS / image / lidar 等 easy modality 前期更快达到自身上限
4. GPS+CSI 或 easy+CSI joint training 中，CSI 容易被压制
5. CSI-prioritized / G2D-style training 能恢复 CSI 贡献
```

核心思想：不要使用会明显破坏信息上限的 destructive degradation，而是增加 information-preserving hardening。

---

# 一、实验分组

总共做 4 组：

```text
Group A: CSI-only upper bound 与 destructive negative control
Group B: 只改 CSI 预处理 hardening
Group C: 只改 CSI encoder 架构
Group D: 预处理 + 架构组合
Group E: easy modality + CSI 多模态验证
```

所有实验固定：

```text
dataset: rainy_MMW_Town10_skybridge_seed24_l1p3
task: beam prediction
epochs: 100
seeds: [24, 25, 26]  # 时间紧可以先只跑 24
metric:
  - beam/accuracy_val
  - beam/val_adba
  - top-k if available
  - train/val loss
  - E50/E80/E90
  - final_mean_last10
  - best_val
```

---

# 二、判定指标

每个 run 训练结束后，统一计算：

```python
final_acc = mean(val_acc over last 10 epochs)
best_acc = max(val_acc)
final_adba = mean(val_adba over last 10 epochs)
best_adba = max(val_adba)

E50 = first epoch where val_acc >= 0.50 * final_acc
E80 = first epoch where val_acc >= 0.80 * final_acc
E90 = first epoch where val_acc >= 0.90 * final_acc

ceiling_gap = final_acc_clean_teacher - final_acc_variant
speed_ratio_E90 = E90_variant / E90_clean_teacher
```

目标筛选条件：

```text
ceiling_gap <= 0.02 ~ 0.03
speed_ratio_E90 >= 1.5
best_acc 不明显低于 clean teacher
val_adba gap <= 0.02
```

如果某个设置：

```text
final_acc 下降 > 0.05
```

则判定为 destructive，不作为主实验，只作为 robustness / negative control。

---

# 三、Group A：基准组

这组用于确认 clean CSI 的上限，以及证明 destructive degradation 会降上限。

## A0：clean full teacher

```yaml
name: csi_A0_clean_full_teacher

modalities: [csi]

dataset:
  csi_degradation:
    enabled: false
  csi_hardening:
    enabled: false

model:
  csi_encoder:
    type: pilot_dual_view_csi
    d_model: 64
    delay_taps: 32
    view_fusion: symmetric_gate
    use_internal_gru: true
    view_gate_warmup_epochs: 0

    pilot_estimator:
      enabled: false

representation_core:
  type: single_gru
```

目的：作为 CSI ceiling。

---

## A1：clean + pilot noise only

```yaml
name: csi_A1_clean_pilot_mild

modalities: [csi]

dataset:
  csi_degradation:
    enabled: false
  csi_hardening:
    enabled: false

model:
  csi_encoder:
    type: pilot_dual_view_csi
    d_model: 64
    delay_taps: 32
    view_fusion: symmetric_gate
    use_internal_gru: true
    view_gate_warmup_epochs: 0

    pilot_estimator:
      enabled: true
      mode: est_snr
      train_snr_min_db: 25
      train_snr_max_db: 35
      eval_snr_db: null

representation_core:
  type: single_gru
```

目的：看 mild pilot estimation 是否只变慢、不降上限。

---

## A2：destructive degradation negative control

```yaml
name: csi_A2_destructive_medium

modalities: [csi]

dataset:
  csi_hardening:
    enabled: false
  csi_degradation:
    enabled: true
    awgn_snr_db: 10
    subcarrier_dropout: 0.2
    antenna_dropout: 0.1
    temporal_dropout: 0.1

model:
  csi_encoder:
    type: pilot_dual_view_csi
    d_model: 64
    delay_taps: 32
    view_fusion: symmetric_gate
    use_internal_gru: true
```

目的：作为反例。预期它会降低上限，不作为主方法。

---

# 四、Group B：只改 CSI 预处理 hardening

这组不改模型架构，只看哪些预处理能做到“变慢但上限不掉”。

所有 B 组统一：

```yaml
model:
  csi_encoder:
    type: pilot_dual_view_csi
    d_model: 64
    delay_taps: 32
    view_fusion: symmetric_gate
    use_internal_gru: true
    view_gate_warmup_epochs: 0

    pilot_estimator:
      enabled: false

representation_core:
  type: single_gru
```

---

## B1：common phase only

```yaml
name: csi_B1_common_phase

dataset:
  csi_degradation:
    enabled: false
  csi_hardening:
    enabled: true
    apply_after_rms: true
    common_phase:
      enabled: true
      mode: train_random_eval_fixed_or_off
      phi_range: [-3.1415926, 3.1415926]
    subcarrier_phase_slope:
      enabled: false
    antenna_calibration:
      enabled: false
    antenna_permutation:
      enabled: false
```

说明：全局相位对很多 beam 任务理论上应接近 nuisance，不应明显降上限。

---

## B2：subcarrier phase slope only

```yaml
name: csi_B2_phase_slope

dataset:
  csi_degradation:
    enabled: false
  csi_hardening:
    enabled: true
    apply_after_rms: true
    common_phase:
      enabled: false
    subcarrier_phase_slope:
      enabled: true
      slope_std: 0.01
      mode: train_random_eval_fixed_or_off
    antenna_calibration:
      enabled: false
    antenna_permutation:
      enabled: false
```

说明：模拟轻微 SFO / timing offset。不要太大，先用 0.01。

---

## B3：antenna calibration only

```yaml
name: csi_B3_antenna_calibration

dataset:
  csi_degradation:
    enabled: false
  csi_hardening:
    enabled: true
    apply_after_rms: true
    common_phase:
      enabled: false
    subcarrier_phase_slope:
      enabled: false
    antenna_calibration:
      enabled: true
      gain_range: [0.95, 1.05]
      phase_std_deg: 5.0
      mode: fixed_by_seed
      seed: 24
    antenna_permutation:
      enabled: false
```

说明：这是最像真实硬件误差的版本，预期最稳。

---

## B4：fixed antenna permutation only

```yaml
name: csi_B4_antenna_permutation

dataset:
  csi_degradation:
    enabled: false
  csi_hardening:
    enabled: true
    apply_after_rms: true
    common_phase:
      enabled: false
    subcarrier_phase_slope:
      enabled: false
    antenna_calibration:
      enabled: false
    antenna_permutation:
      enabled: true
      mode: fixed_by_seed
      seed: 24
```

说明：固定天线置换是可逆的，理论信息不丢，但可能破坏 CNN 对天线空间结构的捷径。

---

## B5：mild information-preserving combo

```yaml
name: csi_B5_hardening_mild_combo

dataset:
  csi_degradation:
    enabled: false
  csi_hardening:
    enabled: true
    apply_after_rms: true

    common_phase:
      enabled: true
      phi_range: [-3.1415926, 3.1415926]

    subcarrier_phase_slope:
      enabled: true
      slope_std: 0.005

    antenna_calibration:
      enabled: true
      gain_range: [0.97, 1.03]
      phase_std_deg: 3.0
      mode: fixed_by_seed
      seed: 24

    antenna_permutation:
      enabled: true
      mode: fixed_by_seed
      seed: 24
```

说明：主候选之一。强度较轻，优先保证上限不掉。

---

## B6：medium information-preserving combo

```yaml
name: csi_B6_hardening_medium_combo

dataset:
  csi_degradation:
    enabled: false
  csi_hardening:
    enabled: true
    apply_after_rms: true

    common_phase:
      enabled: true
      phi_range: [-3.1415926, 3.1415926]

    subcarrier_phase_slope:
      enabled: true
      slope_std: 0.01

    antenna_calibration:
      enabled: true
      gain_range: [0.95, 1.05]
      phase_std_deg: 5.0
      mode: fixed_by_seed
      seed: 24

    antenna_permutation:
      enabled: true
      mode: fixed_by_seed
      seed: 24
```

说明：主候选之二。如果 B5 变慢不明显，就看 B6。

---

# 五、Group C：只改 CSI encoder 架构

这组不改输入，只改模型，让 CSI 不那么容易直接走捷径。

所有 C 组统一：

```yaml
dataset:
  csi_degradation:
    enabled: false
  csi_hardening:
    enabled: false
```

---

## C1：view gate warmup

```yaml
name: csi_C1_gate_warmup

model:
  csi_encoder:
    type: pilot_dual_view_csi
    d_model: 64
    delay_taps: 32
    view_fusion: symmetric_gate
    use_internal_gru: true

    view_gate_warmup_epochs: 30
    view_gate_warmup_mode: mean

    pilot_estimator:
      enabled: false

representation_core:
  type: single_gru
```

实现要求：epoch < 30 时，frequency/delay fusion 强制 mean；epoch >= 30 后恢复 symmetric_gate。

---

## C2：no internal GRU

```yaml
name: csi_C2_no_internal_gru

model:
  csi_encoder:
    type: pilot_dual_view_csi
    d_model: 64
    delay_taps: 32
    view_fusion: symmetric_gate
    use_internal_gru: false
    view_gate_warmup_epochs: 0

    pilot_estimator:
      enabled: false

representation_core:
  type: single_gru
```

目的：去掉 CSI 的双重时间建模，只保留外层 `single_gru`。

---

## C3：delay view warmup

```yaml
name: csi_C3_delay_warmup

model:
  csi_encoder:
    type: pilot_dual_view_csi
    d_model: 64
    delay_taps: 32
    view_fusion: symmetric_gate
    use_internal_gru: true

    delay_view_warmup_epochs: 30
    delay_view_warmup_mode: freq_only

    pilot_estimator:
      enabled: false

representation_core:
  type: single_gru
```

实现要求：epoch < 30 时只用 frequency view；epoch >= 30 时启用 frequency + delay dual-view。

---

## C4：weaker tokenizer

```yaml
name: csi_C4_weaker_tokenizer

model:
  csi_encoder:
    type: pilot_dual_view_csi
    d_model: 64
    delay_taps: 32
    view_fusion: symmetric_gate
    use_internal_gru: true

    tokenizer:
      hidden_channels: 16     # default 如果是 32/64，则降一档
      dropout: 0.2
      use_second_conv: true

    pilot_estimator:
      enabled: false

representation_core:
  type: single_gru
```

目的：只降低前端 CNN 容量，看是否变慢但上限不明显掉。

---

## C5：raw frequency only

```yaml
name: csi_C5_freq_only

model:
  csi_encoder:
    type: pilot_dual_view_csi
    d_model: 64
    delay_taps: 32
    view_fusion: freq_only
    use_internal_gru: true

    pilot_estimator:
      enabled: false

representation_core:
  type: single_gru
```

说明：这个可能会降上限。如果上限掉得多，不作为主方案。

---

# 六、Group D：预处理 + 架构组合

这组是主候选。只在 B、C 中比较稳的设置上组合。

---

## D1：mild hardening + gate warmup

```yaml
name: csi_D1_mild_hardening_gate_warmup

dataset:
  csi_hardening: ${B5_hardening_mild_combo}
  csi_degradation:
    enabled: false

model:
  csi_encoder:
    type: pilot_dual_view_csi
    d_model: 64
    delay_taps: 32
    view_fusion: symmetric_gate
    use_internal_gru: true
    view_gate_warmup_epochs: 30
    view_gate_warmup_mode: mean

    pilot_estimator:
      enabled: false

representation_core:
  type: single_gru
```

---

## D2：mild hardening + no internal GRU

```yaml
name: csi_D2_mild_hardening_no_internal_gru

dataset:
  csi_hardening: ${B5_hardening_mild_combo}
  csi_degradation:
    enabled: false

model:
  csi_encoder:
    type: pilot_dual_view_csi
    d_model: 64
    delay_taps: 32
    view_fusion: symmetric_gate
    use_internal_gru: false

    pilot_estimator:
      enabled: false

representation_core:
  type: single_gru
```

---

## D3：mild hardening + gate warmup + no internal GRU

```yaml
name: csi_D3_mild_hardening_gate_warmup_no_internal_gru

dataset:
  csi_hardening: ${B5_hardening_mild_combo}
  csi_degradation:
    enabled: false

model:
  csi_encoder:
    type: pilot_dual_view_csi
    d_model: 64
    delay_taps: 32
    view_fusion: symmetric_gate
    use_internal_gru: false
    view_gate_warmup_epochs: 30
    view_gate_warmup_mode: mean

    pilot_estimator:
      enabled: false

representation_core:
  type: single_gru
```

这是我最推荐的主候选。

---

## D4：medium hardening + gate warmup + no internal GRU

```yaml
name: csi_D4_medium_hardening_gate_warmup_no_internal_gru

dataset:
  csi_hardening: ${B6_hardening_medium_combo}
  csi_degradation:
    enabled: false

model:
  csi_encoder:
    type: pilot_dual_view_csi
    d_model: 64
    delay_taps: 32
    view_fusion: symmetric_gate
    use_internal_gru: false
    view_gate_warmup_epochs: 30
    view_gate_warmup_mode: mean

    pilot_estimator:
      enabled: false

representation_core:
  type: single_gru
```

如果 D3 变慢不明显，再用 D4。

---

## D5：mild hardening + delay warmup + no internal GRU

```yaml
name: csi_D5_mild_hardening_delay_warmup_no_internal_gru

dataset:
  csi_hardening: ${B5_hardening_mild_combo}
  csi_degradation:
    enabled: false

model:
  csi_encoder:
    type: pilot_dual_view_csi
    d_model: 64
    delay_taps: 32
    view_fusion: symmetric_gate
    use_internal_gru: false

    delay_view_warmup_epochs: 30
    delay_view_warmup_mode: freq_only

    pilot_estimator:
      enabled: false

representation_core:
  type: single_gru
```

---

# 七、Group E：多模态验证组

先从 Group D 里选出 1–2 个满足：

```text
ceiling_gap <= 0.03
E90_ratio >= 1.5
```

的 CSI slow variant，然后做多模态验证。

假设选中的是：

```text
csi_slow = D3
```

则跑：

---

## E0：easy modality only

建议先用 GPS。如果 MMW 里 GPS/GNSS 很强，就用 GPS；如果没有，就用 image/lidar 中收敛最快的那个。

```yaml
name: E0_gps_only

modalities: [gps]

model:
  type: modular_sequence
  representation_core:
    type: single_gru
```

---

## E1：clean CSI + GPS joint

```yaml
name: E1_gps_clean_csi_joint

modalities: [gps, csi]

dataset:
  csi_hardening:
    enabled: false
  csi_degradation:
    enabled: false

model:
  type: modular_sequence
  csi_encoder: ${A0_clean_full_teacher_csi_encoder}
  representation_core:
    type: early_concat_gru
```

---

## E2：slow CSI + GPS joint

```yaml
name: E2_gps_slow_csi_joint

modalities: [gps, csi]

dataset:
  csi_hardening: ${D3_or_selected}
  csi_degradation:
    enabled: false

model:
  type: modular_sequence
  csi_encoder: ${D3_or_selected_csi_encoder}
  representation_core:
    type: early_concat_gru
```

目标：观察普通 joint 是否更偏向 GPS。

---

## E3：slow CSI + GPS + CSI-prioritized warmup

```yaml
name: E3_gps_slow_csi_csi_prioritized

modalities: [gps, csi]

training:
  modality_priority:
    enabled: true
    schedule:
      - epochs: [0, 30]
        train_modalities: [csi]
        freeze_modalities: [gps]
        train_fusion: true
      - epochs: [30, 100]
        train_modalities: [gps, csi]
        freeze_modalities: []
        train_fusion: true

dataset:
  csi_hardening: ${D3_or_selected}
  csi_degradation:
    enabled: false

model:
  type: modular_sequence
  csi_encoder: ${D3_or_selected_csi_encoder}
  representation_core:
    type: early_concat_gru
```

这对应 G2D/SMP 的思想：让低置信或慢学模态先获得不受干扰的训练阶段。G2D 里 SMP 会按 teacher confidence 排序，并在不同 epoch 范围只更新被优先的模态 encoder，之后再联合训练。

---

## E4：slow CSI + GPS + G2D-style distillation

```yaml
name: E4_gps_slow_csi_g2d_style

modalities: [gps, csi]

teachers:
  csi_teacher_ckpt: path/to/A0_clean_full_teacher/best.ckpt
  gps_teacher_ckpt: path/to/E0_gps_only/best.ckpt

training:
  g2d_style:
    enabled: true
    feature_distill:
      enabled: true
      alpha: 0.1
    logit_distill:
      enabled: true
      beta: 0.5
      temperature: 2.0
    confidence_score:
      enabled: true
      source: teacher_gt_prob
    smp:
      enabled: true
      schedule: confidence_rank
      warmup_epochs_per_weak_modality: 30
```

G2D 的 loss 同时包含 supervised student loss、feature distillation 和 logit distillation，用 unimodal teacher 约束 multimodal student 不要丢掉单模态表示。

---

# 八、需要实现的代码改动

## 1. 新增 `csi_hardening`

在 dataset 或 CSI encoder 前处理处实现。建议位置：

```text
_as_complex_csi
→ train_rms normalization
→ apply_csi_hardening
→ PilotCSIChannelEstimator
```

新增文件可放：

```text
src/kd_sensing/data/csi_hardening.py
```

或：

```text
src/kd_sensing/models/csi_hardening.py
```

核心函数：

```python
def apply_csi_hardening(
    h: torch.Tensor,
    cfg: dict,
    training: bool,
    global_step: int | None = None,
) -> torch.Tensor:
    """
    h: complex tensor [B,T,Nsc,Nant]
    return: complex tensor [B,T,Nsc,Nant]
    """
```

实现模块：

```python
apply_common_phase(h, phi_range)
apply_subcarrier_phase_slope(h, slope_std)
apply_antenna_calibration(h, gain_range, phase_std_deg, mode, seed)
apply_antenna_permutation(h, mode, seed)
```

注意：

```text
common_phase 可以 train random，eval off 或 eval fixed
antenna_calibration 建议 fixed_by_seed
antenna_permutation 必须 fixed_by_seed
不要每个 batch 重新随机 permutation，否则会变成信息破坏
```

---

## 2. CSI encoder 支持 `use_internal_gru`

在 `src/kd_sensing/models/csi.py` 中：

```python
if cfg.use_internal_gru:
    x, _ = self.gru(x)
else:
    x = x
```

同时保证输出 shape 仍然是：

```text
[B,T,D]
```

---

## 3. CSI encoder 支持 `view_gate_warmup_epochs`

增加方法：

```python
def set_epoch(self, epoch: int):
    self.current_epoch = epoch
```

在 trainer 每个 epoch 开始时调用。

fusion 逻辑：

```python
if self.current_epoch < view_gate_warmup_epochs:
    fused = 0.5 * freq_feat + 0.5 * delay_feat
    aux["view_gate"] = torch.full([B,T,2], 0.5)
else:
    fused = symmetric_gate(freq_feat, delay_feat)
```

---

## 4. CSI encoder 支持 `delay_view_warmup_epochs`

逻辑：

```python
if self.current_epoch < delay_view_warmup_epochs:
    delay_feat = torch.zeros_like(freq_feat)
    fused = freq_feat
else:
    use normal dual-view fusion
```

或者：

```python
if warmup_mode == "freq_only":
    fused = freq_feat
```

---

## 5. 统一日志

每个 run 额外记录：

```text
beam/accuracy_val
beam/val_adba
beam/loss_train
beam/loss_val

csi/view_gate_freq_mean
csi/view_gate_delay_mean
csi/input_power_mean
csi/input_phase_std
csi/hardening_enabled

optimization/grad_norm_csi_encoder
optimization/grad_norm_gps_encoder
optimization/grad_norm_fusion
optimization/grad_cos_csi_gps  # 可选
```

多模态 run 还要记录：

```text
modality_confidence/csi_gt_prob
modality_confidence/gps_gt_prob
modality_confidence/csi_entropy
modality_confidence/gps_entropy
```

---

# 九、统一分析脚本

新增：

```text
scripts/analyze_csi_hardening_sweep.py
```

输入：

```bash
python scripts/analyze_csi_hardening_sweep.py \
  --runs_root outputs/rainy_MMW_Town10_skybridge_seed24_l1p3 \
  --pattern "csi_*" \
  --clean_teacher_run csi_A0_clean_full_teacher \
  --out outputs/csi_hardening_analysis
```

输出：

```text
summary.csv
ranked_candidates.csv
learning_curves.png
ceiling_gap_vs_E90_ratio.png
multimodal_imbalance_summary.csv
```

`summary.csv` 字段：

```text
run_name
group
seed
best_acc
final_acc_last10
best_adba
final_adba_last10
E50
E80
E90
ceiling_gap_acc
ceiling_gap_adba
E90_ratio
is_destructive
is_slow_high_ceiling
recommendation
```

筛选逻辑：

```python
is_destructive = ceiling_gap_acc > 0.05
is_slow_high_ceiling = ceiling_gap_acc <= 0.03 and E90_ratio >= 1.5
```

排序：

```python
score = (
    2.0 * E90_ratio
    - 50.0 * max(0, ceiling_gap_acc)
    - 20.0 * max(0, ceiling_gap_adba)
)
```

---

# 十、推荐优先运行顺序

不要一口气跑完所有。按这个顺序：

```text
第一批：
A0, A1, A2
B3, B4, B5, B6
C1, C2

第二批：
D1, D2, D3, D4

第三批：
E0, E1, E2, E3

第四批：
E4
```

第一批的目标是找到：

```text
上限不掉但 E90 变大的 CSI-only 设置
```

第二批的目标是找到：

```text
最佳 slow-high-ceiling CSI
```

第三批才验证：

```text
easy modality 是否压制 slow CSI
```

---

# 十一、最终你应该优先看这几个对比

最重要的是这 6 个：

```text
A0 clean full teacher
B5 mild hardening
B6 medium hardening
C2 no internal GRU
D3 mild hardening + gate warmup + no internal GRU
A2 destructive degradation
```

如果结果是：

```text
A0 final = 0.95, E90 = 10
D3 final = 0.93~0.95, E90 = 25~40
A2 final = 0.85, E90 = 35
```

那说明：

```text
D3 是你要的 high-ceiling slow-learning CSI
A2 只是 destructive degradation
```

然后再做：

```text
GPS-only
GPS + A0 clean CSI
GPS + D3 slow CSI
GPS + D3 slow CSI + CSI-prioritized
GPS + D3 slow CSI + G2D-style
```

这条结果线最容易写论文。
