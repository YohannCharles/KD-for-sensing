下面是**修正版 Codex 实现方案**。这一版已经按你的新设定重写：

```text
num_pred = 3
label = [t+1, t+2, t+3]
不再包含 beam8 / 当前最后历史 beam
logits shape = [B, 3, 64]
labels shape = [B, 3]
```

G2D 的核心仍然是：多个单模态 teacher 指导 multimodal student，通过 supervised loss、feature distillation、logit distillation，以及基于 teacher confidence 的 Sequential Modality Prioritization 缓解强模态压制弱模态的问题。

---

```text
任务：在 KD-for-sensing 中实现适配未来 beam prediction 的 G2D-lite / G2D-global / G2D-horizon baseline

重要标签约定：
当前项目的预测标签已经修改为只包含未来 3 个 beam：

label = [t+1, t+2, t+3]

因此：
num_pred = 3
H = 3
C = 64
student logits shape = [B, 3, 64]
teacher logits shape = [B, 3, 64]
labels shape = [B, 3]

horizon index 含义：
h=0 -> t+1
h=1 -> t+2
h=2 -> t+3

注意：
不再存在 h0=beam8。
不要再把当前最后一个历史 beam 纳入训练 label、loss、metric 或 diagnostics。
```

---

# 1. 实现目标

```text
目标：
1. 实现 G2D-lite：
   supervised CE + feature KD + logit KD。

2. 实现 G2D-global：
   在 G2D-lite 基础上加入全局 SMP 梯度调制。

3. 实现 G2D-horizon：
   输出 horizon-wise teacher confidence、modality ranking 和 imbalance diagnostics。
   训练调制先用 future 3 步平均 confidence，不做复杂的 per-horizon backward。

4. 输出适合通信多模态失衡分析的诊断结果：
   - 每个模态在 t+1/t+2/t+3 上的 teacher confidence
   - modality weak-to-strong ranking
   - student branch confidence
   - confidence ratio
   - SMP active modalities
   - horizon-wise top1/top3/top5

5. 不破坏现有 CRAF/MARF。
   G2D 作为通用多模态失衡 baseline，MARF 仍作为主方法。
```

---

# 2. 实现前检查

```text
请先只读检查以下文件：

openspec/specs/
src/kd_sensing/modalities.py
src/kd_sensing/registries.py
src/kd_sensing/engine/batch.py
src/kd_sensing/engine/model_output.py
src/kd_sensing/engine/trainer.py
src/kd_sensing/engine/validator.py
src/kd_sensing/evaluation/metrics.py
src/kd_sensing/models/fusion/
src/kd_sensing/distillers/
src/kd_sensing/losses/
configs/
tests/

重点确认：
1. batch.py 中 labels 是否已经变为 [B,3]。
2. model_output.py 是否假设 horizon=num_pred+1。
3. validator.py / metrics.py 是否仍然输出旧的 h0/beam8 指标。
4. CRAF/MARF 是否仍然输出 [B,4,64]。
5. 单模态 teacher 是否也已经输出 [B,3,64]。
6. 所有旧的 num_pred+1 逻辑都需要统一改成 num_pred。
```

---

# 3. 必须统一的 shape 约定

所有模型输出统一为：

```text
logits: [B, H, C]
H = num_pred = 3
C = 64
```

所有 label 统一为：

```text
labels: [B, H]
H = 3
labels[:, 0] = t+1 beam
labels[:, 1] = t+2 beam
labels[:, 2] = t+3 beam
```

命名建议：

```text
horizon_names = ["t+1", "t+2", "t+3"]
```

不要再使用：

```text
h0 = current
beam8
num_pred + 1
future_only
```

因为现在全部标签都是 future labels。

---

# 4. 新增配置文件

新增：

```text
configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml
configs/fusion/image_radar_gps_lidar_mmwave_g2d_global.yaml
configs/fusion/image_radar_gps_lidar_mmwave_g2d_horizon.yaml
```

推荐配置结构：

```yaml
model:
  type: fusion_student
  modalities: [image, radar, gps, lidar, mmwave]
  num_classes: 64
  num_pred: 3

distiller:
  type: g2d
  enabled: true

  teachers:
    image:
      model_type: image_teacher
      checkpoint: null
      strict_load: true
    radar:
      model_type: radar_teacher
      checkpoint: null
      strict_load: true
    gps:
      model_type: gps_teacher
      checkpoint: null
      strict_load: true
    lidar:
      model_type: lidar_teacher
      checkpoint: null
      strict_load: true
    mmwave:
      model_type: mmwave_teacher
      checkpoint: null
      strict_load: true

  loss:
    supervised_weight: 1.0
    feature_weight: 0.1
    logit_weight: 0.5
    temperature: 4.0

    # 新标签已经全是未来步，因此默认使用 all。
    horizons: all
    # all 等价于 [0, 1, 2]
    # h=0 -> t+1
    # h=1 -> t+2
    # h=2 -> t+3

    feature_align:
      enabled: true
      mode: mse
      pool: last
      normalize: true
      projection: auto

    logit_align:
      enabled: true
      mode: kl_teacher_to_student
      detach_teacher: true

  smp:
    enabled: false
    mode: none
    ranking_source: teacher_confidence
    warmup_epochs: 0

    # 可选：global / horizon
    # g2d_lite: enabled=false
    # g2d_global: enabled=true, mode=global
    # g2d_horizon: enabled=true, mode=horizon_diagnostic

    tau:
      per_modality: 5
      joint: 30

    prioritize_low_confidence_first: true
    suppression: complete

  diagnostics:
    enabled: true
    save_every_epoch: true
    save_teacher_confidence: true
    save_student_branch_confidence: true
    save_confidence_ratio: true
    save_modality_ranking: true
    save_horizon_metrics: true
```

---

# 5. 新增模块

建议新增：

```text
src/kd_sensing/distillers/g2d.py
src/kd_sensing/distillers/teacher_ensemble.py
src/kd_sensing/losses/g2d.py
src/kd_sensing/diagnostics/g2d_diagnostics.py

tests/test_g2d_loss.py
tests/test_g2d_distiller.py
tests/test_g2d_smp.py
tests/test_g2d_diagnostics.py
```

注册：

```text
src/kd_sensing/registries.py

新增：
distiller type = "g2d"
loss type = "g2d_loss"
```

---

# 6. TeacherEnsemble 设计

```python
class TeacherEnsemble(nn.Module):
    def __init__(self, teacher_cfgs, modality_order):
        super().__init__()
        ...

    @torch.no_grad()
    def forward(self, batch):
        """
        Returns:
            {
              "image": NormalizedModelOutput,
              "radar": NormalizedModelOutput,
              "gps": NormalizedModelOutput,
              "lidar": NormalizedModelOutput,
              "mmwave": NormalizedModelOutput,
            }

        每个 output 至少需要：
            logits: [B, 3, 64]
            features: Tensor 或 dict
        """
```

要求：

```text
1. teacher 全部 eval()。
2. teacher 参数 requires_grad=False。
3. teacher forward 使用 torch.no_grad()。
4. checkpoint strict_load 默认 true。
5. 如果 teacher checkpoint 缺失，且 distiller.enabled=true，必须报错。
6. 不允许 teacher 输出 [B,4,64] 时静默通过。
7. 如果发现 teacher logits horizon != num_pred，应抛出清晰错误：
   Expected teacher logits horizon=3, got H=...
```

---

# 7. G2D loss 实现

总损失：

```text
L_total = λ_sup * L_sup + α * L_feat + β * L_logit
```

其中：

```text
L_sup:
student fused logits 对 labels 的 CE。

L_feat:
student 每个模态 encoder feature 对齐对应 unimodal teacher feature。

L_logit:
student fused logits 对齐每个 unimodal teacher logits。
```

---

## 7.1 Supervised CE

输入：

```text
student_logits: [B, 3, 64]
labels: [B, 3]
```

实现：

```python
B, H, C = student_logits.shape
assert H == 3

loss_sup = F.cross_entropy(
    student_logits.reshape(B * H, C),
    labels.reshape(B * H),
)
```

如果配置 `horizons` 不是 all，则支持：

```yaml
horizons: [0, 1, 2]
```

但默认就是 all。

---

## 7.2 Logit KD

每个 teacher：

```text
teacher_logits[m]: [B, 3, 64]
student_logits: [B, 3, 64]
```

实现：

```python
T = temperature

teacher_prob = F.softmax(teacher_logits / T, dim=-1)
student_log_prob = F.log_softmax(student_logits / T, dim=-1)

loss_kl = F.kl_div(
    student_log_prob,
    teacher_prob,
    reduction="batchmean",
) * (T * T)
```

对 5 个模态求平均：

```python
loss_logit = mean([
    KL(student_logits, teacher_logits[m])
    for m in modalities
])
```

注意：

```text
teacher logits 必须 detach。
teacher 不参与梯度更新。
```

---

## 7.3 Feature KD

实现通用 feature 提取函数：

```python
def extract_modality_feature(output, modality: str, pool: str = "last"):
    """
    支持：
    1. output.features 是 Tensor [B,T,D]
    2. output.features 是 Tensor [B,D]
    3. output.features 是 dict[modality] -> Tensor
    4. output.input_features / output.output_features / output.modality_features
    """
```

规则：

```text
如果 feature shape 是 [B,T,D]：
  pool=last -> feature[:, -1, :]
  pool=mean -> feature.mean(dim=1)

如果 feature shape 是 [B,D]：
  直接使用。

如果 student feature dim != teacher feature dim：
  projection=auto 时，为每个 modality 创建 Linear(student_dim -> teacher_dim)。
```

feature loss：

```python
if normalize:
    s_feat = F.normalize(s_feat, dim=-1)
    t_feat = F.normalize(t_feat.detach(), dim=-1)

loss_feat_m = F.mse_loss(s_feat, t_feat)
```

总 feature loss：

```python
loss_feat = mean(loss_feat_m for m in modalities)
```

---

# 8. Teacher confidence 计算

由于标签已经是：

```text
[t+1, t+2, t+3]
```

所以 confidence 直接是：

```python
confidence[m, h] = mean(
    softmax(teacher_logits[m][:, h, :], dim=-1)[labels[:, h]]
)
```

其中：

```text
h=0 -> t+1
h=1 -> t+2
h=2 -> t+3
```

输出：

```text
teacher_confidence: [M, 3]
```

同时保存：

```text
teacher_confidence_t1
teacher_confidence_t2
teacher_confidence_t3
teacher_confidence_avg
```

不要再叫：

```text
future_avg
```

因为现在全部都是 future。可以叫：

```text
teacher_confidence_avg_pred
```

或者：

```text
teacher_confidence_avg_horizon
```

推荐 json 字段：

```json
{
  "teacher_confidence": {
    "image": {
      "t+1": 0.12,
      "t+2": 0.10,
      "t+3": 0.08,
      "avg": 0.10
    }
  }
}
```

---

# 9. SMP 实现

G2D-global 使用每个模态 3 个 horizon 的平均 confidence 排序：

```python
score[m] = mean(confidence[m, 0:3])
```

排序：

```text
低 confidence -> 高 confidence
```

也就是：

```text
weak modality first
strong modality last
```

SMP scheduler：

```python
class SMPScheduler:
    def __init__(
        self,
        modalities,
        per_modality_tau=5,
        joint_tau=30,
        prioritize_low_confidence_first=True,
    ):
        ...

    def rank_modalities(self, confidence_avg):
        """
        confidence_avg:
            dict[str, float]

        return:
            weak_to_strong modality list
        """

    def active_modalities(self, epoch, confidence_avg):
        """
        epoch 0 ~ tau-1:
            weakest modality

        next tau:
            second weakest modality

        ...

        final:
            all modalities
        """
```

梯度调制：

```python
loss.backward()

active_modalities = smp_scheduler.active_modalities(epoch, confidence_avg)

apply_smp_gradient_mask(
    model=model,
    active_modalities=active_modalities,
    keep_fusion_head=True,
)

optimizer.step()
```

关键规则：

```text
1. active modality encoder 保留梯度。
2. inactive modality encoders 梯度清零。
3. fusion module / prediction head 保留梯度。
4. teacher 永远不更新。
5. 不建议每个 iteration 切 requires_grad，直接 backward 后清零 inactive modality encoder grads。
```

实现：

```python
def apply_smp_gradient_mask(model, active_modalities):
    for modality in all_modalities:
        if modality not in active_modalities:
            params = get_modality_parameters(model, modality)
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
```

---

# 10. G2D-horizon 诊断

G2D-horizon 先不做真正 per-horizon gradient masking，只做诊断。

保存每个 horizon 的 modality ranking：

```text
ranking_t+1:
  modalities sorted by confidence[:, 0]

ranking_t+2:
  modalities sorted by confidence[:, 1]

ranking_t+3:
  modalities sorted by confidence[:, 2]
```

示例：

```json
{
  "modality_ranking": {
    "avg": ["image", "radar", "lidar", "gps", "mmwave"],
    "t+1": ["image", "radar", "lidar", "gps", "mmwave"],
    "t+2": ["radar", "image", "lidar", "gps", "mmwave"],
    "t+3": ["image", "lidar", "radar", "gps", "mmwave"]
  }
}
```

这个诊断非常重要，用来分析：

```text
1. 强模态是否一直强。
2. 弱模态是否在某些 horizon 上变强。
3. mmWave / GPS 是否只在 t+1 强，在 t+3 下降。
4. image / lidar / radar 是否对远期预测更有贡献。
```

---

# 11. Diagnostics 输出

每个 epoch 保存：

```text
outputs/<scene>/<run_name>/diagnostics/g2d_epoch_<epoch>.json
```

内容示例：

```json
{
  "epoch": 3,
  "num_pred": 3,
  "horizon_names": ["t+1", "t+2", "t+3"],

  "teacher_confidence": {
    "image": {
      "t+1": 0.12,
      "t+2": 0.10,
      "t+3": 0.08,
      "avg": 0.10
    },
    "radar": {
      "t+1": 0.15,
      "t+2": 0.11,
      "t+3": 0.09,
      "avg": 0.1167
    },
    "gps": {
      "t+1": 0.35,
      "t+2": 0.28,
      "t+3": 0.21,
      "avg": 0.28
    },
    "lidar": {
      "t+1": 0.18,
      "t+2": 0.14,
      "t+3": 0.11,
      "avg": 0.1433
    },
    "mmwave": {
      "t+1": 0.60,
      "t+2": 0.40,
      "t+3": 0.24,
      "avg": 0.4133
    }
  },

  "modality_ranking_weak_to_strong": {
    "avg": ["image", "radar", "lidar", "gps", "mmwave"],
    "t+1": ["image", "radar", "lidar", "gps", "mmwave"],
    "t+2": ["image", "radar", "lidar", "gps", "mmwave"],
    "t+3": ["image", "radar", "lidar", "gps", "mmwave"]
  },

  "active_modalities": ["image"],

  "loss": {
    "supervised": 1.23,
    "feature_kd": 0.08,
    "logit_kd": 0.31,
    "total": 1.39
  }
}
```

如果 student 有 unimodal aux logits，则额外保存：

```json
{
  "student_branch_confidence": {
    "image": {
      "t+1": 0.10,
      "t+2": 0.08,
      "t+3": 0.07,
      "avg": 0.0833
    }
  },
  "confidence_ratio": {
    "image": {
      "t+1": 0.83,
      "t+2": 0.80,
      "t+3": 0.875,
      "avg": 0.835
    }
  }
}
```

ratio 定义：

```text
confidence_ratio[m,h] =
student_branch_confidence[m,h] / teacher_confidence[m,h]
```

加 epsilon 防止除零。

---

# 12. Metrics 修改

在 validator / evaluation metrics 中增加：

```text
top1_t1
top1_t2
top1_t3
top1_avg

top3_t1
top3_t2
top3_t3
top3_avg

top5_t1
top5_t2
top5_t3
top5_avg
```

其中：

```text
t1 = t+1
t2 = t+2
t3 = t+3
```

不要再输出：

```text
top1_h0
top1_future_avg
top1_current
beam8_acc
```

推荐主指标：

```text
val_top1_avg
```

或者更明确：

```text
val_top1_pred_avg
```

保存到 metrics.json：

```json
{
  "val_top1_t1": 0.42,
  "val_top1_t2": 0.35,
  "val_top1_t3": 0.29,
  "val_top1_avg": 0.3533,
  "val_top3_avg": 0.58,
  "val_top5_avg": 0.67
}
```

---

# 13. 汇总脚本

新增：

```text
tools/analysis/collect_multimodal_imbalance_results.py
```

读取：

```text
outputs/<scene>/<run_name>/metrics.json
outputs/<scene>/<run_name>/train_log.json
outputs/<scene>/<run_name>/diagnostics/*.json
```

输出：

```text
outputs/analysis/multimodal_imbalance_summary.csv
```

字段：

```text
scene
run_name
method

top1_t1
top1_t2
top1_t3
top1_avg

top3_avg
top5_avg

teacher_conf_image_t1
teacher_conf_image_t2
teacher_conf_image_t3
teacher_conf_image_avg

teacher_conf_radar_t1
teacher_conf_radar_t2
teacher_conf_radar_t3
teacher_conf_radar_avg

teacher_conf_gps_t1
teacher_conf_gps_t2
teacher_conf_gps_t3
teacher_conf_gps_avg

teacher_conf_lidar_t1
teacher_conf_lidar_t2
teacher_conf_lidar_t3
teacher_conf_lidar_avg

teacher_conf_mmwave_t1
teacher_conf_mmwave_t2
teacher_conf_mmwave_t3
teacher_conf_mmwave_avg

ranking_avg
ranking_t1
ranking_t2
ranking_t3

final_active_modalities
```

---

# 14. 实验配置矩阵

建议方法：

```text
1. Majority baseline
2. Last-observed-beam baseline
   注意：这不是 label 中的 beam8，而是输入历史最后一个 beam 直接作为 t+1/t+2/t+3 的 naive prediction。

3. Best unimodal teacher
4. Joint fusion
5. Token Transformer fusion
6. CRAF no prior
7. CRAF teacher prior
8. G2D-lite
9. G2D-global
10. G2D-horizon diagnostics
11. MARF
12. MARF + quality-aware
```

训练命令示例：

```bash
conda run -n kd_mm_beam python scripts/train.py \
  --config configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml \
  data.dataset.scene=9 \
  training.run_name=scene9_g2d_lite_pred3

conda run -n kd_mm_beam python scripts/train.py \
  --config configs/fusion/image_radar_gps_lidar_mmwave_g2d_global.yaml \
  data.dataset.scene=9 \
  training.run_name=scene9_g2d_global_pred3

conda run -n kd_mm_beam python scripts/train.py \
  --config configs/fusion/image_radar_gps_lidar_mmwave_g2d_horizon.yaml \
  data.dataset.scene=32 \
  training.run_name=scene32_g2d_horizon_pred3
```

---

# 15. 测试要求

## 15.1 G2D loss shape test

```text
tests/test_g2d_loss.py

构造：
B=2
H=3
C=64
M=5

student_logits: [B,3,64]
labels: [B,3]
teacher_logits[m]: [B,3,64]
student_features[m]: [B,8,64]
teacher_features[m]: [B,8,64]

检查：
1. total loss 是 scalar。
2. CE 使用全部 3 个 horizon。
3. feature_weight=0 时 feature loss 不影响 total。
4. logit_weight=0 时 KL loss 不影响 total。
5. teacher logits / teacher features 不产生 grad。
6. 如果 logits 是 [B,4,64]，必须抛错。
```

## 15.2 Teacher confidence test

```text
检查：
confidence[m,h] = batch mean softmax(teacher_logits[m][:,h])[labels[:,h]]

H 必须等于 3。
字段必须对应：
h=0 -> t+1
h=1 -> t+2
h=2 -> t+3
```

## 15.3 SMP scheduler test

```text
给定 confidence avg：
image=0.10
radar=0.20
gps=0.50
lidar=0.30
mmwave=0.80

weak_to_strong 应为：
image, radar, lidar, gps, mmwave

per_modality_tau=2：

epoch 0,1 active image
epoch 2,3 active radar
epoch 4,5 active lidar
epoch 6,7 active gps
epoch 8,9 active mmwave
epoch >=10 active all modalities
```

## 15.4 Gradient mask test

```text
dummy model:
image_encoder
radar_encoder
gps_encoder
lidar_encoder
mmwave_encoder
fusion
head

active_modalities=["image"]

检查：
image_encoder grad 保留
radar/gps/lidar/mmwave encoder grad 清零
fusion/head grad 保留
```

## 15.5 Metrics test

```text
给定 logits [B,3,64] 和 labels [B,3]。

检查输出：
top1_t1
top1_t2
top1_t3
top1_avg
top3_avg
top5_avg

不得输出旧字段：
top1_h0
top1_future_avg
beam8_acc
```

## 15.6 Diagnostics test

```text
检查 diagnostics json 包含：

num_pred = 3
horizon_names = ["t+1", "t+2", "t+3"]
teacher_confidence
modality_ranking_weak_to_strong
active_modalities
loss.supervised
loss.feature_kd
loss.logit_kd
loss.total
```

最后执行：

```bash
conda run -n kd_mm_beam pytest -q tests/test_g2d_loss.py tests/test_g2d_distiller.py tests/test_g2d_smp.py tests/test_g2d_diagnostics.py
conda run -n kd_mm_beam pytest -q
```

---

# 16. 实现优先级

```text
Priority 1:
全项目检查并修正旧的 num_pred+1 假设。
确保所有模型、loss、metrics 都使用 H=num_pred=3。

Priority 2:
实现 G2D-lite。
先不加 SMP，只验证：
CE + feature KD + logit KD 可以训练。

Priority 3:
实现 teacher confidence diagnostics。
确认每个模态都有 t+1/t+2/t+3 confidence。

Priority 4:
实现 G2D-global SMP。
用 3 个 horizon 的平均 confidence 做模态排序。

Priority 5:
实现 G2D-horizon diagnostics。
保存 t+1/t+2/t+3 各自的 weak-to-strong ranking。

Priority 6:
跑 Scene9。
确认 G2D-lite / G2D-global 正常。

Priority 7:
跑 Scene32。
重点分析 G2D 是否会因为 suppression GPS/mmWave 而导致远期预测下降。
```

---

# 17. 关键注意事项

```text
1. 不要再使用 num_pred + 1。
   新的 horizon 数就是 num_pred=3。

2. 不要再把 beam8 / 当前最后 beam 放进 label。
   beam8 只能作为历史输入信息，不能作为预测 label。

3. 不要再使用 future_only 配置。
   现在所有 label 都是 future，horizons=all 即可。

4. 不要再输出 h0/current 指标。
   输出 t+1/t+2/t+3。

5. Last-observed-beam baseline 可以保留。
   但它只是 naive baseline：
   用输入历史最后 beam 直接预测 t+1/t+2/t+3。
   它不是训练 label 的一部分。

6. G2D-global 的排序使用：
   avg_confidence = mean(confidence[t+1], confidence[t+2], confidence[t+3])

7. G2D-horizon 主要用于诊断：
   分析不同 horizon 下强弱模态是否变化。

8. 如果某些旧 checkpoint 输出 [B,4,64]，不要自动截断。
   应该报错或要求重新训练 teacher。

9. MARF 仍是主方法。
   G2D 是通用多模态失衡 baseline。
```

---

# 18. 验收标准

```text
实现完成后必须满足：

1. conda run -n kd_mm_beam pytest -q 通过。

2. 所有 teacher / student logits 都是：
   [B,3,64]

3. 所有 labels 都是：
   [B,3]

4. metrics.json 中出现：
   val_top1_t1
   val_top1_t2
   val_top1_t3
   val_top1_avg
   val_top3_avg
   val_top5_avg

5. diagnostics 中出现：
   horizon_names = ["t+1", "t+2", "t+3"]
   teacher_confidence
   modality_ranking_weak_to_strong
   active_modalities
   loss.supervised
   loss.feature_kd
   loss.logit_kd
   loss.total

6. 不再出现旧字段：
   top1_h0
   top1_future_avg
   beam8_acc
   num_pred_plus_one

7. G2D-lite 可以在 Scene9 跑通至少 1 epoch。

8. G2D-global 可以正确显示每个 epoch 的 active modalities。

9. 不影响已有 fusion_student / craf_fusion / marf_fusion 的训练。
```

---

这版的核心变化是：**所有东西都围绕 `[t+1, t+2, t+3]` 三个未来标签重写**。这样 G2D 的 confidence、SMP 排序、metrics 和 diagnostics 都会更干净，也不会再被 beam8 当前步信号干扰。
