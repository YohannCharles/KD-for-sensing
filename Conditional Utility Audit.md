你可以把下面这一整段直接发给 Codex。
这一版的目标不是马上改 MARF，而是先做 **Phase 1：Conditional Utility Audit**，判断 `image / radar / lidar` 到底是：

```text
1. 本身几乎没有价值；
2. 在当前 MARF 里没有被学到；
3. 平均很弱，但在某些通信状态 / 某些 horizon 下有条件性增益。
```

DeepSense 官方任务本来就以 Top-k beam accuracy 作为核心指标，最近的多模态 beam prediction 工作也常把 DBA 作为距离感知补充指标，因此 Phase 1 不要只看 Top1，而要同时看 `Top1 / Top3 / DBA`，并且拆到 `t+1 / t+2 / t+3`。([deepsense6g.net][1])

---

```text
任务：实现 MARF Phase 1 —— Conditional Utility Audit

项目：KD-for-sensing
当前模型：marf_fusion
当前 Scene：32
当前标签：
  num_pred = 3
  label = [t+1, t+2, t+3]
  logits shape = [B, 3, 64]

Phase 1 的目的：
不先改 MARF 结构，不先调 router，不先加通信特征。
先通过系统化 subset evaluation + per-sample analysis，回答：

1. 弱模态 image/radar/lidar 在 gps+mmwave 的强模态基线之上，是否存在边际价值？
2. 如果平均上没有提升，它们是否在某些样本、某些 horizon、某些通信状态下有条件性增益？
3. 当前 all-modal 比 strong-only 差，是因为：
   A. 弱模态本身没有价值；
   B. 弱模态有价值，但当前 MARF 没学会用；
   C. 弱模态只在特定状态有价值，当前 router 没识别出来？

本阶段不做：
- 不修改 MARF 主结构
- 不修改 router
- 不加入新的通信 feature
- 不改 loss
- 不解冻 encoder
- 不做 GPS-conditioned image masking
- 不做新模型训练，先以“评估与诊断”为主
```

# 1. Phase 1 的总体设计

Phase 1 分成两层，两个层次都要做。

## Layer A：当前 MARF 的路径使用审计

用当前已训练的 MARF checkpoint，在验证集上分别评估：

```text
all                   = image + radar + gps + lidar + mmwave
strong_only           = gps + mmwave
strong_plus_image     = gps + mmwave + image
strong_plus_radar     = gps + mmwave + radar
strong_plus_lidar     = gps + mmwave + lidar
single_best_mmwave    = mmwave
weak_only             = image + radar + lidar
```

这一层回答：

```text
当前 MARF 在现有训练结果下，弱模态加入后有没有被利用？
```

## Layer B：单模态补充信息审计

仅看 Layer A 还不够，因为当前 MARF 的弱模态可能根本没训练好。
所以还要利用现有单模态 teacher 输出，检查弱模态是否包含 **strong path 没有的互补信息**。

对每个样本、每个 horizon，保存：

```text
strong_only 的预测和概率
image_teacher 的预测和概率
radar_teacher 的预测和概率
lidar_teacher 的预测和概率
gps_teacher 的预测和概率
mmwave_teacher 的预测和概率
```

然后统计：

```text
当 strong_only 错时，image/radar/lidar 是否能答对？
当 strong_only 的 CE 很高时，某个弱模态是否对 ground truth 给出更高概率？
是否存在某些通信状态下，弱模态 teacher 的优势更明显？
```

这一层回答：

```text
弱模态是“没用”，还是“当前融合模型没学会用”？
```

---

# 2. 需要新增的 subset 定义

目前 validator 已经支持：

```text
all
top_prior / strong_only = gps + mmwave
single_best_prior = mmwave
weak_only = image + radar + lidar
```

请扩展为统一的 subset registry，建议新增文件：

```text
src/kd_sensing/evaluation/subset_specs.py
```

内容建议：

```python
from collections import OrderedDict

SCENE32_CONDITIONAL_UTILITY_SUBSETS = OrderedDict({
    "all": ["image", "radar", "gps", "lidar", "mmwave"],
    "strong_only": ["gps", "mmwave"],
    "strong_plus_image": ["gps", "mmwave", "image"],
    "strong_plus_radar": ["gps", "mmwave", "radar"],
    "strong_plus_lidar": ["gps", "mmwave", "lidar"],
    "single_best_mmwave": ["mmwave"],
    "weak_only": ["image", "radar", "lidar"],
})
```

要求：

```text
1. modality 顺序必须仍然遵守 src/kd_sensing/modalities.py 的中心契约：
   image -> radar -> gps -> lidar -> mmwave

2. subset 名称和模态列表必须写入输出文件 metadata。

3. validator 不要把 subset 逻辑硬编码在多个地方，统一从 subset_specs.py 读取。
```

---

# 3. 扩展 validator：支持 per-sample prediction dump

当前 validator 只保存 aggregate metrics 不够。
Phase 1 必须保存 **逐样本、逐 horizon、逐 subset** 的预测结果。

建议新增：

```text
src/kd_sensing/diagnostics/conditional_utility.py
```

并在 validator 增加开关：

```yaml
evaluation:
  conditional_utility_audit:
    enabled: true
    dump_per_sample_predictions: true
    dump_teacher_predictions: true
    subsets:
      - all
      - strong_only
      - strong_plus_image
      - strong_plus_radar
      - strong_plus_lidar
      - single_best_mmwave
      - weak_only
```

输出目录：

```text
outputs/scene32/<run_name>/conditional_utility/
```

输出文件建议：

```text
subset_predictions.parquet
teacher_predictions.parquet
conditional_utility_summary.json
conditional_utility_by_bucket.csv
conditional_utility_per_sample_delta.parquet
```

如果项目当前不依赖 `pyarrow`，可以先输出 `csv.gz`；但优先推荐 `parquet`，因为逐样本 × 3 horizon × 多 subset 行数会比较多。

---

# 4. `subset_predictions` 的字段设计

每一行代表：

```text
一个样本 + 一个 horizon + 一个 subset
```

建议字段：

```text
sample_id
seq_id
frame_idx
horizon_idx              # 0,1,2
horizon_name             # t+1,t+2,t+3
gt_beam

subset_name
modalities

pred_top1
pred_top2
pred_top3
top1_prob
top2_prob
top3_prob
gt_prob
ce                       # -log(gt_prob + eps)

top1_hit                 # 0/1
top3_hit                 # 0/1
top5_hit                 # 0/1
beam_distance_top1       # circular or current project DBA-compatible beam distance
dba_score                # per-sample DBA contribution if available
```

注意：

```text
1. 如果项目当前 DBA 是基于 top-3 计算，请复用现有 evaluation/metrics.py 的逻辑，
   不要在 diagnostics 中重新写一套不一致版本。

2. 如果现有 batch 中没有 sample_id，请构造稳定可复现 ID：
   sample_id = f"{seq_id}:{frame_idx}"
   如果 frame_idx 不可得，则在 dataset 里补充返回 index。

3. horizon_idx:
   0 -> t+1
   1 -> t+2
   2 -> t+3
```

---

# 5. `teacher_predictions` 的字段设计

每一行代表：

```text
一个样本 + 一个 horizon + 一个 teacher modality
```

字段：

```text
sample_id
seq_id
frame_idx
horizon_idx
horizon_name
gt_beam

teacher_modality          # image/radar/gps/lidar/mmwave
pred_top1
pred_top2
pred_top3
top1_prob
top2_prob
top3_prob
gt_prob
ce
top1_hit
top3_hit
top5_hit
beam_distance_top1
dba_score
```

要求：

```text
1. teacher 必须使用已有 teacher_registry.json 加载。
2. teacher checkpoint strict_load=true。
3. teacher 输出必须是 [B,3,64]。
4. teacher forward 使用 no_grad。
5. teacher_predictions 只在 conditional_utility_audit.enabled=true 时生成，避免普通训练评估变慢。
```

---

# 6. 核心统计：subset 的边际增益

以 `strong_only = gps + mmwave` 为基线。

对每个弱模态 `m in {image, radar, lidar}`，计算：

```text
ΔCE_m      = CE(strong_only) - CE(strong_plus_m)
ΔTop1_m    = Top1(strong_plus_m) - Top1(strong_only)
ΔTop3_m    = Top3(strong_plus_m) - Top3(strong_only)
ΔDBA_m     = DBA(strong_plus_m) - DBA(strong_only)
```

注意：

```text
ΔCE > 0 代表 strong_plus_m 更好
ΔTop1 > 0 代表 strong_plus_m 更好
ΔTop3 > 0 代表 strong_plus_m 更好
ΔDBA > 0 代表 strong_plus_m 更好
```

输出逐样本 delta 表：

```text
conditional_utility_per_sample_delta.parquet
```

字段：

```text
sample_id
seq_id
frame_idx
horizon_idx
horizon_name
weak_modality

ce_strong_only
ce_strong_plus
delta_ce

top1_strong_only
top1_strong_plus
delta_top1

top3_strong_only
top3_strong_plus
delta_top3

dba_strong_only
dba_strong_plus
delta_dba

strong_correct_top1
strong_correct_top3
strong_plus_correct_top1
strong_plus_correct_top3
```

---

# 7. Oracle 设计

必须实现两个 oracle。

## 7.1 Subset Oracle

候选 subset：

```text
strong_only
strong_plus_image
strong_plus_radar
strong_plus_lidar
all
```

对每个样本、每个 horizon，选 CE 最小的 subset：

```python
oracle_subset = argmin_subset(CE_subset)
```

统计：

```text
oracle_top1
oracle_top3
oracle_dba
oracle_ce
oracle_gain_vs_strong_only
oracle_choice_distribution
oracle_choice_distribution_by_horizon
```

输出：

```text
oracle_subset_summary.json
```

重点看：

```text
如果 oracle 相比 strong_only 几乎没有提升：
  弱模态当前表征下条件价值很小。

如果 oracle 明显高于 strong_only：
  弱模态存在隐藏价值，只是当前 MARF 没学会什么时候用。
```

## 7.2 Teacher Complementarity Oracle

用 teacher logits 检查单模态是否能补 strong path：

```text
strong_only 预测错，
但 image_teacher / radar_teacher / lidar_teacher 预测对
```

统计：

```text
rescue_top1_count
rescue_top3_count
rescue_rate_given_strong_top1_wrong
teacher_gt_prob_advantage_rate
teacher_ce_better_than_strong_rate
```

定义：

```python
teacher_rescue_top1 = (
    strong_only_top1_hit == 0 and teacher_top1_hit == 1
)

teacher_gt_prob_advantage = (
    teacher_gt_prob > strong_only_gt_prob
)

teacher_ce_better = (
    teacher_ce < strong_only_ce
)
```

输出：

```text
teacher_complementarity_summary.json
```

这个 oracle 很关键，因为它可以区分：

```text
A. 弱模态 teacher 自己也完全没信息；
B. 弱模态 teacher 有信息，但 MARF 没把它利用起来。
```

---

# 8. 通信状态 bucket 设计

Phase 1 必须按通信状态分桶，不然只能得到平均结论。

先不要加新特征到 MARF，只做分析。
请从已有输入中计算以下 bucket features：

## 8.1 mmWave 状态

从 mmWave power vector 计算：

```text
mmwave_entropy
mmwave_top1_prob
mmwave_top1_top2_margin
mmwave_peak_sharpness
mmwave_total_power
mmwave_peak_drift            # 当前窗口内 top1 beam index 的变化幅度
```

建议实现：

```text
src/kd_sensing/diagnostics/communication_state_features.py
```

## 8.2 GPS 状态

基于当前 relative_polar / 原始 GPS 序列，如果能拿到原始位置则优先原始位置：

```text
range_to_bs
bearing
delta_range
delta_bearing
angular_velocity
gps_jump_magnitude
```

如果当前 dataset 只能拿到 `relative_polar = [dist, sin_theta, cos_theta]`，则可先基于它恢复：

```text
bearing = atan2(sin_theta, cos_theta)
delta_bearing = wrapped angle difference
```

## 8.3 beam transition 状态

基于标签或历史 beam：

```text
beam_transition_t1 = 1 if label[t+1] != last_input_beam else 0
beam_transition_t2 = 1 if label[t+2] != label[t+1] else 0
beam_transition_t3 = 1 if label[t+3] != label[t+2] else 0
```

如果当前 batch 不直接返回 last_input_beam，需要确认 dataset 是否可取；若不可得，先只做 future transition：

```text
t+1 -> t+2
t+2 -> t+3
```

---

# 9. 分桶策略

不要一开始手工设很多绝对阈值，优先按验证集分位数分桶。

推荐：

```text
mmwave_entropy:
  low / mid / high 按 33% 和 66% 分位数

mmwave_margin:
  low / mid / high

gps_jump_magnitude:
  low / high 按 50% 分位数

range_to_bs:
  near / mid / far 按 33% 和 66% 分位数

angular_velocity:
  low / high

beam_transition:
  transition / stable
```

输出：

```text
conditional_utility_by_bucket.csv
```

字段：

```text
bucket_feature
bucket_name
weak_modality
horizon_name
num_samples

strong_only_top1
strong_plus_top1
delta_top1

strong_only_top3
strong_plus_top3
delta_top3

strong_only_dba
strong_plus_dba
delta_dba

mean_delta_ce
positive_delta_ce_rate
oracle_choice_rate
teacher_rescue_rate
```

---

# 10. 必须输出的核心 summary

生成：

```text
conditional_utility_summary.json
```

至少包含：

```json
{
  "run_name": "...",
  "scene": 32,
  "num_samples": 0,
  "horizons": ["t+1", "t+2", "t+3"],

  "aggregate_metrics": {
    "all": {},
    "strong_only": {},
    "strong_plus_image": {},
    "strong_plus_radar": {},
    "strong_plus_lidar": {},
    "single_best_mmwave": {},
    "weak_only": {}
  },

  "marginal_utility_vs_strong_only": {
    "image": {
      "delta_top1_avg": 0.0,
      "delta_top3_avg": 0.0,
      "delta_dba_avg": 0.0,
      "delta_ce_avg": 0.0,
      "positive_delta_ce_rate": 0.0
    },
    "radar": {},
    "lidar": {}
  },

  "marginal_utility_by_horizon": {
    "image": {
      "t+1": {},
      "t+2": {},
      "t+3": {}
    }
  },

  "oracle_subset": {
    "top1_avg": 0.0,
    "top3_avg": 0.0,
    "dba_avg": 0.0,
    "gain_vs_strong_only": {},
    "choice_distribution": {}
  },

  "teacher_complementarity": {
    "image": {},
    "radar": {},
    "lidar": {}
  }
}
```

---

# 11. 需要生成的图

新增脚本：

```text
tools/analysis/analyze_conditional_utility.py
```

输出到：

```text
outputs/scene32/<run_name>/conditional_utility/figures/
```

至少生成：

```text
1. subset_metrics_bar.png
   - strong_only / strong+image / strong+radar / strong+lidar / all
   - Top1 / Top3 / DBA

2. marginal_delta_by_horizon.png
   - x: t+1/t+2/t+3
   - y: delta Top1 / delta DBA
   - lines: image/radar/lidar

3. oracle_choice_distribution.png
   - 每个 horizon 选择了哪个 subset

4. teacher_rescue_rate.png
   - 当 strong_only 错时，image/radar/lidar teacher 的 rescue rate

5. delta_ce_histogram_<modality>.png
   - 看 weak modality 是平均无用，还是有长尾正收益样本

6. bucket_heatmap_delta_dba.png
   - bucket: mmwave_entropy / beam_transition / range / angular_velocity
   - modality: image/radar/lidar
   - value: delta DBA
```

绘图要求：

```text
1. 不要只画 overall average，必须支持 horizon 维度。
2. 图中明确标注 positive / negative gain。
3. 不要覆盖已有可视化工具。
```

---

# 12. Phase 1 的判断规则

请在 `conditional_utility_summary.json` 里自动给出一个初步 diagnosis 字段。

建议规则：

## 12.1 判断弱模态是否整体有用

```text
如果某个 weak modality：
  delta_dba_avg > 0
  且 positive_delta_ce_rate > 0.5
则标记：
  "global_useful"
```

## 12.2 判断是否存在条件性价值

```text
如果 overall delta <= 0，
但至少存在一个 bucket 满足：
  delta_dba > 0.01
  或 mean_delta_ce > 0
  且样本数 >= min_bucket_samples
则标记：
  "conditionally_useful"
```

## 12.3 判断当前融合是否没学会

```text
如果 teacher_complementarity 明显存在：
  rescue_rate_given_strong_wrong > threshold
但 strong_plus_modality 在 current MARF 中没有提升，
则标记：
  "representation_exists_but_not_exploited"
```

## 12.4 判断可能本身无价值

```text
如果：
  oracle gain 很小
  且 teacher complementarity 很小
  且所有 bucket 都无明显正增益
则标记：
  "currently_low_utility"
```

阈值不要硬编码死，放到 config：

```yaml
conditional_utility_audit:
  thresholds:
    min_bucket_samples: 100
    conditional_delta_dba: 0.01
    teacher_rescue_rate: 0.05
    oracle_gain_dba: 0.01
```

输出示例：

```json
"diagnosis": {
  "image": "conditionally_useful",
  "radar": "currently_low_utility",
  "lidar": "representation_exists_but_not_exploited"
}
```

---

# 13. 推荐新增配置

新增：

```text
configs/analysis/scene32_marf_conditional_utility_audit.yaml
```

示例：

```yaml
analysis:
  type: conditional_utility_audit
  scene: 32
  checkpoint: outputs/scene32/scene32_marf/checkpoints/best_top1.pt

  subsets:
    all: [image, radar, gps, lidar, mmwave]
    strong_only: [gps, mmwave]
    strong_plus_image: [image, gps, mmwave]
    strong_plus_radar: [radar, gps, mmwave]
    strong_plus_lidar: [gps, lidar, mmwave]
    single_best_mmwave: [mmwave]
    weak_only: [image, radar, lidar]

  teacher_registry:
    path: outputs/scene32/teacher_registry.json
    strict_load: true

  output_dir: outputs/scene32/scene32_marf/conditional_utility

  horizons: [0, 1, 2]
  horizon_names: [t+1, t+2, t+3]

  thresholds:
    min_bucket_samples: 100
    conditional_delta_dba: 0.01
    teacher_rescue_rate: 0.05
    oracle_gain_dba: 0.01

  bucket_features:
    - mmwave_entropy
    - mmwave_top1_top2_margin
    - range_to_bs
    - angular_velocity
    - gps_jump_magnitude
    - beam_transition
```

执行入口建议新增：

```text
tools/analysis/run_conditional_utility_audit.py
```

命令：

```bash
conda run -n kd_mm_beam python tools/analysis/run_conditional_utility_audit.py \
  --config configs/analysis/scene32_marf_conditional_utility_audit.yaml
```

---

# 14. 需要改动或新增的文件清单

建议新增：

```text
src/kd_sensing/evaluation/subset_specs.py
src/kd_sensing/diagnostics/conditional_utility.py
src/kd_sensing/diagnostics/communication_state_features.py
tools/analysis/run_conditional_utility_audit.py
tools/analysis/analyze_conditional_utility.py
configs/analysis/scene32_marf_conditional_utility_audit.yaml

tests/test_conditional_utility_metrics.py
tests/test_conditional_utility_oracle.py
tests/test_communication_state_features.py
tests/test_subset_specs.py
```

可能需要修改：

```text
src/kd_sensing/engine/validator.py
src/kd_sensing/evaluation/metrics.py
src/kd_sensing/diagnostics/__init__.py
```

如果 dataset 当前不返回 `sample_id / seq_id / frame_idx / last_input_beam`，可能还需要最小修改：

```text
dataset __getitem__
batch preparation
```

但要求：

```text
只增加 diagnostics 所需 metadata，不改变训练主流程。
```

---

# 15. 测试要求

## 15.1 subset spec test

```text
检查：
all
strong_only
strong_plus_image
strong_plus_radar
strong_plus_lidar
single_best_mmwave
weak_only

都存在，且 modality 顺序合法。
```

## 15.2 marginal delta test

构造 toy 数据：

```text
strong_only ce = 1.0
strong_plus_image ce = 0.7
```

检查：

```text
delta_ce = 0.3
positive_delta_ce = true
```

## 15.3 oracle test

构造多个 subset CE：

```text
strong_only = 0.8
strong_plus_image = 0.3
strong_plus_radar = 0.5
all = 0.6
```

检查：

```text
oracle_subset = strong_plus_image
```

## 15.4 teacher complementarity test

构造：

```text
strong_only top1 wrong
image_teacher top1 correct
```

检查：

```text
teacher_rescue_top1 = true
```

## 15.5 bucket feature test

检查：

```text
mmwave_entropy
top1-top2 margin
bearing delta
angular velocity
gps jump magnitude
```

输出 shape 和数值合法。

## 15.6 end-to-end smoke test

用 tiny dummy logits / labels 跑：

```text
subset_predictions
teacher_predictions
summary json
by_bucket csv
```

检查文件都生成。

---

# 16. Phase 1 完成后的人工阅读顺序

完成代码后，不要立刻继续改 MARF。先按这个顺序看结果：

## 先看总体

```text
strong_only vs all
strong_only vs strong+image
strong_only vs strong+radar
strong_only vs strong+lidar
```

## 再看 oracle

```text
oracle gain vs strong_only 有多大？
```

## 再看 teacher complementarity

```text
当 strong_only 错时，image/radar/lidar teacher 有没有 rescue 能力？
```

## 再看条件桶

重点看：

```text
1. 高 mmWave entropy
2. 低 mmWave margin
3. beam transition
4. 高 angular velocity
5. GPS jump
6. t+2 / t+3
```

这些 bucket 里弱模态是否有正收益。

---

# 17. Phase 1 结果如何指导 Phase 2

请不要在代码里自动做这些改动，只在 summary 里输出建议。

## 情况 A：oracle 几乎没提升

```text
结论：
弱模态当前表征下确实没有明显通信价值。

下一步：
不要强行拉高弱模态权重。
优先做 safe fusion / strong-path preservation，
让 all 至少不差于 strong_only。
```

## 情况 B：teacher 有 rescue，但 current MARF 没提升

```text
结论：
弱模态包含信息，但当前 MARF 没学会利用。

下一步：
开启 subset training，
加 weak modality aux，
考虑弱模态 adapter 解冻。
```

## 情况 C：整体平均不提升，但某些 bucket 明显有提升

```text
结论：
弱模态是 conditionally useful。

下一步：
做 MARF-Comm：
把 mmWave uncertainty、beam transition、range、angular velocity 等通信状态输入 router，
让 router 学“什么时候打开弱模态”。
```

## 情况 D：strong+weak 平均提升

```text
结论：
弱模态有直接边际价值。

下一步：
优先修正当前 all-modal 路径，
开启 subset training，
再考虑更强的跨模态交互。
```

---

# 18. 验收标准

Phase 1 完成必须满足：

```text
1. 能使用一个 MARF checkpoint 独立运行 audit。
2. 能输出所有 subset 的 aggregate metrics。
3. 能输出 per-sample prediction dump。
4. 能输出 strong_only vs strong+weak 的 per-sample delta。
5. 能输出 subset oracle。
6. 能输出 teacher complementarity oracle。
7. 能按通信状态 bucket 输出 delta Top1 / Top3 / DBA / CE。
8. 能生成 summary json、csv 和图。
9. 不改 MARF 训练逻辑，不影响现有训练。
10. pytest -q 全部通过。
```

---

## 我建议你自己重点盯住的几个结果

真正决定下一步方向的不是总平均，而是这四个数：

```text
1. oracle_gain_vs_strong_only_dba
2. teacher_rescue_rate_given_strong_wrong
3. positive_delta_ce_rate_by_bucket
4. delta_dba on:
   - high mmwave entropy
   - beam transition
   - t+2 / t+3
```

### 你最希望看到的理想结果

```text
平均上：
  strong+weak 不一定明显超过 strong_only

但在：
  high mmWave entropy
  beam transition
  t+2/t+3
这些子集里：
  image 或 lidar 或 radar 有明显正收益

同时：
  oracle 明显高于 strong_only
```

如果出现这个结果，你的后续论文故事就会非常清楚：

```text
弱模态不是全局强模态，
而是在特定链路状态下具有边际通信价值。
因此问题不是“均衡所有模态”，
而是“识别何时需要调用外部感知模态”。
```

这会比普通 AI 里的“多模态平衡”更像一个真正的通信问题。

[1]: https://www.deepsense6g.net/vision-aided-beam-prediction/?utm_source=chatgpt.com "Vision-Aided Beam Prediction"
