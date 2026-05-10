## 结论先说清楚

你这次 Phase 1 的结果，已经基本否定了一个很重要的假设：

```text
在干净的 Scene32 上，image / radar / lidar 并不是“明明有很大价值，只是被 GPS/mmWave 压制了”。
```

更接近真实的结论是：

```text
Scene32 当前设定下是 GPS + mmWave 主导；
弱模态的边际效用接近 0；
all-modal 略低于 strong-only，当前主要问题是弱模态负迁移，而不是弱模态被严重埋没。
```

证据很一致：

```text
strong_only Top1 = 0.4240, DBA = 0.9037
all         Top1 = 0.4232, DBA = 0.9036

weak modality 加入后的 DBA 增益只有 0.0002~0.0003
oracle DBA gain 只有 +0.00093
teacher rescue 只有 6.48%
weak_only Top1 只有 0.0159
```

所以，**现在不要立刻做更复杂的 MARF router，不要先做 GPS 引导 image mask，不要先做大规模跨模态交互。**
当前证据还不支持你把精力投到这些方向上。

---

# 下一步我最建议做什么

## 做一个 Phase 1.5：把“弱模态几乎没用”这个结论彻底坐实

你当前的 Phase 1 很有价值，但它还主要是在**一个已训练 MARF checkpoint 上做 subset masking 评估**。它说明：

```text
当前 MARF 没有从弱模态获得明显收益。
```

但它还不能 100% 说明：

```text
即使专门为 strong+weak subset 训练，弱模态也没有价值。
```

所以我建议下一步不是 MARF-v2，而是先补三个验证。

---

## 1. 先给现有结果加统计置信度

你现在的增益太小了：

```text
delta_dba ≈ +0.0002 ~ +0.0003
```

这种量级很可能只是波动。
下一步应直接基于你已经生成的逐样本表，做：

```text
paired bootstrap 95% CI
```

最好不要按单个窗口独立采样，而是按 `seq_id` 做 **cluster bootstrap**，因为同一序列里的样本高度相关。

至少对这些差值做置信区间：

```text
strong_plus_image - strong_only
strong_plus_radar - strong_only
strong_plus_lidar - strong_only
all - strong_only
```

指标：

```text
Top1
Top3
DBA
CE
```

### 你现在应该期待的判断

如果 `delta_dba` 的 95% CI 跨 0，那么就不要把 `+0.00031` 解读成真实收益。
当前 `global_useful` 的 diagnosis 阈值也需要改掉，`global_delta_dba: 0.0` 太宽了，建议至少改到：

```text
global_delta_dba >= 0.001
且 95% CI 下界 > 0
```

这一步能避免你后面被极小数值误导。

---

## 2. 训练 dedicated fixed-subset baseline，而不是只看当前 MARF 的 masking 结果

请专门训练下面 5 个固定子集模型：

```text
A. strong_only        = gps + mmwave
B. strong_plus_image  = gps + mmwave + image
C. strong_plus_radar  = gps + mmwave + radar
D. strong_plus_lidar  = gps + mmwave + lidar
E. all                = image + radar + gps + lidar + mmwave
```

为什么这一步必须做？

因为你现在的 `strong_plus_image / radar / lidar` 是在一个已经几乎把 anchor 压到 mmWave 的 MARF 上做评估。
这对“当前 MARF 有没有利用它们”很公平，但对“这些弱模态如果被专门训练，能否带来增益”还不够公平。

### 训练要求

为了让结论干净，第一轮先不要搞复杂：

```text
同一训练预算
同一 checkpoint 选择规则
同一 encoder 初始化
同一 loss
同一评估协议
同一 random seeds
```

建议至少：

```text
3 seeds
```

输出：

```text
mean ± std
Top1_t1/t2/t3
Top3_t1/t2/t3
DBA_t1/t2/t3
avg
```

### 这一步最关键的 baseline 是谁？

不是 `all`，而是：

```text
专门训练的 strong_only = gps + mmwave
```

因为它才是 Scene32 上真正应该被打败的基线。
你现在的 `strong_only 0.4240` 只是当前 MARF 子路径表现，专门训练后的 `gps+mmwave` 很可能还会更强。

---

## 3. 把 Phase 1 在多个 checkpoint 上重跑一遍

至少重跑：

```text
best_top1 checkpoint
best_dba checkpoint
final checkpoint
```

原因是你现在的 MARF：

```text
best top1 在 epoch 75
final metrics 在 epoch 100
```

不同 checkpoint 的 router 塌缩程度可能不同。
如果三个 checkpoint 都得到同样结论，说明 Phase 1 的判断很稳。

---

# 做完 Phase 1.5 后怎么决策

## 情况 A：专门训练后，strong+weak 仍然没有显著超过 strong_only

这是我目前认为**概率最大**的结果。

如果出现：

```text
strong+image / radar / lidar
在 3 seeds 下都没有稳定超过 strong_only，
oracle gain 仍然很小，
bootstrap CI 也不支持真实收益，
```

那就应该正式接受这个结论：

```text
干净 Scene32 不是一个适合讲“弱模态被强模态压制但本来很有用”的主场景。
```

这时你的下一步应该改成：

### 路线 1：如果你只想提高 Scene32 精度

直接把精力转向：

```text
GPS + mmWave 强路径建模
```

而不是继续扩弱模态融合。

可以做的改进包括：

```text
1. 更强的 mmWave 时序建模
2. GPS 运动学特征：range、bearing、angular velocity、delta bearing
3. horizon-aware prediction head
4. beam transition 建模
5. top1 sharpening / ranking loss
```

这条线对精度最实际。

### 路线 2：如果你还想保留 MARF 论文主线

把 MARF 在 Scene32 的目标改成：

```text
safe fusion
```

不是“让所有模态都贡献”，而是：

```text
all-modal 至少不比 strong-only 差；
弱模态只有在有证据时才允许 residual 介入。
```

这时再做：

```text
subset training
strong-path preservation loss
weak residual sparsity
anchor 只给 strong modalities，weak modalities residual-only
```

这会比强行把 weak anchor 权重拉高更合理。

---

## 情况 B：专门训练后，某个 weak modality 在某些 bucket / horizon 上有稳定收益

如果出现：

```text
strong_plus_image / lidar / radar
在某些通信状态下稳定正收益，
并且 bootstrap CI 不跨 0，
```

那才进入你之前设想的：

```text
MARF-Comm
```

也就是把 router 从：

```text
“哪个模态自己强”
```

升级为：

```text
“在当前通信状态下，哪个模态能给 strong path 带来边际收益”
```

这时再加：

```text
mmWave entropy
mmWave margin
range
angular velocity
GPS jump
beam transition
```

去学 conditional utility gate，才是有证据支持的下一步。

---

# 你现在最不该做的三件事

## 1. 不要因为 oracle Top1 +0.0082 就立刻做复杂 subset router

这个提升说明：

```text
少数样本上选择 subset 有一点价值
```

但 DBA 只提升：

```text
+0.00093
```

说明通信意义上的收益非常小。
这还不足以支持你投入大量结构设计。

---

## 2. 不要因为 bucket 里有轻微信号，就立刻做 GPS-image 几何交互

你自己也已经看到：

```text
只是少数 t+3 / 低 range / 低运动变化 bucket 有很轻微信号
```

这更像“局部现象”，还不是“明确机制”。
现在就做 GPS 裁 image、sector attention、BEV alignment，风险很高，可能最后只是把工程复杂度堆上去，却没明显收益。

---

## 3. 不要继续把“模态失衡”理解成“所有模态都该被用到”

你这次实验最有价值的科学发现就是：

```text
通信任务里的模态失衡，不一定等价于弱模态被压制；
有时就是不同模态的任务效用本来不对称。
```

这比普通 AI 里的“把弱模态拉起来”更贴近通信系统。

---

# 我建议你接下来的最优路线

## 近期一周内

```text
1. 修 diagnosis 阈值
2. 给现有 audit 加 cluster bootstrap CI
3. 重跑 best_top1 / best_dba / final checkpoint audit
4. 训练 dedicated fixed-subset baselines：
   strong_only
   strong+image
   strong+radar
   strong+lidar
   all
```

## 如果结果继续支持“weak utility 很低”

```text
5. 把 Scene32 定位为：
   RF/GPS-dominant clean setting + negative transfer case

6. 做 MARF-Safe：
   - subset training
   - strong-path preservation
   - weak residual-only
   - no-harm objective

7. 另开一个更有通信味的实验：
   在 mmWave degraded / GPS perturbed 情况下重跑 conditional utility audit
```

这一点我很建议你做。
因为外部感知模态在**干净 mmWave + GPS 已经很强**时没什么价值很正常；它们真正可能有价值的场景往往是：

```text
mmWave 不确定
RF 受噪声或缺失
GPS 跳变
beam transition
```

这比在 clean Scene32 上硬讲“五模态平衡”更像真实通信问题。

---

# 如果你问我“现在唯一最该做的下一步是什么”

我的答案是：

```text
先训练 dedicated strong_only / strong+weak 固定子集模型，
并对现有 tiny gains 做 bootstrap 置信区间。
```

因为这一步会决定你后面整条研究路线：

```text
如果弱模态专门训练后仍然没用：
  不要再往复杂多模态融合上投太多；
  转向 strong-path + safe fusion + RF-degraded robustness。

如果弱模态专门训练后在某些状态下稳定有用：
  再做 MARF-Comm 的条件效用 router。
```

你现在已经不缺“再加一个模块”的想法了。
你现在最需要的是先把**弱模态到底值不值得救**这件事彻底判清楚。
