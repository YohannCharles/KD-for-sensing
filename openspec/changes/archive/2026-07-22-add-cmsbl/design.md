## Context

当前主线是 T2 + BCACL U2：四模态 encoder 产生 Beam 表示，supervised Router 与融合 Beam prototype 负责推理，U2 仅在训练期对自然观测模态增加 private/shared CE。CMSBL 继续只改变训练目标，不增加推理模块。

本地证据已经关闭三条前序链路：

- PGCD -> prototype diagnostics -> PR-SQDF：动态质量不稳定超过全局 prior，停止。
- missing evidence -> residual -> feature/prototype fusion -> fallback：后续 gate 未通过，停止。
- BT-SCL：Stage B 从未启动且已被 CMSBL 主线取代，终止。

这些路线的数值报告继续保留在现有 `outputs/`，精确实现可由 Git/OpenSpec archive 追溯；current source 不承担历史复现兼容。

## Goals / Non-Goals

**Goals:**

- 使 current source 只包含 T2/S1、MMW baselines、BCACL U2、CMSBL 和传递运行依赖。
- CMSBL disabled 时保持 U2 state dict、forward、loss、采样和 checkpoint 行为不变。
- 用一个线性调度、一个固定 Top-1 capacity JSON 和一个 15-mask loss EMA 表达主线假设。
- 长期 CMSBL 状态可 checkpoint/resume，且只由训练 batch 更新。
- 保持现有 600-entry mask schedule、15-pattern validation 和推理路径不变。

**Non-Goals:**

- 不保留失败实验的 runtime、配置、thin wrapper、compatibility stub 或 current spec。
- 不实现 cosine/two-level 调度、prototype separability reference、sampling reweighting、residual KL、V5/V6。
- 不增加动态推理权重、MoE、teacher checkpoint、第二 trainer 或新 console script。
- 不触碰 `outputs/`、cache、dataset、日志和 checkpoint。

## Decisions

### 1. 一个 change 同时拥有主线和退役边界

`add-cmsbl` 是唯一 active change。前序完成 change 以不合并 retired specs 的方式归档；失败或未启动 change 先记录停止结论再归档。current specs 只描述最终闭包，避免另建 cleanup change 与 CMSBL 重叠。

### 2. BCACL 只保留 U2

`BCACLModule` 只包含每模态 projection/private head 和共享 head。删除 modality prototype、quality、teacher selection、phase1/phase2 和 relation KL。U2 以 `aux_joint` 方式把 private/shared CE 加到现有融合 loss；`observed_mask` 与 `fusion_mask` 继续分离。

### 3. CMSBL 是 extension state，不是模型

CMSBL 纯函数放在 `losses/cmsbl.py`，配置放在 `losses/cmsbl_config.py`。它不注册模型、不创建参数、不改变推理 forward。关闭时仍调用原 U2 标量归约。

### 4. M1 只用 linear decay

private/shared 各自配置 start、end、start epoch 和 end epoch。区间外 clamp，区间内线性插值；`start=end` 即 constant，不再维护多种 schedule enum。

### 5. M2 只接受固定 standalone Top-1 JSON

容量文件在启动时读取一次，必须包含 dataset、source split、四模态 Top-1 和 SHA provenance；outer/test 来源 fail closed。train-only 逐模态 Top-1 epoch EMA 与固定 reference 形成 `max(0, C-A)/(C+eps)`，有界权重只作用于 private/shared CE。

### 6. M3 只重加权已有 loss

`image/radar/gps/lidar` 固定为 bit 0--3，非空 mask ID 为 1--15。每 epoch 按未加权的 per-sample fusion CE + BPA restoration 更新 EMA；warmup/低计数 mask 权重为 1，其余权重 clip 后归一化为均值 1。采样 panel、随机流、Router、superset consistency 和 private/shared loss不变。

### 7. 长期状态进 checkpoint，诊断只保留一份结构化事实

extension payload 保存 capacity identity/EMA 和 mask loss EMA/count。epoch accumulator 不跨 checkpoint。每 epoch 写一个 JSON，并复用现有 TensorBoard 标量；不额外维护可由 JSON 派生的 CSV。

### 8. 最小实验矩阵

- V0：BCACL U2。
- V1：V0 + M1。
- V2：V0 + M2。
- V3：V0 + M3。
- V4：V0 + M1 + M2 + M3。

本 change 交付代码、focused tests 和 synthetic smoke；真实 inner 实验由后续显式训练命令触发，不由源码 helper 自动调度。

## Risks / Trade-offs

- [删除历史 runtime 后无法从 HEAD 一键复跑旧试验] -> Git history、OpenSpec archive 和现有本地产物承担追溯；current contract 明确不提供兼容入口。
- [固定 capacity 与当前数据身份不匹配] -> 启动时校验 dataset、modalities、metric、split 和 SHA。
- [mask loss EMA 自强化] -> EMA 使用加权前 raw loss，warmup、min count、clip 和均值归一化限制反馈。
- [并行工作区存在未提交实现] -> 只保留与最终规格一致的部分，不回退无关用户改动，不修改任何 outputs。

## Migration Plan

1. 更新 current specs、研究 brief、inventory 和 retired-route 结论。
2. 删除前序路线代码、YAML、脚本、analysis 和测试；先保持 core tests 可收集。
3. 收缩 U-Mask/BCACL 并完成 CMSBL M1--M3。
4. 归档前序 changes，且不把 retired capability 合并回 current specs。
5. 运行 OpenSpec、architecture、config、compile、focused 和 full regression。
