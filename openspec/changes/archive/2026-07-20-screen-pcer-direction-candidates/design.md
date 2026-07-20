## Context

PCER A0-A3 已共享 MMW 15-domain、seed1、16 epoch、batch32、BPA/prototype、pcer curriculum 和 validation-best checkpoint。诊断显示 A1 仍有小幅样本动态价值，A3 target 高熵且 equal-coalition 与部署 policy 失配，missing_lidar 暴露非 lidar evidence 过弱。八方向必须保持共同训练身份，只改变候选机制，并在八张 A40 上 fail-independent 并行运行。

## Goals / Non-Goals

**Goals:**

- 用一套共享 block evidence/router API 表达 B0-B7，不复制 encoder、prototype 或训练循环。
- 保持缺失位置权重为零、所有层级归一化正确，并让 direct beam loss 始终可向预测 router 反传。
- 让 supervised target 只在缓存 evidence/router 层计算并 detach；on-policy LOO 不重复 backbone。
- 用真实训练 batch preflight、固定 S0-S5 评测、权重替换、target/梯度/成本诊断形成 Pareto 决策。

**Non-Goals:**

- 不调旧 equal-coalition target，不新增数据集、prototype、模型注册名或 canonical recipe。
- 不跑 multi-seed、outer test、双模态组合矩阵或论文预算。
- 不从 A1/A2/A3 最终 checkpoint 分别初始化候选。

## Decisions

1. **一个 opt-in PCER 组件族。** `model.primary.pcer.mode` 扩展为 `evidence_only`、`block_router`、`hierarchical_router`、`mask_residual_router`，同时保留历史 `evidence_static/counterfactual_router`。B0/B7 的 `evidence_only` 只导出 block evidence，实际融合仍由 old modality router 完成；其余模式在 block evidence 上融合。
2. **共享 flat scoring，最少新增结构。** Flat router 复用现有 feature/confidence/entropy/embedding scorer。Hierarchical router 用同一 block score 做 modality alpha 与 within-modality beta 的显式双层 softmax；mask residual router 用 availability descriptor MLP 产生 prior，再叠加零均值动态 score 和 sigmoid residual scale。B4/B5、B1/B2/B3 分别共享参数结构，确保差异只来自 supervision。
3. **三种 target 一个 KL owner。** Standalone target 使用负单块 topology CE；on-policy block/modality target 将缓存 `[B,N,D/K]` 扩展成 BxN 或 BxM removal views，一次向量化重跑轻量 router/fusion。Target 分支在 `no_grad` 中执行，prediction 不 detach。
4. **B7 只增加必要 evidence 约束。** 每 batch 以 `(epoch + step) mod M` 均衡选择删除模态，用 detached full-view teacher 和一个 LOMO student forward 计算 KL；unimodal auxiliary 直接复用主 forward 的 pooled modality logits，不再运行 backbone。
5. **共同初始化从头 seed1。** 历史 quick 配置没有共同基础 checkpoint，八方向均从 seed1 scratch；encoder 和 prototype 在候选 router 实例化前创建，保证共同模块初始 RNG 一致。生成配置保留在 outputs，不成为 canonical 输入。
6. **preflight 最多自动调整一次。** 每方向真实一个训练 batch计算 beam/common/new loss 与梯度。只有新增加权 loss 不在 beam loss 的 1%-100% 区间时，按目标 10% 比例一次缩放其 lambda 并记录；不读取 validation/test。
7. **I/O 操作设置统一但不冒充方法差异。** 八任务均用 4 workers/prefetch1，减少共享盘 96-worker 争用；数据、batch、有效 batch 和随机 mask 不变。Host timing 每 10 batch 记录吞吐与峰值 CUDA memory。
8. **评测一次 forward 多模式重算。** Validation 统计 global/mask mean logits；test 同一 evidence 评测 dynamic/global/mask replacement。历史 A0-A3 直接读取已有 metrics，B7 额外运行四个单模态 survival view。

## Risks / Trade-offs

- [B7 多一个 student backbone forward，成本更高] → 明确计入吞吐/显存并由 Pareto 规则惩罚，不给额外 epoch。
- [On-policy BxN 路由扩展增加显存] → 只扩展 20-block 缓存 tensor，不扩展原始传感器或 encoder graph，target 使用 no-grad。
- [GPU0 有无关显存占用] → 启动时记录 baseline memory；只要剩余显存满足统一阈值就运行，不终止外部进程。
- [单 seed 小效应存在噪声] → 结果只分 Winner/Promising/Reject，不升级 claim；若无人超过 A1 则优先简单方向或 B7 evidence 证据。

## Migration Plan

先添加默认兼容和专项 synthetic tests，再生成八组 config 做单 batch preflight；通过后 GPU0-7 并行训练并用 best checkpoint 评测。回滚只需删除 opt-in 模式和筛选脚本，本地 outputs 可独立清理；canonical T2 和历史 checkpoint 不变。

## Open Questions

无。新增 loss 的一次量级调整由预注册规则自动完成，方向选择只在全部八组完成后进行。
