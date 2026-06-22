## Context

现有 P0-P5 对照主要在 Image+GPS 和 JEPA sweep 内部比较，缺少使用 LiDAR/radar 的公开强 baseline。TII DeepSense6G 2022 challenge 方案属于外部 workflow/paper reproduction：它有自己的预处理、模型结构和 checkpoint provenance，最小可维护路径是包内复现 owner + 统一指标适配，而不是复制进通用训练循环。

约束：

- 所有项目 Python 命令使用 `conda run -n kd_mm_beam ...`。
- 外部 repo、checkpoint、预处理缓存和预测结果默认位于 ignored `outputs/analysis/tii_vlrg_transformer_reproduction/` 或用户显式指定路径。
- 不提交真实 `dataset/`、checkpoint、cache、log 或下载的外部源码副本。
- 不新增 root-level 旧脚本，不恢复 package-level facade。

## Goals / Non-Goals

**Goals:**

- 提供 TII VLRG Transformer baseline 的可审计复现入口。
- 记录外部源码版本、checkpoint、输入模态、split、scene set、metric profile、DBA provenance 和复现状态。
- 将 TII clean DBA 或 P0-P5 real-forward/预测结果适配为本仓库统一 summary rows。
- 支持在没有真实数据或 checkpoint 的测试环境中做 dry-run、manifest 和 summary adapter 验证。

**Non-Goals:**

- 不把 TII 官方模型重写成长期维护的普通 `modular_sequence` 模型。
- 不把外部 repo 或 checkpoint vendoring 到源码。
- 不把 radar-only/LiDAR-only sanity baseline 混入本 change。
- 不承诺立刻生成真实 claim；缺少外部 artifact 时只能标记为 `pending`、`unavailable` 或 `not_comparable`。

## Decisions

1. 采用 workflow/paper reproduction 路径。
   - 理由：TII 包含官方 challenge 预处理、外部 checkpoint 和模型脚本，直接用 workflow owner 比改造通用训练循环更少代码。
   - 备选：新增 whole-model exception。暂不采用，因为第一版只需要可审计复现和 summary adapter，没必要把官方实现重写成本仓库模型。

2. 新增窄 owner：`kd_sensing.baselines.tii_vlrg_transformer`。
   - owner 负责 manifest 解析、外部命令构造、dry-run、prediction/metric CSV ingestion 和 summary row 生成。
   - 包内 CLI 只做参数解析和 owner 调用，避免 root script 和兼容 facade。

3. 统一输出为 baseline reproduction manifest。
   - manifest 至少包含 model_id、source_repo、source_commit、checkpoint_path、enabled_modalities、scene_set、split、metric_profile、prediction_path、metrics_path、status 和 warnings。
   - summary adapter 输出字段对齐现有 DBA/P0-P5 表：model、source、overall_clean、P0-P5、overall_p0_p5_mean、strict_comparability 和 provenance。

4. 真实执行与测试解耦。
   - 单元测试只覆盖 dry-run manifest、命令安全、缺失 artifact 状态和 synthetic metrics adapter。
   - 真实 DeepSense6G + TII checkpoint 运行通过用户显式命令执行，不作为 CI/单元测试前置。

## Risks / Trade-offs

- [Risk] 外部 repo 结构或 checkpoint 路径变化。→ Mitigation: manifest 记录 source commit 和 artifact fingerprint；缺失或不匹配时标记 unavailable，不静默回退。
- [Risk] TII 原协议与当前 P0-P5 split/DBA 口径不完全一致。→ Mitigation: strict comparability 字段不一致时禁止 claim upgrade，只保留 external-reference row。
- [Risk] 引入 radar/LiDAR 后数据准备成本高。→ Mitigation: 第一版允许 dry-run 和 imported prediction CSV；真实预处理作为 opt-in 步骤。
- [Risk] wrapper 膨胀成第二套训练平台。→ Mitigation: owner 只包装外部 workflow 和 summary ingestion，不复制长期通用训练 loop。
