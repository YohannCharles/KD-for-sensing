## Context

Scene31 BTAPA tau1 当前是最值得保留的候选，但旧 V3、proto baseline 和 BTAPA tau1 的 only-radar 指标差异过大。代码里已经存在 `src/kd_sensing/eval/missing_patterns.py`、共享 `run_evaluation_pass`、训练 runtime early stopping 能力和 BTAPA 分析脚本，因此本变更应收敛这些已有路径，而不是新增一套独立训练/评估框架。

## Goals / Non-Goals

**Goals:**
- 将 missing pattern 的标准顺序、alias、mask 和名称集中到一个 helper。
- 用同一套评估函数重新 load checkpoint 复评 V3/proto/BTAPA tau1。
- 增加 tau1 seed2/seed3、es20 配置、seed mean±std 分析和关键 launcher。
- 让 summary 正确认出 early-stopped run。

**Non-Goals:**
- 不扩 RBMA、JEPA、KD、fullaux 或新的 prototype 变体。
- 不覆盖已有 outputs、logs、checkpoint 或历史结果。
- 不把本地分析脚本注册成新的 package console script。
- 不启动完整训练作为实现验证。

## Decisions

1. 复用现有 `src/kd_sensing/eval/missing_patterns.py` 作为统一 helper 的落点。  
   原因：已有 eval helper 已被脚本引用，改这里比新增 `utils/` 再迁移更小。替代方案是新增 `utils/missing_patterns.py`，但会制造第二个口径入口。

2. apples-to-apples 脚本复用现有配置加载、模型构建、checkpoint load、dataloader 和 `run_evaluation_pass`。  
   原因：需求要求重新 load checkpoint 和统一 eval，复用 runtime 才能避免指标计算漂移。脚本只负责选择 checkpoint、构造 force mask、写出表格和打印结论。

3. seed/es20 配置从 `main_v3_strong_reliability_btapa_tau1.yaml` 复制最小差异。  
   原因：用户明确要求除 seed 与输出路径外保持一致，es20 只增加 max epoch 与 early stopping 字段。

4. early stopping 状态识别优先读现有 metrics/train log/run metadata。  
   原因：训练 runtime 已有 early stopping 规格，summary 不应假设未跑满就是失败；需要兼容已有旧 run。

5. 分析脚本允许缺失 seed 并输出 n。  
   原因：seed2/seed3 可能还未跑完，分析入口应支持阶段性汇总。

6. checkpoint 选择集中到 `kd_sensing.utils.checkpoint_resolver`。  
   原因：多个脚本按文件名前缀、mtime 或 glob 顺序各选一次会把 seed run 与非 seed run 混淆；resolver 先读 metrics/sidecar，再回落到文件名 acc，并明确返回 warning。

7. eval pattern 输出保留标准四模态顺序，但 forward mask 按模型配置顺序生成。  
   原因：报告需要固定 `[gps, image, radar, lidar]` 口径；模型实际 `modalities` 可能仍是 `image/radar/gps/lidar`，不能把报告 mask 直接送进模型。

## Risks / Trade-offs

- pattern 顺序从旧 `image/radar/lidar/gps` 收敛到 `gps/image/radar/lidar` 可能影响旧脚本读法 → 保留 alias 与旧函数名，输出 canonical 名称，并让脚本显式使用统一 helper。
- 不同 run 的 config/checkpoint 命名可能不一致 → checkpoint manifest 记录策略、路径、epoch、warning，缺失时不崩溃。
- 真实复评需要本地 dataset/checkpoint → 实现验证以 help/dry-run/单元级 smoke 为主，无法替代完整实验结果。
- early stopping 字段历史格式可能分散 → summary 使用 metrics、train log 和 metadata 多来源识别，找不到标记时降级为 incomplete_has_checkpoint。

## Migration Plan

1. 新增/修正 missing pattern helper，并让 eval/analysis/summary 脚本复用。
2. 新增 apples-to-apples 与 seed mean±std 分析脚本。
3. 新增 tau1 seed/es20 YAML 和关键 launcher。
4. 修 summary 状态识别并增强 BTAPA 分析输出。
5. 补 proto baseline 三 seed 复核、8 卡 launcher、debug consistency、proto-vs-BTAPA mean±std 分析。
6. 运行 OpenSpec strict validate、脚本 help/dry-run、配置加载和可用的 focused tests。
