## Context

现有 Scene31 night-grid workflow 已有 es20 配置生成器、8 GPU launcher、fresh eval 和粗筛分析。当前请求是在不改变 baseline 与 es20 输出的前提下，补充 40 epoch follow-up、组合方法和可复现汇总。相关能力仍属于本地 local/manual 实验面，不升级为 package CLI。

## Goals / Non-Goals

**Goals:**
- 复用现有 `kd-sensing-train`、fresh eval 和 U-MaskBeamJEPA loss/sampler 字段。
- 生成 next-round es40 配置和 manifest，使每个 run 可单独运行，也能批量调度。
- 支持 P0/P1 分组 launcher，失败继续，最后输出失败列表。
- 汇总 fresh eval 到 per-run、method mean±std、delta-vs-proto 和 filtered Markdown/CSV。
- 用 focused sanity check 防止 seed、epoch、λ、sampler 和 condBTAPA 选择性开关漂移。

**Non-Goals:**
- 不新增模型结构或新 loss。
- 不恢复 weakKD hardall/sensingonly 主线。
- 不修改 proto / BTAPA tau1 baseline 定义。
- 不覆盖已有 es20 配置、输出、日志或 checkpoint。
- 不把本地脚本升级为长期 package CLI。

## Decisions

1. 复用已有配置 overlay 语义。
   - 方案：next-round YAML 继续基于 `configs/scene31/templates/main_v3_proto_es20_base.yaml`，显式覆盖 `training.epochs/max_epochs=40`、seed、sampler 和 condBTAPA 字段。
   - 理由：现有 base 已固定 proto baseline、early stopping、evaluation missing patterns 和输出风格；直接覆盖更小，不触碰 baseline。
   - 替代：新增 es40 base。当前只差 epoch/seed/方法字段，会增加重复配置。

2. 组合方法只组合现有开关。
   - 方案：`missing_pattern_sampler=uniform` 与 `use_pattern_conditional_btapa=true`、`btapa_apply_patterns=[radar_only,lidar_only]` 同时启用，λ 通过 `btapa_lambda` 控制。
   - 理由：loss 中 sampler 与 selective condBTAPA 已解耦；不需要新增 loss 或训练循环分支。
   - 替代：为组合方法复制 loss 分支。会增加 silent divergence 风险。

3. next-round 配置放在独立目录。
   - 方案：使用 `configs/scene31/next_round/` 与独立 manifest。
   - 理由：避免混淆已有 64-run es20 night-grid manifest，同时保留 top-level Scene31 local/manual 边界。
   - 替代：继续写入 `configs/scene31/night_grid/`。容易让 es20 与 es40 manifest 混在一起。

4. 汇总脚本读取 fresh eval CSV，不重新评估模型。
   - 方案：`summarize_scene31_next_round.py` 只消费 `night_grid_metrics.csv`、run dir 中的 `eval_matrix.csv` 或显式 run 目录下的 CSV。
   - 理由：汇总应可重复、只读、快速；fresh eval 仍由现有 eval script 或 launcher 调用。

## Risks / Trade-offs

- [Risk] next-round 配置目录导致 fresh eval 找不到 YAML。→ Mitigation：把 re-eval 配置搜索路径扩展到 `configs/scene31/next_round`，并在 launcher 中传 next-round manifest。
- [Risk] run name 中 λ 与 YAML 实际 λ 不一致。→ Mitigation：新增 sanity check 测试解析 manifest/config 并逐项校验。
- [Risk] `--overwrite` 误覆盖输出。→ Mitigation：launcher 默认 skip 已存在 checkpoint；只有显式 `--overwrite` 才传入 `output.overwrite=true`。
- [Risk] filtered 阈值无人满足导致空表。→ Mitigation：仍输出最接近 top10，并列出未达标条件。
