## Context

Scene31 funnel 当前是 manifest-backed local/manual workflow：配置由 generator 生成，训练走 `kd-sensing-train --config <generated-yaml>`，fresh eval 复用 `scripts/reevaluate_apples_to_apples.py`，summary 写入 ignored output root。

本次只处理 PatternFiLM d8 quick-screen 的后续补跑。现状约束是 seed1 不能直接作为论文结论，fresh eval 缺 miss2 pattern 行，且不能继续扩 JTT、MVFR、MP-DRO、beamsoft、condBTAPA、weakKD 或新 transformer/imputation 方向。

## Goals / Non-Goals

**Goals:**

- 让 PatternFiLM d8 seed1-5 能通过同一 Scene31 funnel 生成逻辑、训练 runner 和 fresh eval runner 管理。
- fresh eval 默认包含缺两个模态的 pattern，并将 mask 语义写入 `missing_bucket_mapping.json`。
- summary 输出 miss1/miss2/miss3、beam proximity、delta vs uniform 和保守晋级结论。
- 对 PatternFiLM d8 的 dim、identity init、sampler、epoch、禁用其它方法线等做 sanity check。

**Non-Goals:**

- 不重新设计 missing-modality 主线。
- 不新增 package CLI。
- 不训练或评估 JTT、MVFR、MP-DRO、beamsoft、condBTAPA、weakKD、AMBER、transformer 或 imputation 新方向。
- 不删除或覆盖旧 checkpoint；需要重评时只通过显式 overwrite eval 控制 fresh eval 输出。

## Decisions

- **复用 Scene31 funnel generator，而不是手写 seed YAML。** 这样 seed2-5 与 seed1 只差 seed，run name、output root、epoch、sampler 和 config layout 继续由同一个 manifest 负责。
- **miss2 pattern 使用现有 `missing_<a>_<b>` 语义。** 当前 mask helper 已支持 `missing_gps_radar` 这类名字，表示缺 gps 与 radar、保留其它模态；fresh eval 只需让这些 pattern 进入评估列表，避免新增命名解析。
- **PatternFiLM 放在现有 `modular_sequence` 内的 pre-head 路径。** 配置字段已经位于 `model.primary.pattern_film`，实现应只消费 availability mask，不读取标签、loss 或其它评估指标。
- **新增专用薄 runner/summary。** `scripts/run_scene31_patternfilm_d8.sh` 只负责本次 train/eval group 和 GPU 队列；业务逻辑仍调用 `kd-sensing-train` 与 `reevaluate_apples_to_apples.py`。`scripts/summarize_scene31_patternfilm_d8.py` 复用现有 summary helper，再补 PatternFiLM d8 结论。

## Risks / Trade-offs

- PatternFiLM seed1 若历史 checkpoint 未真正消费 `pattern_film`，新实现会使 seed2-5 与旧 seed1 不完全同构 → mitigation：最终 sanity check 明确报告 seed1 配置/实现 caveat，不重写 seed1 checkpoint。
- 重评旧 fresh eval 会覆盖同名 fresh eval 目录 → mitigation：runner 默认跳过已有 ok eval，只有 `--overwrite-eval` 时重跑。
- uniform reference seeds 分散在多个 root → mitigation：summary/runner 支持 `--extra-root`，找不到完整 n=5 时 warning 并 fallback 到固定 reference。
