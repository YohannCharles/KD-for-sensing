## Context

当前仓库的核心训练、数据、模型和诊断能力已经迁入 `src/kd_sensing` 下的 owner 模块，并通过 OpenSpec、inventory 和架构边界测试约束旧 KD/Hist/Top8/BGAM/viewer 等路线不回流。最近一次审计仍发现几类低价值表面：

- `src/kd_sensing/diagnostics/cnn_hybrid_jepa_visual_prior_sweep.py` 是旧 full sweep 生成器，约 2.7k 行；当前更小的 `jepa_visual_architecture_sweep.py` 已承载推荐 architecture sweep schema。
- `src/kd_sensing/engine/loso_data.py` 提供 public dataloader builder，但 CodeGraph 未发现内部调用；保留价值主要来自旧 supporting 契约。
- `scripts/run_next_v3_experiments.sh`、`scripts/run_rbma_strong_encoder_4gpu_queue.sh`、`scripts/run_m2beam_single_modal_scene31_queue.sh` 和 `scripts/run_rbma_missing_workflow.py` 是本地/manual 运行面，和 `configs/scene31/`、RBMA strong encoder、M2Beam 单模态 overlay 强耦合。
- BeamBench spec 仍硬编码已删除的 `kd_sensing.cli.beambench_check_dataset` 场景，和 inventory 中“包内旧 wrapper 已删除”的事实冲突。
- 工作区存在大量 ignored `__pycache__` / `.pyc` 噪声；这些不属于源码，但会干扰人工审计。

本 change 是清理和收口，不是模型改进。实现必须避免移动或删除真实 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 和历史权重。

## Goals / Non-Goals

**Goals:**

- 将旧 full sweep、孤岛 LOSO helper、本地 queue 脚本/overlay 和过期 spec 引用分类收敛。
- 尽量删除代码，而不是新增抽象层；确需兼容时只保留薄 reader 或入口提示。
- 让 current docs、OpenSpec、inventory、pyproject、真实路径和架构边界测试重新一致。
- 为每个删除/保留候选留下验证命令和回滚条件。
- 清除 ignored bytecode/cache 噪声，但不把本地实验产物纳入源码清理。

**Non-Goals:**

- 不重构 `DeepSense6GDataset`、`MMWDataset`、`ModularSequenceModel`、`engine.batch`、训练主循环或模型 forward。
- 不改变训练数值语义、split 语义、beam label 口径、checkpoint schema、默认输出目录或当前模型 registry。
- 不新增长期通用 runner 框架；如果本地脚本需要替代，优先使用已有 `kd-sensing-train` 与小型 manifest/文档命令。
- 不删除 `outputs/`、`logs/`、`dataset/`、checkpoint、cache 或历史权重。

## Decisions

### 1. 先修支持面漂移，再删源码

先同步 BeamBench spec、inventory、架构边界测试和文档中的过期入口引用，再删除旧模块。这样删除失败时不会留下“spec 仍要求旧入口”的半迁移状态。

替代方案是先删代码再补文档。拒绝原因：这会让验证红点混杂，难以区分行为破坏和治理漂移。

### 2. 旧 CNN/hybrid full sweep 默认退役，必要时保留薄兼容 reader

当前推荐 sweep 是 `jepa_visual_architecture_sweep.py` 和 `configs/diagnostics/jepa_visual_architecture_sweep_manifest.yaml`。旧 full sweep 如果只被测试、模型摘要 CLI 或旧文档消费，应迁移到当前 manifest reader；如果仍有历史报告需要读取旧 manifest，可保留不执行训练的轻量 `load_full_sweep_manifest` 兼容 reader，并删除 runner、job graph、shell 生成和 cleanup 逻辑。

替代方案是拆分旧 2.7k 行模块。拒绝原因：它代表过期实验矩阵，继续拆分会把低价值路线固化成更多文件。

### 3. LOSO 保留数据规划语义，不保留无调用 engine dataloader facade

`cross-scene-loso-workflow` 仍可作为 supporting capability，保留 fold planning、target adapt/test 和 few-shot sampling 的需求；但这不要求 `engine/loso_data.py` 的 builder public surface 长期存在。若当前源码、CLI、tests、docs 和 specs 均可迁到 `kd_sensing.data.loso` 或声明未来 workflow 另起 change，则删除 `engine/loso_data.py`。

替代方案是把 `loso_data.py` 继续登记为 monitor。拒绝原因：CodeGraph 已显示 public builder 无内部调用，继续保留会误导后续 workflow 复用未维护入口。

### 4. Scene31/RBMA 本地运行面降级为 local/manual 或收进一个最小 runner

固定 GPU shell 和 overlay YAML 的生命周期必须明确：

```text
local/manual configs
      │
      ├── 仍需跑：登记 owner、输出边界、删除触发条件
      ├── 可由同一 runner 覆盖：合并配置列表，删除 shell
      └── 结论已沉淀：删除或归档为历史说明
```

优先删除固定 GPU shell；如果还需要批量运行，复用 `scripts/run_rbma_missing_workflow.py` 的有界并发思路或直接用文档命令。不要新增长期 package CLI。

### 5. 本地 bytecode 只作为工作区清理，不作为源码任务

删除 `__pycache__` 和 `.pyc` 可以直接执行，因为它们被忽略且不属于源码；但 tasks 必须把它和源码删除分开记录。架构边界测试继续只拒绝这些路径被 git 跟踪，不因为本地存在而失败。

## Risks / Trade-offs

- 旧 full sweep 仍被私人脚本调用 → 保留一个只读兼容 reader 或在最终说明给出当前 manifest 替代命令。
- 删除 `engine/loso_data.py` 影响未登记外部用法 → 删除前检查 README/docs/OpenSpec/pyproject/tests/CodeGraph；若无法确认，先标记退役并保留 deprecation stub。
- 本地 Scene31/RBMA 配置仍在跑 → 不批量删除所有 overlay；先按“当前需要运行/可合并/已沉淀”三类处理。
- 规格 delta 太宽 → 只改与旧 sweep、LOSO、local/manual surface 和 BeamBench 入口漂移相关的 requirement，避免重写核心 workflow。
- 验证耗时扩大 → 每个 wave 只跑 focused tests，最后再跑架构边界和 OpenSpec strict validate。

## Migration Plan

1. Wave 0：修复 BeamBench spec 漂移、inventory 分类和架构边界预期；清理 ignored bytecode。
2. Wave 1：迁移模型架构摘要和测试对旧 full sweep manifest 的依赖；删除或降级旧 full sweep runner。
3. Wave 2：确认并删除或退役 `engine/loso_data.py`；保留 `data/loso.py` 的 fold/few-shot 支撑语义。
4. Wave 3：收敛 Scene31/RBMA/M2Beam 本地脚本和 overlay；删除固定 GPU shell 或把配置列表合并到一个 local/manual runner。
5. Wave 4：更新 docs、OpenSpec lifecycle/inventory 和验证命令；运行 strict validate 与 focused tests。

回滚策略：每个 wave 都应是可单独 revert 的小提交或小 patch。若旧 sweep/LOSO 删除暴露外部依赖，优先恢复薄兼容 reader/stub，而不是恢复完整旧 runner。

## Open Questions

- 旧 `cnn_hybrid_jepa_visual_prior_sweep` 是否仍有正在运行的本地实验需要继续生成 job graph？若有，本 change 只删除执行 runner，保留 manifest expansion reader。
- `configs/scene31/` 和 strong-encoder RBMA overlay 是否还有未完成实验？若有，先登记 local/manual 和删除触发条件，不立即删除全部 YAML。
- `engine/loso_data.py` 是否被仓库外脚本直接 import？仓库内无调用，但若用户确认外部使用，应改为 deprecation stub。
