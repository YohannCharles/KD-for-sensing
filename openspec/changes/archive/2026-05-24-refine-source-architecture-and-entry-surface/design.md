## Context

项目当前已经完成包结构、轻量导入、训练 runtime 拆分、virtual config 和入口表面积的一轮整理。进一步观察显示，源码体积本身不大，但复杂度集中在少数大文件和矩阵式配置中：

- `tools/visualization/viewer_utils.py` 和 `tools/visualization/gradio_multimodal_viewer.py` 同时承担 manifest 读取、过滤、图表、prediction summary 和 Gradio 编排。
- `src/kd_sensing/preprocessing/raymobtime_s008.py` 同时承担路径解析、文件审计、index 构建、beam label normalization、ray feature 提取和 cache 写出。
- `src/kd_sensing/diagnostics/complementarity.py` 同时承担 schema adapter、case mining、summary 和写出。
- `src/kd_sensing/models/csi.py` 同时承担 pilot estimation、CSI hardening、view tokenization、fusion 和 encoder 注册。
- `configs/` 中仍保留若干高级 fusion / CRAF / MARF / CSI 组合实体 YAML，需要继续判断哪些可由 recipe 表达。
- `scripts/`、`tools/analysis/`、`tools/visualization/` 已有 allowlist，但生命周期和新增规则还可以更硬。

本变更只处理源码、配置和入口表面积，不清理、不移动、不压缩真实数据或本地实验产物。

## Goals / Non-Goals

**Goals:**

- 将当前高复杂度文件拆成职责明确的窄模块，并保留现有公开入口和用户命令。
- 为配置二次瘦身建立候选分类、等价检查和实体 YAML 删除规则。
- 明确入口生命周期，防止 `scripts/` 和 `tools/` 重新积累重复 wrapper。
- 增强 `tests/test_architecture_boundaries.py` 或等价快速检查，使架构回归能在不读真实数据、不加载 checkpoint、不启动训练的情况下暴露。
- 更新文档 inventory，使后续新增配置或入口必须说明理由。

**Non-Goals:**

- 不删除、迁移或压缩 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或下载压缩包。
- 不改变默认数据目录、训练语义、评估指标、checkpoint 选择策略或 manifest 输出语义。
- 不引入新第三方依赖。
- 不把研究脚本强行改造成包内 CLI，除非它已经覆盖核心公共 workflow。

## Decisions

### 1. 先建立表面积 inventory，再做删除和拆分

实现前先更新 `docs/project_surface_inventory.md` 或等价检查数据，记录当前大文件、配置 YAML、入口脚本和保留原因。删除实体 YAML 或入口 wrapper 时，必须能指向 inventory 中的分类和替代路径。

备选方案是直接拆文件和删配置，但这样难以解释为什么某些研究脚本继续保留、某些高级 YAML 暂时不动。先 inventory 的成本低，能让后续批次更稳。

### 2. 大文件按领域拆，不做跨领域“公共工具箱”

拆分目标按现有模块边界走：

```text
viewer_utils.py
  ├─ manifest_io / filters
  ├─ figures
  ├─ prediction_tables
  └─ public compatibility imports

raymobtime_s008.py
  ├─ paths / audit
  ├─ index
  ├─ beam_labels
  ├─ ray_features
  └─ cache writer / preprocessor registry

complementarity.py
  ├─ schema
  ├─ cases
  ├─ summaries
  └─ writers

models/csi.py
  ├─ estimation
  ├─ hardening
  ├─ tokenizers
  └─ encoders / registry glue
```

备选方案是新建一个 `common/` 或 `helpers/` 聚合层。该方案容易形成新的兼容聚合层，与项目既有窄模块方向冲突，因此不采用。

### 3. 公开入口保持，内部迁移到窄模块

用户可见入口继续存在：console scripts、包内 CLI、`gradio_multimodal_viewer.py`、已允许的研究/诊断脚本和数据准备脚本。内部实现可以迁移到窄模块，旧文件如需保留应只做薄协调或兼容导出。

备选方案是大规模重命名入口，但会打断 README、实验脚本和用户习惯，不符合本次“优化而不破坏 workflow”的目标。

### 4. 配置二次瘦身采用“recipe 等价优先”

高级 fusion、CRAF、MARF、CSI/GPS/mmWave 组合实体 YAML 先分三类：

- 可由 recipe 无损生成：补测试后删除实体 YAML。
- 可由 recipe 生成但有显式差异：把差异转成 overlay option 或保留并记录原因。
- 实验草案或人工样例：保留，但必须标注生命周期和不能 recipe 化的字段。

删除前的等价检查至少覆盖 experiment、task、dataset type、enabled modalities、model type、loss/distillation、training、output run name 和 checkpoint 来源。

### 5. 架构检查要检查“新增表面积”，不检查本地产物

快速检查继续只针对已跟踪源码、配置、文档和 OpenSpec artifact。它应拒绝重复入口、可生成 YAML 重新实体化、未说明的 allowlist 扩张和大文件职责回流；它不扫描或清理本地 `dataset/`、`outputs/`、`logs/`。

## Risks / Trade-offs

- 大文件拆分可能造成 import 循环 → 先拆纯函数/纯数据 helper，再迁移重依赖函数；每批跑轻量导入边界测试。
- 配置 recipe 化可能漏掉实体 YAML 的实验差异 → 删除前加入关键字段等价测试，并在设计或测试中显式记录允许差异。
- 入口删除可能影响本地脚本习惯 → 只删除已有 console script 或包内 CLI 覆盖的重复 wrapper，README 继续指向稳定入口。
- 过多架构检查可能束缚研究脚本 → 检查允许研究/诊断脚本保留，但要求 lifecycle 分类和 OpenSpec 说明。
- 拆分过程中可能只移动代码而没有降低复杂度 → 每个拆分批次都以“修改某一职责无需触碰无关模块”为验收条件。

## Migration Plan

1. 更新表面积 inventory 和架构边界测试基线，记录当前 YAML 数量、脚本 allowlist 和待拆大文件。
2. 拆 `tools/visualization/viewer_utils.py`，保持 Gradio viewer 行为和测试不变。
3. 拆 `src/kd_sensing/preprocessing/raymobtime_s008.py`，保持 preprocessor registry、CLI 和 cache 文件语义不变。
4. 拆 `src/kd_sensing/diagnostics/complementarity.py` 与 `src/kd_sensing/models/csi.py` 中职责最清楚的部分。
5. 建立高级配置 recipe 候选清单，逐批增加等价测试，再删除可生成实体 YAML。
6. 清理或重分类重复入口，更新 README/docs 中推荐命令。
7. 运行快速架构检查、相关单元测试、CLI help smoke、`openspec validate refine-source-architecture-and-entry-surface --strict`，最后运行全量测试。

如果某批拆分出现风险，可以保留原公开文件作为薄 facade 回退；回退不需要恢复数据或产物，因为本变更不修改这些内容。

## Open Questions

- CSI 模型拆分是否先只拆 `CSIHardening` 和 pilot estimation，还是同步拆 encoder 注册？建议实施时先从 hardening / estimation 开始。
- CRAF/MARF 实体 YAML 中哪些差异应转成 overlay option，哪些应继续作为人工样例保留？建议由等价检查先输出差异表再决定。
