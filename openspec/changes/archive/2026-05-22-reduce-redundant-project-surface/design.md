## Context

当前仓库已经具备清晰的 `src/kd_sensing` 包边界、canonical config 生成机制、包内 CLI 和架构边界测试，但多轮实验功能沉积后仍存在四类维护表面积：

- 实体配置矩阵：`configs/fusion/` 和部分实验矩阵 YAML 与 virtual/overlay 规则重复。
- 重复入口：部分 `scripts/`、`tools/` 文件只是包内 CLI 或诊断 workflow 的旧路径/fallback。
- 文档沉积：README 承载了大量实验细节，部分 OpenSpec archived spec 仍保留 TBD purpose 或历史迁移叙述。
- 回归缺口：已有架构测试能防止旧 facade 回流，但还没有专门检查“可生成配置重新实体化”“重复入口重新出现”“本地产物进入源码表面积”等问题。

本变更必须遵守现有约束：训练、评估、预处理和诊断的用户可见语义不变；所有验证命令使用 `conda run -n kd_mm_beam ...`；本地数据、输出、cache、checkpoint 和 pycache 不进入源码变更。

## Goals / Non-Goals

**Goals:**

- 将可生成的高级 fusion 配置矩阵迁移到 canonical/overlay recipe，删除重复实体 YAML。
- 删除或明确降级重复脚本入口，推荐路径统一到包内 CLI 和 console scripts。
- 收缩 README 和历史 OpenSpec 沉积，使 README 只保留上手和入口导向，需求契约回到 specs。
- 增加轻量架构/表面积检查，防止同类冗余再次增长。
- 保持训练、评估、预处理、checkpoint registry、final/resolved config 和 console script 行为兼容。

**Non-Goals:**

- 不重写 DeepSense6G、MMW 或 Raymobtime dataset 的样本语义。
- 不改变模型结构、训练 loss、评估指标、checkpoint 格式或 registry 选择逻辑。
- 不删除 `All_models/` 中已跟踪的历史复现权重。
- 不移除 Gradio viewer 本身；只处理重复 manifest wrapper、推荐命令和文档职责。
- 不把研究性一次性分析脚本强行产品化；保留时必须标注边界。

## Decisions

### 1. 先 inventory，再删除

实现前先建立表面积 inventory：统计实体 YAML、脚本入口、README/OpenSpec 待整理项、源码中的重复 wrapper 和本地产物泄漏。删除只针对满足以下条件的对象：

- 有等价包内 CLI 或 virtual/overlay config 替代；
- 已有或新增测试覆盖替代入口；
- 文档中的推荐命令已迁移；
- 不影响 final config 保存完整解析配置。

替代方案是直接按文件大小删除最大文件。该方式风险高，容易误删仍承担唯一功能的研究脚本或实体配置，因此不采用。

### 2. 高级配置采用 overlay-first

G2D、CRAF、MARF、objective 和 ablation 配置优先表达为 recipe/table。实体 YAML 只保留三类：

- 用户直接编辑的真实 base/example 配置；
- 尚无法由 recipe 无损表达的实验配置；
- 为兼容现有已发布命令必须短期保留的路径。

被删除的 YAML 路径必须仍能由配置加载器生成，或必须有明确迁移错误指向新路径。

### 3. 重复入口用生命周期规则管理

当 `tools/` 或 `scripts/` 入口与 `kd_sensing.cli.*` 或 console script 提供同一工作流时，保留包内入口，删除 fallback wrapper。仍保留的脚本必须满足至少一个条件：

- 是真实的开发/研究 workflow，当前没有等价包内 CLI；
- 是薄 wrapper 且已被 spec 明确允许；
- 是 shell orchestration，无法直接作为 Python console script 表达。

### 4. README 变成入口地图

README 保留安装、环境、快速健康检查、主要命令、数据/产物边界和链接。长实验矩阵、分析说明、viewer 操作细节和历史迁移说明迁移到 `docs/` 或 OpenSpec specs。OpenSpec specs 补齐 purpose，并把历史过程叙述归档或收敛为当前行为要求。

### 5. 回归检查以禁止项为主，规模预算为辅

测试应强制拒绝明确错误的增长：`__pycache__`/`.pyc` 被跟踪、重复 CLI wrapper、已删除 fallback 路径、可生成配置重新实体化。README/OpenSpec 行数不适合作为硬性稳定 API，但检查可以输出 inventory 或对明确 TBD/purpose 缺失失败。

## Risks / Trade-offs

- 直接脚本路径删除会影响本地习惯命令 → 在 README/docs 中给出 console script 迁移路径，并用 CLI help 测试保证替代入口可用。
- overlay 生成和旧实体 YAML 可能存在细微字段差异 → 删除实体 YAML 前增加关键字段等价测试，并确认 `final_config.yaml` 保存完整解析结果。
- 文档迁移可能导致信息不易找到 → README 保留链接索引，docs 文件按工作流命名。
- 表面积检查过严会妨碍合理实验扩展 → 检查只拒绝明确禁止项；新增真实 workflow 通过 OpenSpec change 更新 allowlist。

## Migration Plan

1. 建立 inventory 和配置等价测试，确认哪些 YAML/脚本是冗余。
2. 扩展 canonical/overlay recipe，保证目标配置在实体文件删除后仍可加载。
3. 更新 README/docs 推荐命令，删除重复 wrapper 或改为明确研究脚本。
4. 整理 OpenSpec specs：补 purpose，迁移历史叙述，新增表面积回归要求。
5. 运行 focused checks，再运行 OpenSpec strict validation 和必要 pytest。

回滚策略：如果某个删除项发现仍承担唯一功能，恢复该文件并在 tasks 中标记为“暂缓删除”，同时补充迁移条件或 recipe 支持。

## Open Questions

本 proposal 的第一批实施范围先采用保守决策：

- 第一批删除候选聚焦 `configs/fusion/` 中可由现有或新增 overlay recipe 表达的 G2D/CRAF/MARF/ablation 配置；CSI hardening matrix 先只 inventory，不直接删除。
- Gradio Web UI 启动脚本暂不新增 console script，继续保留 `tools/visualization/gradio_multimodal_viewer.py`；本变更只处理 manifest 导出 wrapper 和推荐命令。
- README 收缩后的长实验矩阵先集中到 `docs/experiment_matrix.md`，viewer 细节继续放在 `tools/visualization/README.md` 或后续独立 docs 中。
