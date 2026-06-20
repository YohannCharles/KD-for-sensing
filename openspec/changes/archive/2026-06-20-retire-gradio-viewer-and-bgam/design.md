## Context

当前项目已经有多轮退役和 surface 收口：仓库级 Gradio viewer 在现行规格中已不再作为 Web UI 维护，但 viewer manifest 导出、`kd-sensing-visualize-modalities` alias、viewer manifest helper 和对应测试仍是 current 诊断支持面。BGAM 仍更深地嵌入 current 主线：DeepSense6G/MMW BGAM specs、GPS pseudo-history BGAM、配置、CLI、模型/loss/dataset/engine、README、实验文档、维护索引和架构边界测试都把它视为保留 workflow。

用户已明确这些内容“永远用不到了”，因此本变更按无兼容迁移的退役处理。实施时需要同时修改源码、配置、测试、README、OpenSpec current specs、`docs/maintainer_context_index.yaml` 和 `docs/project_surface_inventory.md`，否则会出现 spec 仍要求入口存在、测试仍 allowlist 入口、README 仍推荐命令的漂移。

## Goals / Non-Goals

**Goals:**

- 删除 BGAM 和 viewer manifest/Gradio viewer 相关的当前源码支持面、安装入口、配置、测试和推荐文档。
- 将 BGAM、GPS pseudo-history BGAM、BGAM-only TopK 支撑、viewer manifest 导出从 current/supporting 叙事改为 retired-tombstone 或完全移出当前要求。
- 更新架构边界和维护索引，使已退役入口缺失成为期望状态，而不是测试失败。
- 保留 archive 历史记录和本地运行产物边界，不把历史输出或 checkpoint 纳入源码变更。

**Non-Goals:**

- 不清理用户本地 `dataset/`、`outputs/`、`logs/`、cache 或 checkpoint；如需删除本地产物，应另行生成清理 manifest。
- 不重写 JEPA visual analysis、MMW GPS v2、CSI hardening、通用训练/评估/预处理、run index 或 cleanup workflow。
- 不提供 BGAM 或 viewer manifest 的兼容 stub、迁移 CLI、虚拟配置或 registry fallback。
- 不改写 OpenSpec archive 中的历史 change 内容；archive 只作为历史背景保留。

## Decisions

### 1. 退役采用 hard removal，而不是 deprecation stub

BGAM 和 viewer manifest 入口将从 `pyproject.toml`、README、CLI help tests 和 architecture allowlist 中移除，专属源码模块、配置和 focused tests 也随之删除或改写。旧命令不应继续以薄 alias 或“提示已退役”的 stub 形式安装。

备选方案是保留 stub CLI 输出迁移提示。该方案会继续污染安装入口和 CLI help 矩阵，而且用户已经明确永远不用，因此不保留兼容层。

### 2. 通用能力按语义保留，专属 BGAM/viewer 命名边界删除

实现清理时不能只按字符串删除。普通 Top-K metrics、circular metrics、GPS v2 logits、LiDAR preprocessing、JEPA visual analysis 和通用 dataset/cache 工具如果仍被当前 workflow 使用，应保留并迁出 BGAM-only 引用；只有 BGAM 专属 manifest、dataset、model、loss、engine、debug mask、CLI/config/test 和 viewer manifest 专属导出链路应删除。

### 3. OpenSpec 先收敛 current 叙事，再实施源码

本 change 的 delta specs 将删除 BGAM current requirements，并修改 project architecture、experiment workflow、visual diagnostics、mainline docs 和 lifecycle 边界。实施阶段先让 specs 与 inventory/README 的目标支持面一致，再删除源码和测试，避免一边删入口一边被 current spec 要求恢复。

### 4. 维护索引和架构边界测试必须同步

`docs/maintainer_context_index.yaml` 中的 entrypoint owner metadata、hotspots、routing、validation commands 和 retired-route guard 需要同步；`tests/test_architecture_boundaries.py`、`tests/test_cli_help.py` 和相关 characterization tests 需要从“要求 BGAM/viewer 可用”改成“拒绝它们回流”。这一步是防止未来误把删除当回归的关键。

### 5. Archive 与本地产物不参与删除

`openspec/changes/archive/` 中的 Gradio/BGAM 历史 change 不改写。本地 `outputs/analysis/*bgam*`、viewer cache、dataset manifest、checkpoint 等运行产物不在本 change 中删除；最终说明只提示它们是本地产物，除非用户另开清理任务。

## Risks / Trade-offs

- [Risk] 某些当前 JEPA visual analysis 或离线诊断代码间接复用了 viewer manifest helper。→ Mitigation: 实施前用 import/call graph 和 focused tests 确认真实依赖；必要时把通用 JSON/asset 写出 helper 改名迁入非 viewer 模块，再删除 viewer 命名入口。
- [Risk] TopK/candidate helper 既有 BGAM 支撑语义又被 MMW GPS v2 或其它保留 workflow 消费。→ Mitigation: 删除前按 owner 和调用方分类，保留通用指标/候选处理，移除 BGAM-only manifest enrich 和 CLI。
- [Risk] README、inventory、OpenSpec 和测试不同步会导致 agent 未来误恢复入口。→ Mitigation: 把文档、索引、spec delta 和 architecture tests 放在同一 implementation wave，并运行 OpenSpec validate 与架构边界测试。
- [Risk] 删除大量测试后覆盖面短期下降。→ Mitigation: 用保留 workflow 的 CLI help、config load、JEPA visual analysis、architecture boundaries 和全量 pytest 替代 BGAM/viewer focused tests。

## Migration Plan

1. 更新 OpenSpec current specs 和 lifecycle inventory，把 BGAM 与 viewer manifest 从 current/supporting 支持面移除。
2. 删除 BGAM/viewer manifest console scripts、CLI 模块、专属配置、源码模块和 focused tests。
3. 更新 README、docs、maintainer context index、project surface inventory、mainline catalog/protocol/claim 文档中的入口、状态和验证命令。
4. 调整架构边界、CLI help、config load 和诊断相关测试，使退役入口不存在成为预期。
5. 运行 `openspec validate retire-gradio-viewer-and-bgam --strict`、`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`、`conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`，最终按风险运行 `conda run -n kd_mm_beam pytest -q`。

## Open Questions

- `kd_sensing.diagnostics.viewer_manifest_*` 中是否有少量通用 asset/statistics helper 被 JEPA visual analysis 复用；若有，实施时应先迁入非 viewer 命名模块再删除原模块。
- DeepSense6G/MMW TopK candidate selector specs 中剩余 supporting 语义是否仍被非 BGAM workflow 消费；若没有，应随 BGAM 一并退役。
