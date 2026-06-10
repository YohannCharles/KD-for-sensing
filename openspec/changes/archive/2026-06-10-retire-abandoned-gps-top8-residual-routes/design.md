## Context

仓库当前把 Top8 selector、GPS coarse anchor、GPS prior 后接 residual delta correction 和 camera residual 写成了当前支持能力：它们有 console scripts、配置、engine、data/model/loss 模块、README 工作流、OpenSpec specs 和 focused tests。用户已经确认这条差值学习论文路线不可行，主要问题是 GPS 粗预测精度上限过低，后续模态学习差值/残差很难稳定提升。用户随后明确要求 BGAM 模块保留，因此 BGAM 不属于本次删除范围。

本变更的目标不是修补这些方法，而是把它们从当前支持面中移除。由于用户明确“不管兼容”，实现时不保留旧入口包装层、不做配置迁移、不新增 migration guard stub。

## Goals / Non-Goals

**Goals:**

- 删除 Top8 candidate selector 训练/plot/compare、GPS coarse anchor、residual fusion、camera residual 和 geometry residual delta 相关源码、配置、CLI 和测试。
- 同步更新 README、docs、OpenSpec 当前 specs、`pyproject.toml` 和架构边界测试，使退役路线不再出现在 quickstart、推荐实验矩阵或安装入口中。
- 保留仍属于当前主线的普通 beam prediction 能力，包括 GPS-Rel-Polar、GPS-only v2/adapter、MMW GPS v2、DeepSense6G/MMW GPS+LiDAR BGAM、BGAM 依赖的 TopK candidate manifest/loss 支撑代码、CSI hardening、Raymobtime、JEPA、viewer manifest、通用 Top-K 指标和 circular metrics。
- 明确源码清理不自动删除本地实验产物；`outputs/`、`logs/`、cache 和 checkpoint 后续若要删除，需单独走 manifest。

**Non-Goals:**

- 不重新设计新的多模态残差路线、TopK fallback、BGAM 替代方案或论文实验矩阵。
- 不清理 dataset、All_models、历史权重或本地 outputs。
- 不删除通用字符串命中项，例如普通 evaluation `topk` 指标、viewer prediction top-k 展示、GPS v2 residual diagnostic plot、CSI sweep 中的候选排序统计。
- 不为了旧命令体验保留 alias、stub CLI、兼容配置或 registry fallback。

## Decisions

### Decision 1: 按研究路线删除，而不是按关键词删除

删除范围以语义归属为准：Top8 selector 训练/plot/compare、GPS coarse anchor、GPS anchored residual/camera residual 和 geometry residual delta。BGAM 模块、BGAM 依赖的 TopK candidate manifest/loss 支撑代码、普通 `topk` metric、GPS v2 自身诊断里的 signed residual 统计、CSI hardening candidate ranking 不属于本次删除范围。

备选方案是按 `topk|residual|candidate` 全仓库扫删。这里不采用，因为会误删通用评估指标、诊断 helper 和其它保留 workflow。

### Decision 2: 不保留兼容入口

从 `pyproject.toml` 直接移除旧 console scripts，并删除对应 CLI 文件、默认配置和 tests。旧命令在 editable install 刷新后应表现为不存在，而不是打印“已退役”后退出。

备选方案是保留 thin CLI guard。这里不采用，因为用户明确“不管兼容”，并且 guard 仍会占据文档、测试和入口维护面。

### Decision 3: 先断公开入口，再删内部实现

实现顺序应先移除 README/docs/pyproject/configs/scripts 中的公开路径，再删除 `src/kd_sensing` 内部模块和测试，最后跑引用扫描和 focused validation。这样如果某个内部 helper 仍被保留主线使用，会在导入或测试阶段暴露。

备选方案是先删源码再修引用。这里不采用，因为跨模块引用较多，先断入口更容易识别真正仍被依赖的保留代码。

### Decision 4: OpenSpec 归档后保留“已退役”事实

每个退役 capability 的 delta spec 应新增“已退役” requirement，并移除原有正向运行要求。这样归档后当前 specs 会解释为什么这些路径不再存在，而不是让读者误以为 OpenSpec 丢失了背景。

备选方案是完全删除 capability spec。这里不采用，因为项目已经多次通过 specs 记录研究线退役状态，保留退役事实有助于防止同一路线被误恢复。

### Decision 5: Camera AE 与 camera residual 分开判断

`deepsense6g-camera-ae-residual-correction` 中的 residual workflow 必须退役；Camera AE 训练/特征导出如果仅服务 camera residual，也随之删除。如果实现中存在被其它当前主线引用的通用 image AE 或 Camera AE 能力，应把它迁到非 residual capability 后再保留。

备选方案是保留 Camera AE 作为“未来可能有用”的孤立入口。这里不采用，除非实现期发现它已经是其它保留 workflow 的实际依赖。

## Risks / Trade-offs

- [Risk] 误删通用 `topk` 或 circular metric，导致普通评估、viewer 或 CSI 工作流回归。→ Mitigation: tasks 中要求按模块和引用确认，保留 `evaluation`、viewer、CSI 中的普通 Top-K 指标；验证运行架构边界和核心 focused tests。
- [Risk] 删除 GPS v2 logits export 会影响 GPS v2 自身诊断或 BGAM 输入。→ Mitigation: 只删除 residual/coarse anchor downstream 契约；GPS v2 自身诊断和 BGAM 需要的 logits export 保留。
- [Risk] README/docs 中旧路线引用较多，容易漏掉。→ Mitigation: 使用引用扫描覆盖 README、docs、OpenSpec、configs、scripts、tests、pyproject 和 `src/kd_sensing`。
- [Risk] 本地 outputs 仍包含旧结果，用户可能以为源码清理会删除实验产物。→ Mitigation: 文档和最终报告明确本 change 不自动清理本地产物；后续需要时走 runtime cleanup manifest。
- [Risk] active change `cleanup-project-surface-drift` 也在处理表面积，可能产生文档或 guardrail 冲突。→ Mitigation: 实现时优先复核两个 active changes 的 touched files；若同一文件有新改动，保留用户/其它 change 的最新内容并合并语义。

## Migration Plan

1. 更新 OpenSpec delta specs，把退役路线从当前能力改为已退役能力。
2. 从 `pyproject.toml` 移除退役 console scripts，删除对应包内 CLI 文件和配置文件；保留 BGAM console scripts/configs。
3. 删除退役路线专属 `data`、`engine`、`models`、`losses` 模块及导出/注册引用；保留 BGAM 与其必要 TopK candidate 支撑模块。
4. 删除或改写专属 tests；更新 `tests/test_architecture_boundaries.py`，使它断言这些入口已不存在，而不是仍在包内。
5. 更新 README、README_REPRODUCE、docs inventory 和实验矩阵，当前推荐 workflow 只保留 GPS v2、supervised/adaptation、CSI、Raymobtime、JEPA、viewer 等仍支持路线。
6. 运行 `openspec validate retire-abandoned-gps-top8-residual-routes --strict`、引用扫描、架构边界测试和必要的核心 smoke。

回滚策略：如果实现中发现某个模块被保留主线实际依赖，先从删除列表移出并记录保留原因；不恢复已确认退役的公开入口。若整批清理需要暂停，可保留 OpenSpec proposal/design/tasks 作为未应用方案，不做源码删除。

## Open Questions

- Camera AE 训练/特征导出是否还有非 residual 当前用途；实现时以引用和 docs 为准。
- GPS v2 logits export 是否需要迁入 GPS v2 capability 保留为自身诊断，而不是随 GPS coarse anchor spec 一起退役。
- `docs/target_shot_geometry_residual.md` 和 distribution-shift 中的 residual 诊断是否仅服务退役路线；若仍服务当前 split/label-space 分析，应保留或重命名为非 residual 路线。
