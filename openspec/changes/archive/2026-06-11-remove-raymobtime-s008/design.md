## Context

Raymobtime s008 当前是一个横切工作流：它有 `raymobtime_s008` dataset type、专用预处理器、专用 selection 模型、`coord/ray` 模态扩展、配置验证规则、run metadata 分支、配置文件、文档、focused tests 和本地数据/产物约定。删除它不是单点删文件，而是一次退役数据集家族子流程的变更。

本仓库已有源码与本地产物边界：`dataset/` 是本地数据输入，`outputs/`、`logs/`、cache 和 checkpoint 是运行产物，不能把清理动作扩大到非目标数据集或源码外的未知路径。用户明确要求删除 Raymobtime s008 相关代码和数据集，因此本变更允许清理 `dataset/Raymobtime/s008` 和 Raymobtime s008 专属运行产物，但必须先生成可审计清单。

## Goals / Non-Goals

**Goals:**

- 删除 Raymobtime s008 源码实现、注册、配置、文档、测试和 OpenSpec 支持契约。
- 删除或退役 `raymobtime_s008` dataset/preprocessor/model/objective 配置路径，使旧配置快速失败并说明该工作流已退役。
- 删除 Raymobtime s008 本地数据目录和专属运行产物时，先生成 machine-readable manifest，限定匹配路径、原因、类型和大小。
- 保持 DeepSense6G、MMW、CSI、viewer、通用训练/评估/预处理 CLI 和轻量导入边界可用。

**Non-Goals:**

- 不迁移 Raymobtime s008 到新的数据集抽象，不保留兼容 wrapper，不提供替代 Raymobtime workflow。
- 不删除 Raymobtime 以外的数据集、`All_models/` 已跟踪权重、OpenSpec artifacts 或其它实验输出。
- 不重新设计训练循环、指标系统、模态契约或 dataset descriptor 的通用架构。

## Decisions

1. Raymobtime s008 采用硬退役，不保留兼容入口。
   - 选择：删除实现文件、注册导入、配置和文档入口；旧 `raymobtime_s008` 配置由 migration guard 或配置校验快速失败。
   - 理由：用户要求删除所有相关代码和数据集，保留兼容入口会继续扩大测试面并违背退役目标。
   - 替代方案：保留薄 wrapper 并提示迁移。该方案仍要求维护 registry、配置解析和测试，不适合本次清理。

2. 数据清理通过 manifest 驱动。
   - 选择：实现前先生成候选清单，覆盖 `dataset/Raymobtime/s008`、`outputs/raymobtime_s008`、Raymobtime s008 logs/cache/checkpoint/diagnostic 路径；确认候选只属于 Raymobtime s008 后再删除。
   - 理由：真实数据和运行产物可能很大且不可恢复，需要可审计边界。
   - 替代方案：直接删除匹配目录。该方案风险过高，可能误删外部 data_root 或非目标实验。

3. `coord` 和 `ray` 作为 Raymobtime 专属模态扩展一并删除。
   - 选择：从中心化模态契约和 batch 准备中移除 `coord/ray` 支持；若其它保留 workflow 没有依赖，则测试应拒绝这些模态。
   - 理由：现有 `coord/ray` 契约由 Raymobtime s008 引入，退役后继续保留会制造无归属接口。
   - 替代方案：保留通用 `coord/ray` 实验模态。当前没有非 Raymobtime 需求，暂不保留。

4. 文档与 OpenSpec 同步收敛。
   - 选择：README、实验矩阵、研究笔记、项目表面清单、现有 spec delta 同步移除 Raymobtime s008 当前支持表述。
   - 理由：否则实现删除后文档仍会推荐不可运行 workflow。
   - 替代方案：只删源码。该方案会留下错误的用户入口和测试建议。

## Risks / Trade-offs

- [Risk] 删除 Raymobtime 专用代码时遗漏 registry、lazy export、egg-info 或配置引用，导致导入/安装检查失败。→ Mitigation：使用引用扫描和架构边界测试，拒绝 `raymobtime_s008` 运行时导入残留。
- [Risk] `coord/ray` 模态被未来其它实验暗中复用。→ Mitigation：删除前扫描引用；如发现非 Raymobtime 依赖，先拆出独立 OpenSpec 需求，否则一并退役。
- [Risk] 数据清理误删用户外部 `data_root` 或非 Raymobtime 产物。→ Mitigation：只清理项目内明确匹配路径，先写 manifest，记录大小和原因；不跟随 symlink 删除外部位置。
- [Risk] 删除 focused tests 后覆盖下降。→ Mitigation：补充退役行为测试、配置拒绝测试和引用扫描测试，并保留通用 CLI/help 与非 Raymobtime workflow smoke。

## Migration Plan

1. 生成 Raymobtime s008 候选清理 manifest，列出源码、配置、文档、测试、本地数据和运行产物候选。
2. 删除 Raymobtime s008 源码实现和注册入口：dataset、preprocessing、model、config rule、run metadata 分支、layout/descriptor、modalities。
3. 删除 Raymobtime s008 配置、文档和测试引用；更新 README、实验矩阵、研究笔记和项目表面清单。
4. 增加退役 guard 和测试，确保旧 `raymobtime_s008` 配置快速失败，错误信息指出已退役且不提供兼容迁移。
5. 按 manifest 清理 `dataset/Raymobtime/s008` 与 Raymobtime s008 专属 `outputs/`、`logs/`、cache/checkpoint/diagnostic 产物；跳过不存在路径和非目标路径。
6. 运行 focused 验证：OpenSpec strict 校验、架构边界测试、CLI help、配置/注册退役测试，以及最终 `conda run -n kd_mm_beam pytest -q` 或记录无法运行原因。

Rollback 策略：源码删除可通过版本控制回滚；本地数据和运行产物删除不可可靠恢复，因此清理前 manifest 是必须产物，实际删除仅限用户明确要求的 Raymobtime s008 路径。

## Open Questions

- `coord/ray` 模态是否存在非 Raymobtime 的未提交实验依赖？实现前以引用扫描为准；若存在，需要先拆分独立保留需求。
- 是否要删除整个 `dataset/Raymobtime/` 家族目录，还是只删除 `dataset/Raymobtime/s008`？本设计默认只删除 s008，避免影响未来或外部未纳入源码的 Raymobtime 场景。
