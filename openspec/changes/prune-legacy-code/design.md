## Context

当前仓库中 MARF、CRAF、G2D 和 Multimodal-NF 已从“实验能力”沉积成跨模型、训练、配置、数据集、诊断、吞吐 profile、文档和测试的长期维护面。它们不仅有源码模块和测试，还在 OpenSpec active specs 中保留了大量正向承诺，例如 CRAF/MARF 架构注册、G2D distiller/SMP/diagnostics、Multimodal-NF HDF5/index/cache/codebook runtime，以及 README/实验矩阵中的推荐入口。

本 change 是一次破坏性删减。用户明确要求“不用管兼容问题”，因此实现不保留 deprecated alias、旧配置 virtual fallback、兼容 facade 或自动迁移脚本。删除范围只覆盖源码、配置、测试、文档和 OpenSpec 承诺；不得主动删除、移动或解压用户本地 `dataset/`、`outputs/`、`logs/`、checkpoint 或 cache 文件。

仓库中还存在一个窄范围 active change `remove-craf-marf-architectures`，当前只包含 `.openspec.yaml`。本 change 范围更完整，应作为后续实施的主 change；实施时需要处理该空壳 change，避免 OpenSpec 状态长期漂移。

## Goals / Non-Goals

**Goals:**

- 删除 MARF 架构源码、注册名、训练 helper、subset/prior/adapter 逻辑、配置和测试。
- 删除 CRAF 架构源码、teacher-prior CRAF、gate/counterfactual/reliability loss、teacher registry/loader、配置和测试。
- 删除 G2D distiller、SMP、teacher ensemble、diagnostics、训练接入、配置 alias/overlay 和测试。
- 删除 Multimodal-NF dataset family、preprocessing/audit/index/cache/runtime/profile、near-field objective、配置和测试。
- 清理 README、docs、OpenSpec active specs 和快速健康检查中的旧研究线入口。
- 保持当前保留的核心 CLI、标准单模态、标准/模块化 fusion、MMW、DeepSense、Raymobtime、CSI 等能力可运行。

**Non-Goals:**

- 不提供旧配置迁移、旧 checkpoint 读取兼容、同名 virtual alias 或 deprecated registry alias。
- 不新增替代算法、替代数据集或新的实验矩阵。
- 不删除用户本地真实数据、历史输出、日志、cache 或 checkpoint。
- 不把当前保留能力做大规模重构；只在删除旧引用导致必要时做窄修复。

## Decisions

1. 采用直接删除，不做兼容退役期。

   理由：这次目标是精简所有旧代码并明确“不用管兼容问题”。保留 alias 或 deprecated wrapper 会继续污染注册表、配置解析、测试和文档，无法真正降低维护成本。

   备选方案：保留代码并隐藏在 `deprecated` 开关后。该方案适合外部用户迁移期，但与本次删减目标冲突。

2. 先清 OpenSpec/文档契约，再删源码。

   理由：当前 active specs 仍要求旧能力存在。如果先删源码而不改 specs，后续 archive 或验收会继续把旧入口视为缺失功能。实施必须按 specs 删除正向承诺，再同步 README/docs/配置/测试。

   备选方案：只删代码和测试。该方案会留下设计债，后续 agent 会被旧 specs 引导重新补回这些能力。

3. 删除配置入口时不提供同名 virtual fallback。

   理由：仓库已有 virtual/overlay 配置机制，若继续接管旧路径，会让用户误以为 CRAF/MARF/G2D 仍受支持。退役路径必须失败，错误应指向“不受支持/已退役”，而不是静默替换为其它模型。

4. 删除范围以 active 支持面为准，历史 archive 保留。

   `openspec/changes/archive/**` 中的历史记录不需要重写。实现应清理 active `openspec/specs/**`、README/docs、当前源码和测试，但不改写 archive 里的历史变更，除非 OpenSpec CLI 明确要求。

5. 本地数据和产物只从源码支持面删除，不做文件系统清理。

   删除 `configs/multimodal_nf/`、preprocessor、dataset 和 tests 不等于删除 `dataset/MultimodalNF/` 或 `outputs/multimodal_nf/`。这些目录属于用户本地产物边界，实施任务不得主动操作。

6. 验证从旧方法正向测试切换为退役失败和当前能力 smoke。

   删除 `tests/test_g2d_*`、`tests/test_craf_*`、`tests/test_marf_*`、`tests/test_multimodal_nf_*` 后，需要补充或调整测试覆盖：旧注册名/配置失败、当前 CLI help 可用、架构边界不引用退役模块、保留配置仍可加载。

## Risks / Trade-offs

- 外部旧配置立即失败 → 这是预期 breaking change；错误信息应包含具体旧名称并说明入口已退役。
- 删除面很广导致误删当前保留 CSI/fusion/throughput 代码 → 实施时按关键词盘点并分类，只有专属旧研究线代码删除，共用 helper 先确认仍被保留能力依赖。
- OpenSpec active specs 与旧 change 残留不一致 → 实施时同步更新 active specs，并处理空壳 `remove-craf-marf-architectures` change。
- 删除测试降低覆盖 → 用当前保留工作流 smoke、CLI help、config load、registry failure 和架构边界测试补位。
- 历史输出无法再被新工具消费 → 接受该 trade-off；历史文件可保留为静态本地产物，但项目不承诺解析。

## Migration Plan

1. 盘点 `src`、`configs`、`tests`、`scripts`、`tools`、README/docs 和 active OpenSpec 中的 `marf`、`craf`、`g2d`、`multimodal_nf`、`Multimodal-NF`、`MultimodalNF` 引用。
2. 按模块删除源码和注册入口：模型/训练/distillation/diagnostics/data/preprocessing/runtime/profile。
3. 删除配置和 overlay/virtual alias：CRAF、MARF、G2D、Multimodal-NF 相关实体 YAML、recipe、测试期 retired entity expectations。
4. 更新 README/docs 和 active specs，删除推荐命令、实验矩阵、健康检查和数据布局承诺。
5. 删除旧正向测试，新增或调整退役失败测试与当前保留能力 smoke。
6. 运行 focused 验证和 OpenSpec 校验；最后运行 `conda run -n kd_mm_beam pytest -q` 或记录无法运行的原因。

Rollback 只能通过版本控制恢复本 change 删除的源码、配置、测试和 specs；不需要恢复或迁移本地数据。
