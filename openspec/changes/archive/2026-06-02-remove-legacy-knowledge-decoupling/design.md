## Context

当前 HiST-Beam 体系里，旧知识解耦路线同时存在于模型、loss、LOSO planner、summary comparison、配置矩阵、脚本、README 和 OpenSpec 契约中。核心形态是 `v2_shared_private`/`v3_decoupled`：把表征拆成 shared 与 private 分支，用 orthogonality、shared scene confusion 和 private scene preservation 等 loss 期望获得可迁移知识。

用户的多轮实验已经证明该方法不可行，目标场景迁移精度长期在约 10% 徘徊。与此同时，仓库已有新的验证方向：image-only legal probe、target-shot geometry residual foundations、history/residual calibration、V7 shared physical private residual、V8/V9 target prior/prototype probe。此次变更的目标不是继续修补旧路线，而是让代码和实验矩阵停止把它当作主 baseline 或默认 source。

所有项目相关 Python 测试、验证和运行命令必须使用 `conda run -n kd_mm_beam <command>`。`outputs/`、`logs/` 属于本地运行产物，清理时不进入源码变更；但本变更需要提供可审计清单，避免误删当前活跃实验。

## Goals / Non-Goals

**Goals:**

- 从模型注册、配置解析和默认值中退役 `v2_shared_private`、`shared_private`、`v3_decoupled`、`decoupled`。
- 删除旧简单解耦专属的 scene classifier、orthogonality/scene confusion/private preservation loss 路径和相关 diagnostics。
- 让 adapter/prototype、radio/path、image-only、V7 residual、history residual 等保留路线不再依赖 `v3_decoupled` 作为默认 source checkpoint 或 comparison baseline。
- 更新 LOSO 默认矩阵、summary、quick conclusion、README 示例和脚本，停止推荐旧失败路线。
- 为 `outputs/`、`logs/` 旧失败实验产物生成 machine-readable 删除清单，并按清单清理匹配项。
- 保留归档 OpenSpec 历史，确保“为什么退役”可追溯。

**Non-Goals:**

- 不删除 image-only legal probe、target-shot geometry residual foundations、V7 residual、V8/V9 target prior/prototype 或未来 residual/calibration 代码。
- 不把本地数据集、内置复现权重或 `All_models/` 历史权重纳入删除范围。
- 不尝试通过兼容 alias 继续支持旧 variant 名称；遇到旧配置应给出清晰迁移错误或拒绝。
- 不在本变更中提出新算法或新增长实验矩阵。

## Decisions

1. 采用退役而不是隐藏开关。

   - 决策：旧 variant 从 `HIST_BEAM_VARIANTS`、LOSO `SUPPORTED_VARIANTS` 和默认配置中删除；旧名称不作为 alias 保留。
   - 理由：方法已经被判定不可行，保留隐藏开关会继续制造 baseline 选择和 source checkpoint 决策的歧义。
   - 替代方案：保留 `legacy_enabled` 开关。该方案便于复跑历史，但会延长维护面，并与“不新增旧入口/兼容聚合层”的仓库规则冲突。

2. 保留“有 shared/private 字段的现代路线”，删除“简单解耦失败路线”。

   - 决策：V7 residual、history residual、path/radio prototype 等如果仍输出 shared/private 表征，必须把语义绑定到 residual、prototype、geometry 或 calibration 契约；不得复用旧 orthogonality + scene confusion + private scene preservation 作为默认生效机制。
   - 理由：新路线中的 shared/private 可能只是命名相近，不能因为旧失败路线退役而误删真正需要的 residual/calibration 结构。
   - 替代方案：全仓删除所有 shared/private 字段。该方案过于粗暴，会破坏仍活跃的 V7、path/radio、history residual 契约。

3. 先切断依赖，再删除实现。

   - 决策：实施顺序为：更新 source variant 选择和默认 baseline；改配置/脚本/README；调整模型/loss；更新测试；最后清理本地产物。
   - 理由：`v3_decoupled` 当前被 adaptation run 当作 source baseline。如果先删模型，保留路线会在 source checkpoint 选择阶段失效。
   - 替代方案：先删除模型再修调用点。该方案反馈更快，但错误会扩散到多个 runner 和 summary 测试，定位成本更高。

4. 本地产物清理采用“清单优先”。

   - 决策：新增或使用一次性清理脚本/命令生成 `legacy_knowledge_decoupling_cleanup_manifest.json`，记录 path、kind、matched_patterns、size 和保留/删除决策；确认清单只命中旧失败路线后再删除。
   - 理由：用户要求删除输出日志和实验结果，但 `outputs/`、`logs/` 中也有活跃 image-only、target-shot 和 residual 产物。清单能防误删并留下审计依据。
   - 替代方案：直接 `rm -rf outputs/*v3* logs/*v3*`。该方案快，但会误伤以 `v8`/target prior 命名却仍需保留的活跃 probe，也无法解释删除范围。

5. 测试从“旧 baseline 存在”改为“旧 baseline 不可用，保留路线可运行”。

   - 决策：更新测试断言，使 `v3_decoupled` 不再出现在默认 quick matrix、summary comparison baseline 和 supported variants；新增拒绝旧 variant 的测试。保留路线至少覆盖模型构建、loss smoke、LOSO plan 和架构边界。
   - 理由：退役变更的风险不是少了旧指标，而是新路线无意依赖旧 source baseline。
   - 替代方案：只删代码不改测试。该方案会留下错误契约，后续开发者会按测试重新引入旧路线。

## Risks / Trade-offs

- [Risk] 删除旧 loss 时误删 V7/residual/path/radio 仍需要的 shared/private 表征。→ Mitigation：按 variant 分支审计输出字段，保留现代路线的 shared/private diagnostics，但不再绑定旧 scene confusion/private preservation loss。
- [Risk] adapter/prototype 仍隐式调用 `_source_variant_for(...)=v3_decoupled`。→ Mitigation：先更新 source variant 选择策略，并用测试覆盖 image-only、V4/V5/V6/V8/V9/V7 的 source mapping。
- [Risk] 本地产物清理误删当前活跃结果。→ Mitigation：清理 manifest 默认 exclude `image_only_legal*`、`target-shot`、`geometry_residual`、`residual`、`v7`、`v8_target_prior_head`、`v9` 等活跃标识，删除前记录匹配原因。
- [Risk] 历史 README 或归档 OpenSpec 仍出现旧路线名称。→ Mitigation：只更新现行 README/spec/config；归档变更允许保留历史上下文，但不得作为当前需求来源。
- [Risk] 用户仍有旧配置文件需要复查失败结果。→ Mitigation：错误信息指向归档说明和新 baseline 名称；本地运行产物删除清单保留路径摘要和匹配原因。

## Migration Plan

1. 更新现行 OpenSpec delta，标记旧简单 shared/private 解耦路线退役，并调整 LOSO baseline/prototype 依赖契约。
2. 修改配置解析与 runner 默认值，使任何未显式指定的 HiST-Beam 运行不再落到 `v3_decoupled`。
3. 删除旧 variant 配置、脚本默认项、README 示例和 summary comparison 中的 `v3_decoupled` 主 baseline。
4. 删除模型/loss 中仅服务旧路线的 scene classifier、orthogonality/scene loss 计算和 diagnostics，保留现代路线需要的字段。
5. 更新测试并运行相关 `conda run -n kd_mm_beam pytest ...` 与 OpenSpec 校验。
6. 生成 `outputs/`、`logs/` 清理清单，核对不包含活跃实验后执行删除。

Rollback 策略：如果保留路线无法在短期内脱离旧 source checkpoint，可先恢复 source mapping 的新 baseline 选择逻辑，而不是恢复旧解耦模型；旧实验结果如已删除，只能依赖清理 manifest 和已有 git/外部备份追溯。

## Open Questions

- `v4_adapter`、`v5_adapter_proto` 退役旧 source 后默认应该使用 `v1_hierarchical`、`v0_flat`，还是直接要求显式 source variant；实施时需根据现有 checkpoint 复用逻辑选择最小破坏方案。
- 旧失败产物的匹配规则是否应包含所有以 `p3_v8` 命名的结果，还是只删除 metadata/config 中明确记录 `source_variant=v3_decoupled` 或旧 loss 权重非零的结果；实施时优先采用 metadata 证据。
