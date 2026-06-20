## Context

仓库当前已经把主要运行路径收敛到 `src/kd_sensing` 包、当前 CLI、配置 schema、OpenSpec specs 和 README/docs 组合上。但多轮迁移之后，仍有一批历史复杂度被规格和测试继续保护：`docs/maintainer_context_index.yaml` 及其测试 helper、包级 re-export facade、legacy wrapper、removed-name guard table、只有 identity 实现的 adapter registry、退役 tombstone current specs，以及若干不再必要的强依赖和样板 import。

这些表面大多不直接服务训练、评估、预处理或当前诊断工作流，却会让后续修改需要同步维护多份 allowlist、迁移说明和兼容层。本设计的核心是先把 OpenSpec 契约改到允许“删除过度工程面”，再分层实现，避免清理行为违反现有规格。

项目约束保持不变：项目相关 Python 命令必须使用 `conda run -n kd_mm_beam <command>`；不得把数据、输出、日志、cache、checkpoint 或临时运行产物纳入源码变更；不得通过兼容聚合层绕过当前 `src/kd_sensing` 包结构。

## Goals / Non-Goals

**Goals:**

- 删除或合并审计确认的低价值源码、文档、测试和依赖表面。
- 让健康护栏验证关键结构事实，而不是维护大型机器可读镜像。
- 收缩公开兼容导入面，内部代码和文档回到真实 owner 模块、配置名或 registry 名称。
- 把已退役路线的长期防回流机制从多处 guard table 收敛到更小的文档、inventory 和 focused tests。
- 保持当前核心训练、评估、预处理、CLI 和输出边界行为稳定。

**Non-Goals:**

- 不改变训练数学语义、数据 split、beam label 口径、checkpoint schema、默认输出目录或评估指标。
- 不删除 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/` 历史权重。
- 不新增替代框架、长期抽象层或新的通用治理系统。
- 不把本次清理扩展成新的模型、数据集、实验路线或论文结果 claim。

## Decisions

1. **先改契约，再删实现。**  
   现有 specs 明确要求维护索引、models 包级延迟导出、removed-name guard 和 JEPA adapter registry。实现前必须先通过 delta spec 修改这些要求，否则任何代码删除都会变成规格违约。替代方案是直接删代码后修测试；这会让 OpenSpec 和实现短期互相矛盾，不采用。

2. **按风险 wave 实施，而不是一次性扫全仓。**  
   实现分为可独立验证的 wave：依赖与样板瘦身、单实现扩展点收缩、兼容 facade 收缩、健康护栏和维护索引瘦身、退役 tombstone specs 归档。每个 wave 记录目标、回滚条件和 focused tests。替代方案是按文件类型批量删除；风险是更难定位回归来源，不采用。

3. **当前 owner 路径优先于兼容 facade。**  
   新的公共面以 package CLI、配置 schema、canonical registry 名称和真实 owner 模块为准。`kd_sensing.models`、BeamBench legacy wrapper、diagnostics facade 等只在明确当前入口需要时保留薄层；历史兼容 import 不再是默认义务。代价是外部未登记脚本可能需要改 import，但这是本次明确标记的 breaking change。

4. **removed guard 只服务仍有当前迁移价值的名称。**  
   对仍在 docs、配置迁移或常见错误路径中的名称，保留清晰错误是有价值的。对已经完全退役、没有当前迁移目标的旧名称，普通 unknown-name 错误加文档 tombstone 足够。这样可以删掉长期 guard table，同时保留 registry 的基本可诊断性。

5. **单实现 registry 延后到第二个真实实现出现。**  
   `JEPA_DOWNSTREAM_ADAPTERS` 当前只有 identity/no-op，配置选择面并不存在。默认 no-op 可以直接内联，pooler registry 仍保留，因为 mean 和 GPS-query 等 pooler 是实际选择面。若未来新增非 identity adapter，再通过独立 OpenSpec change 恢复窄 registry。

6. **依赖删除以行为等价为边界。**  
   `scikit-image` 图像读取替换为 Pillow 时，必须保持当前 profile、dtype/shape、缓存行为和测试期望。`h5py` 仅在 MMW HDF5 path semantics 或真实 HDF5 读取需要时成为 optional extra 或局部导入。annotation future import 删除只在确认 Python 版本契约不低于 3.10 后批量执行。

7. **健康护栏保留“会造成维护事故”的检查。**  
   保留轻量导入、CLI/pyproject、退役入口回流、未分类 current 入口、本地产物边界和关键 path/schema 检查。删除只复制文档 prose、长期 mirror 大表或为了维护索引而维护索引的检查。

## Risks / Trade-offs

- **[Risk] 外部未登记脚本依赖历史 facade。** → **Mitigation:** 在 proposal、spec 和实现说明中标记 breaking change；文档给出 owner 模块或 canonical registry/config 迁移路径；当前 CLI 保持可用。
- **[Risk] 删除 removed guard 后错误信息变弱。** → **Mitigation:** 只删除无当前迁移价值的 guard；仍常见或仍有当前替代路径的名称保留明确诊断。
- **[Risk] 维护索引瘦身导致架构漂移漏检。** → **Mitigation:** 用更小的 focused tests 覆盖 pyproject scripts、current docs 路径、退役 token、轻量导入和热点回流，不依赖单一大型 YAML。
- **[Risk] 依赖替换改变图像解码细节。** → **Mitigation:** 对图像读取、缓存和 preprocessing profile 跑 focused tests，并在必要时保留 dtype/shape 转换的显式兼容逻辑。
- **[Risk] 批量删除 `from __future__ import annotations` 造成机械 churn。** → **Mitigation:** 独立 wave 执行，先确认 Python 版本契约，再使用格式化和 focused import tests 验证。
- **[Risk] tombstone specs 归档后旧路线语义不清。** → **Mitigation:** 保留集中 retired route summary 和 wording guard；archive 只作为历史，不作为 current 支持面。

## Migration Plan

1. 更新 OpenSpec delta，使当前契约允许瘦身行为。
2. 依赖与样板 wave：替换 `skimage.io.imread`、调整 `h5py` 依赖边界、删除 annotation future import。
3. 单实现和脚本 wave：折叠 identity adapter registry、简化一次性分析脚本、删除只服务已删表面的测试。
4. Facade wave：把内部引用迁到 owner 模块，删除不再作为 current public surface 的 re-export 和 legacy wrapper。
5. Guardrail wave：收缩 `docs/maintainer_context_index.yaml` 或删除它，重写架构测试为 focused checks，更新 `docs/agent_navigation.md` 和 `docs/project_surface_inventory.md`。
6. Lifecycle wave：将不再需要 current runtime guard 的 retired tombstone specs 归档或折叠到集中历史清单。
7. 验证 wave：运行 OpenSpec strict validate、架构边界 focused tests、相关单元测试；若修改面已跨越多个核心模块，再运行 `conda run -n kd_mm_beam pytest -q`。

回滚策略：每个 wave 单独提交或至少单独记录；若某 wave 失败，回退该 wave 的源码/测试/文档修改，保留已通过的前序 wave。依赖删除 wave 失败时优先恢复依赖声明，而不是绕过测试。

## Open Questions

- `h5py` 是否仍被当前默认安装路径直接需要，还是可以完全迁入 optional extra，需要在实现时用 import 和测试引用确认。
- `kd_sensing.models` 是否保留少量当前公共符号，还是彻底只保留轻量 package metadata，需要实现时根据 README/docs/tests 的真实引用决定。
- 退役 tombstone specs 的归档粒度是“一次归档多项”还是“仅归档本次确认无 current guard 价值的项”，实现时以 validation 最小风险为准。
