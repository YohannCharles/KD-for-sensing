## Context

当前研究主线已经转向多模态缺失模态鲁棒性波束预测。实际核心路径集中在 U-MaskBeamJEPA、Scene31-34 local/manual 主线实验、缺失模态 mask/dropout 评估、AMBER/RMBP-MM/TII-VLRG 风格对照，以及仍被文档登记为 secondary/supporting 的 Image+GPS JEPA、MMW/CSI/physics-informed 路线。

仓库表面的问题不是“所有旧代码都该删”，而是“当前主线、历史证据、本地手工脚本、secondary 能力和 retired tombstone 混在一起”。`project_surface_doctor` 已能发现未分类脚本、未分类配置和未登记热点；继续保留未分类入口会降低表面治理的价值。最小可行策略是先分类，再删除/合并；对仍支撑论文证据链的入口只登记生命周期，不强行重构。

约束：

- 所有项目 Python 命令、测试和验证 MUST 使用 `conda run -n kd_mm_beam <command>`。
- 不删除、不移动、不重写 `dataset/`、`outputs/`、`logs/`、checkpoint、cache 或本地实验产物。
- 不新增旧入口兼容 wrapper、virtual alias、二级聚合层或 legacy run path。
- 不把 `scripts/` 研究脚本升级为 package CLI 或 current quickstart 唯一入口。
- OpenSpec 与 docs 是当前契约来源；实现发现范围变化时先更新 artifact。

## Goals / Non-Goals

**Goals:**

- 将缺失模态主线相关源码表面分为 current、secondary/supporting、local/manual、historical、retired 和 delete/merge candidate。
- 合并 Scene31-34 TinyViT/PatchViT encoder ablation 的重复生成器趋势，避免继续按 encoder family 复制脚本。
- 给 final polish、presentation artifact、historical table/conclusion helper 一个明确生命周期：仍有当前论文用途则登记，已经完成使命则删除。
- 缩小 Scene31 与 RBMA/KD/BTAPA 配置表面，只保留当前主线、复现实验或 claim 证据需要的实体 YAML。
- 登记或拆分未登记大 owner，特别是 `gps_query_evidence.py`、`run_metadata.py` 和 `u_mask_beam_jepa.py`。
- 更新 doctor、architecture boundary、focused tests 和 inventory，使清理结果可回归。

**Non-Goals:**

- 不改变 U-MaskBeamJEPA 的数学语义、默认训练语义、指标口径或 checkpoint schema。
- 不退役 Image+GPS JEPA、MMW/CSI/physics-informed 等已登记 secondary/supporting 路线；若要 de-scope，需要单独 change。
- 不删除 migration guards、retired route tests、用户输入边界检查、输出落盘边界检查、split/label-space 检查。
- 不整理本地运行产物，不迁移旧实验输出，不生成新训练结果。
- 不新增统一工作流平台、任务调度器或新的依赖。

## Decisions

### Decision 1: 采用“分类优先”的删除模型

每个候选先进入分类表，再执行删除、合并、保留或后续 change。候选至少记录 owner、当前调用方、公开入口风险、替代入口、验证命令和回滚方式。

理由：当前很多脚本和 YAML 是论文证据链材料，单纯按“未被包内调用”删除会误伤复现实验。分类成本低，能把真正该删的入口从“仍有证据价值”的入口中分离出来。

替代方案：

- 直接删除未被 CodeGraph 调用的文件：速度快，但会误删 shell、paper table、presentation 和本地复现实验材料。
- 只更新 inventory 不删代码：风险低，但不能减少表面噪声，也无法阻止重复 ablation 入口继续增加。

### Decision 2: Encoder ablation 使用单一 generator/runner 语义

TinyViT 与 PatchViT Scene31-34 ablation MUST 由一个参数化生成器和一个 family/manifest 驱动 runner 承担。实现时可以保留更合适的现有文件作为 owner 并改名，或新增一个统一 owner 后删除旧重复文件；但不得新增单独 PatchViT runner。

理由：两个约 400 行生成器只因 encoder family 差异而分裂，是当前最明确的“可并不该扩”的表面。继续复制 runner 会把本地 ablation 变成新维护路线。

替代方案：

- 保留两个生成器并只补 inventory：最少改动，但重复逻辑会继续漂移。
- 为 PatchViT 新增 runner：短期方便，长期扩大脚本面，和主线收缩目标冲突。

### Decision 3: Final/presentation/historical helper 默认不是 current workflow

`export_scene31_34_presentation_artifacts.py`、`run_final_scene31_34_polish.sh`、paper table、per-scene summary 和 conclusion helper 只有两种合法状态：

- 仍服务当前论文交付或复现审计：登记为 local/manual analysis helper，写清输入、输出、owner、删除触发条件。
- 不再服务当前交付：删除脚本、测试和 current 文档引用；保留已沉淀结论到 docs 或报告。

理由：这些脚本不是训练/评估核心能力。把它们混在 current surface 中，会让后续协作者误以为它们是主线入口。

替代方案：

- 全部直接删除：最干净，但可能丢失正在使用的论文交付路径。
- 全部保留：短期安全，但 doctor 会继续暴露未分类噪声。

### Decision 4: 配置缩小只删除“可再生成或无 current 证据”的实体 YAML

Scene31 generated YAML、RBMA/KD/BTAPA overlay、seed/tau/weakKD/PatternFiLM 等配置族 MUST 按以下优先级处理：

1. 当前主线或 claim 证据仍引用：保留并登记 lifecycle。
2. 能由 generator、template、manifest 无损重建：从源码实体 YAML 中移除或迁入生成产物边界。
3. 只服务已完成历史 sweep：删除实体 YAML，并把结论/caveat 留在文档或结果注册表。
4. 无法确认：保留为 local/manual 或后续 change，不在 current quickstart 推荐。

理由：配置文件对复现实验敏感，删除必须以引用和生成能力为证据，而不是只看目录规模。

替代方案：

- 只删 `configs/scene31` 下文件：会忽略 RBMA/KD/BTAPA 的真实噪声来源。
- 大规模迁移到 archive 目录：看似保留，实际仍扩大源码表面；可再生成内容应离开 tracked source。

### Decision 5: 大 owner 先登记/拆分，不作为第一批删除对象

`u_mask_beam_jepa.py`、`run_metadata.py` 和 `gps_query_evidence.py` 是当前大 owner 或诊断 owner。它们的第一动作 MUST 是登记、接受、拆分或标注 split-next；不能因为“大”就删除。若后续要缩小 U-Mask loss/model 内部 branch，必须先证明对应配置和 specs 不再使用这些 branch。

理由：这些文件可能是当前主线或 evidence metadata 的核心。热点治理目标是降低维护成本，不是削掉证据链。

替代方案：

- 立即拆分全部大文件：范围大，容易混入行为变化。
- 完全不管大文件：doctor 会继续报告未登记热点，后续难以控制增长。

### Decision 6: 验收以 doctor 和 focused tests 为准

实现完成后，至少验证：

- `openspec validate prune-missing-modality-mainline-surface --strict`
- `openspec validate --all --strict`
- `conda run -n kd_mm_beam python -m kd_sensing.cli.project_surface_doctor --scope scripts --scope configs --scope hotspots --format markdown --fail-on warning`
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`
- encoder ablation 相关 focused tests，按实现后的统一 test 名称运行
- `conda run -n kd_mm_beam python scripts/verify_compile.py`

理由：本 change 的风险主要是支持面漂移、旧入口回流和引用失效；doctor 与 focused tests 比全量训练更能快速定位这些问题。

替代方案：

- 只运行全量 pytest：成本高，且不一定精准覆盖表面治理。
- 只运行 OpenSpec validate：能验证 artifact，但不能发现真实文件漂移。

## Risks / Trade-offs

- 删除 local/manual 脚本可能打断某个未记录的私人运行命令 → 删除前必须检查 docs/OpenSpec/tests/pyproject/registry 引用；无法确认时先登记为 local/manual，不删除。
- 合并 encoder ablation 生成器可能改变输出 YAML 排序或默认字段 → focused tests 必须对 TinyViT 和 PatchViT 的最小 manifest 输出做结构断言。
- 配置删除可能让历史 claim 难以复跑 → claim 或 table 仍引用的配置必须保留，或先沉淀为可复现 manifest/generator 输入。
- 大 owner 登记后可能没有立即减少行数 → 本 change 接受这个 trade-off；先阻止未登记增长，再把行为拆分留给后续小 wave。
- doctor 从 `fail-on none` 提升到 `fail-on warning` 可能暴露既有红点 → tasks 必须区分本 change 修复范围和既有环境/数据缺失，不用放宽检查掩盖漂移。

## Migration Plan

1. Baseline：运行 surface doctor，记录未分类 scripts/configs/hotspots，并把候选放入 implementation checklist。
2. Wave 1：合并 Scene31-34 encoder ablation generator/runner；删除重复文件或旧 runner 引用；更新 tests 和 inventory。
3. Wave 2：处理 final polish、presentation、historical table/conclusion helper；删除或登记，不留未分类。
4. Wave 3：清理 Scene31 与 RBMA/KD/BTAPA 配置族；只删除可再生成、无 current 引用或已沉淀结论的实体 YAML。
5. Wave 4：登记/拆分/接受大 owner 热点；对 U-Mask branch 的任何删除必须由配置与 spec 证据驱动。
6. Docs/spec sync：更新 inventory、mainline catalog、scoped context、README/experiment matrix 中被影响的 current/local/manual 描述。
7. Validation：执行 OpenSpec、doctor、architecture、CLI/config、encoder ablation 和 compile 验证。

Rollback 策略：每个 wave 保持小提交边界。若删除误伤 current workflow，恢复该文件并改为 local/manual 登记；不得通过新增旧入口 wrapper 回滚。

## Open Questions

无阻塞问题。实现阶段若发现某个 helper 仍被当前论文交付使用，默认动作是登记 local/manual lifecycle，而不是继续追求删除数量。
