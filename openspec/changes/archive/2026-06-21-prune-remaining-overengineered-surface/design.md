## Context

当前仓库已经完成多轮入口收敛：主要运行路径在 `src/kd_sensing` 包、`kd-sensing-*` console scripts、OpenSpec current specs、README/docs 和 `docs/project_surface_inventory.md` 中。上一轮瘦身后，默认依赖、旧 thin scripts、部分 facade、单实现 adapter registry 和维护索引已经明显收缩。

剩余复杂度主要不是核心训练能力，而是维护表面：

- `configs/` 实体 YAML 增长到 141 个，其中 JEPA image+GPS、CSI hardening、BEV-Fusion、pretraining smoke 和 diagnostics manifest 存在大量可由 recipe、overlay 或 manifest 表达的重复。
- `tests/test_architecture_boundaries.py` 超过 2000 行，包含多份 allowlist、文档短语扫描和 OpenSpec lifecycle 镜像，已经接近“测试维护一个治理数据库”。
- `src/kd_sensing/config/migration_guards.py` 继续承担许多完全退役路线的运行时专用错误，而这些路线已经有 tombstone specs、inventory 和 README retired wording。
- 多个 `__init__.py` 或 facade 文件只 re-export owner 模块符号，内部测试和文档仍在引用这些转发层。
- `registry_self_check`、`_typing.AnyConfig`、独立 `SampleRow` 文件、第二份 `deep_merge`、一次性 CSI sweep analyzer 等属于低价值边界。

约束不变：所有项目 Python 命令使用 `conda run -n kd_mm_beam ...`；不得删除或修改 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`All_models/`；不得恢复旧 KD、HiST/Hist、Top8、residual、BGAM、viewer manifest、Raymobtime、CRAF/MARF/G2D 或 Multimodal-NF 当前入口。

## Goals / Non-Goals

**Goals:**

- 将剩余 over-engineered 表面按风险分类，并用 OpenSpec contract 明确哪些 public surface 不再承诺。
- 删除或合并无当前调用、无 registry、无 current docs/OpenSpec 消费且仅由测试或 re-export 维持的源码。
- 将重复实体 YAML 收敛为 recipe/overlay/manifest 或少量人工样例，保留实体配置优先语义。
- 将架构健康检查从 prose/allowlist 镜像改成结构事实验证。
- 保留当前 CLI、配置加载、训练/评估/预处理、诊断输出和本地产物边界。

**Non-Goals:**

- 不改变训练数学语义、模型结构、batch contract、beam label、metric 口径、checkpoint schema 或默认输出分区。
- 不删除真实数据、训练输出、日志、cache、checkpoint 或历史权重。
- 不新增长期治理系统、配置数据库、registry 层或兼容 wrapper。
- 不把本 change 扩展成新的模型、数据集、实验 workflow 或论文结果 claim。
- 不机械拆分当前已登记并接受的大 owner，例如 JEPA visual analysis、JEPA benchmark runner、DeepSense6G dataset 或核心模型文件。

## Decisions

1. **先收缩 contract，再删除实现。**  
   现有 specs 和 inventory 仍保护部分 facade、guard 和治理检查。实现前必须先用 delta specs 明确：低价值 public import 和长期 removed guard 不再是必须承诺。替代方案是直接删除源码再修测试；风险是 OpenSpec archive 时重新把旧承诺带回 current spec，不采用。

2. **按 wave 实施，最小可回滚。**  
   实现顺序为：契约/文档基线、配置矩阵分类、facade/import 收缩、registry/guard/helper 收缩、dataset row 类型收口、健康护栏重写、最终验证。每个 wave 只跑对应 focused tests。替代方案是按 `rg` 扫描一次性大删；回归定位太差，不采用。

3. **删除优先级按“是否当前公开入口”排序。**  
   最高优先删除无调用、无文档、无 registry 的 helper；其次删除内部 re-export facade；最后处理可能有外部脚本依赖的 public owner。对 package CLI 和 README 明确入口，只收缩实现 import，不删用户命令。替代方案是按行数排序；这会误伤当前大 owner，不采用。

4. **配置先分类，再删实体 YAML。**  
   每个候选 YAML 必须归入：canonical/root 保留、recipe 可无损生成、recipe 可生成但有显式差异、人工样例、debug/smoke、diagnostics manifest、归档/删除。只有可生成且有 focused config load 等价检查的实体 YAML 才删除。替代方案是直接删未被 `rg` 引用的 YAML；会漏掉用户手动命令和文档外运行，不采用。

5. **健康护栏验证事实，不验证措辞。**  
   架构边界测试保留 pyproject scripts、真实路径、OpenSpec lifecycle、配置引用、轻量 import、retired token 和本地产物 tracked boundary。删除固定中文短语、长 allowlist 和文档 prose 镜像。替代方案是把 allowlist 搬回 YAML；只是换地方养复杂度，不采用。

6. **retired guard 只保留高频迁移价值。**  
   保留 KD config token、image motion profile/encoder、scene dataset alias 等用户仍可能从当前 config 迁移遇到的错误。完全退役且已经有 tombstone/inventory 的旧路线，不再需要每个 runtime 分支都有专属错误。替代方案是保留所有 guard；维护成本会继续增长，不采用。

7. **BeamBench 保留行为，删除聚合表面。**  
   `kd-sensing-train-beambench-image-ae-gps` 和 Table III runner 继续可用；实现和测试直接 import `image_ae_gps_training.py`、`image_ae_gps_paper_split.py`、`image_ae_gps_config.py` 等 owner。`image_ae_gps.py` 大 re-export owner 可删除或缩到最小 public shim。替代方案是继续让聚合 owner 承担 public API；它会迫使所有内部改动同步维护转发表，不采用。

8. **`SampleRow` 不再独立成 runtime framework。**  
   target-shot split 可直接消费 `Mapping[str, Any]`；如果仍需要 row 类型，迁到 `target_shot_splits.py` 作为局部 dataclass。独立 `dataset_runtime.py` 不再作为长期 runtime capability owner。替代方案是为一个 dataclass 保留单文件；不值得。

9. **不为删除后的简单行为新增替代抽象。**  
   删除 `registry_self_check` 不新增 smoke runner；删除 `_typing.AnyConfig` 后直接使用 `dict[str, Any]`；删除第二份 `deep_merge` 后复用已有实现。替代方案是新增 `types.py`、`merge.py` 或 `checks.py`；只是换名保留复杂度，不采用。

## Risks / Trade-offs

- **[Risk] 外部未登记脚本依赖旧 import path。** → **Mitigation:** proposal/spec 标记 breaking change；README/docs 指向 owner 模块或 package CLI；当前 console scripts 不删除。
- **[Risk] 删除 guard 后错误信息变弱。** → **Mitigation:** 保留高频迁移 guard；退役路线通过 OpenSpec tombstone、inventory 和 README 继续可查；普通 unknown-name 必须列可用名称。
- **[Risk] YAML 删除影响用户手动实验。** → **Mitigation:** 删除前写候选分类和等价检查；保留人工样例和明确 diagnostics manifest；`final_config.yaml` / `resolved_config.yaml` 继续保存完整解析结果。
- **[Risk] 架构测试变短后漏掉回流。** → **Mitigation:** 保留结构性检查：entry point、lifecycle、retired token、tracked artifact、light import、root config、current config reference 和 facade 回流。
- **[Risk] 多个 active change 同时修改 docs/spec。** → **Mitigation:** 本 change 不触碰 `add-scene-conditioned-meta-offset-calibration` 的模型/数据实现；若冲突发生，以当前 change 的 proposal/design/spec/tasks 和 `openspec status` 明确合并顺序。
- **[Risk] star import 替换引入漏导入。** → **Mitigation:** 使用 focused JEPA benchmark tests 和 CLI help smoke 覆盖；不改变 runner output schema。

## Migration Plan

1. **Wave 0：契约和基线。**  
   完成本 change 的 proposal/design/spec/tasks；记录 `git status --short`；运行 `openspec validate prune-remaining-overengineered-surface --strict`。

2. **Wave 1：配置矩阵分类。**  
   生成候选清单：保留 root/canonical、可 recipe、差异 recipe、人工样例、debug/smoke、diagnostics manifest、删除/归档。为可删除 YAML 写最小 config load 等价测试或 manifest 解析测试，再删实体 YAML 和 stale 引用。

3. **Wave 2：facade/import 收缩。**  
   内部源码和测试迁到 owner 模块；删除或最小化 `objective_metadata.py`、`data`/`datasets` lazy export、`models.fusion` removed alias、BeamBench `image_ae_gps.py` 聚合 owner。运行 CLI help、BeamBench focused tests 和轻量 import tests。

4. **Wave 3：registry/guard/helper 收缩。**  
   删除 `registry_self_check`、`_typing.AnyConfig`、第二份 `deep_merge`、低价值 removed guard 和无意义 `__all__` 镜像；保留构建和高频迁移错误。运行 component registry、config load 和 architecture boundary tests。

5. **Wave 4：dataset runtime row 收口。**  
   将 target-shot split 改为纯 Mapping 或局部 row 类型；删除独立 `dataset_runtime.py`；确认 dataset metadata 和 target-shot split artifact tests 通过。

6. **Wave 5：一次性脚本和 benchmark API 风格。**  
   删除或归档 CSI hardening sweep analyzer；将当前结论保留在 docs。显式替换 JEPA benchmark 内部 `import *`；运行 JEPA benchmark focused tests。

7. **Wave 6：健康护栏重写和文档同步。**  
   收缩 `tests/test_architecture_boundaries.py`；更新 inventory、agent navigation、README 和 maintainer context index 的最小事实；运行架构边界和 config/CLI smoke。

8. **最终验证。**  
   运行：
   - `openspec validate prune-remaining-overengineered-surface --strict`
   - `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
   - `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_cli_help.py -q`
   - 受影响 focused tests：component registry、target-shot split、BeamBench、JEPA benchmark
   - 若多个 wave 都触碰核心路径，再运行 `conda run -n kd_mm_beam pytest -q`

回滚策略：每个 wave 保持可独立回退。若某 wave 测试失败，优先恢复该 wave 的文件或恢复实体 YAML/compat shim；不要恢复已验证通过的前序 wave。若配置 recipe 等价性无法证明，保留实体 YAML 并在 inventory 标记为人工样例或后续候选。

## Open Questions

- `image_ae_gps.py` 是直接删除，还是保留只导出两三个 package CLI 仍依赖的公开 runner 的极薄 shim，需要实现时按 README/docs/tests 最终引用决定。
- 部分 diagnostics manifest YAML 是否更适合作为实体样例而非 recipe，需要候选清单给出逐项判断。
- `SampleRow` 是否仍被外部用户脚本导入不可知；项目内可以迁到 Mapping，但最终说明需明确 breaking change。
- `scripts/analyze_csi_hardening_sweep.py` 的调试结论应落在 `docs/research_notes.md` 还是 CSI hardening 相关报告，需要实现时按现有文档承载位置选择最短路径。
