## Context

本仓库已经通过 OpenSpec、维护索引和架构边界测试把“当前支持面”和“历史/退役/本地产物”分开管理。但审计发现仍有几类维护成本没有被收敛：未接入的通用框架、只有自身测试引用的诊断模块、重复 helper、重复完整 YAML 矩阵、thin alias 脚本，以及测试对文档 prose 的逐字镜像。

这些问题的共同点是：它们不是当前论文 workflow 的核心能力，却扩大了修改半径。最小方案不是再加治理层，而是把现有契约改到允许删除、合并和迁移，然后按可验证的波次实施。

## Goals / Non-Goals

**Goals:**

- 删除没有当前调用方、没有公开入口、没有 current docs/OpenSpec 消费的源码模块和测试。
- 把公开入口收敛到 package console scripts，避免继续维护 `scripts/*.py` thin alias。
- 将 CSI hardening 重复实体配置压缩为 base config + overlay/recipe，同时保留当前矩阵语义。
- 将 JEPA benchmark facade 重新变薄：公开 API 保留，private helper 不经 facade 转发。
- 将架构边界测试从 prose mirror 改为结构化检查，继续覆盖入口、lifecycle、路径、AST/import 和本地产物边界。
- 删除未使用 dev 依赖，保持 runtime dependencies 不变。
- 更新 OpenSpec、README/AGENTS/docs、维护索引和 inventory，使删减后的支持面可审计。

**Non-Goals:**

- 不改变训练、评估、预处理、模型 forward、beam label、checkpoint schema 或 runtime output 分区。
- 不删除或移动 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/` 历史权重。
- 不重写 JEPA benchmark、CSI hardening、LiDAR BEV 或 dataset runtime 的核心功能。
- 不新增兼容 wrapper、legacy alias、旧配置虚拟映射或新的“cleanup framework”。
- 不把所有大文件强行拆小；只处理本次审计中有明确删除/合并证据的低价值 surface。

## Decisions

### 1. 入口迁移到 package console scripts

保留 `pyproject.toml` 的 `kd-sensing-*` console scripts，删除 `scripts/train.py`、`scripts/evaluate.py`、`scripts/preprocess.py` 和其它只调用包内 CLI 的 thin alias。README、AGENTS、docs、维护索引和 CLI help tests 改为推荐 console scripts。

替代方案是继续保留 thin alias 并只改文档。这会少破坏短期命令，但仍要维护两套入口和架构 allowlist；本 change 明确接受 breaking migration，换取长期支持面更小。

### 2. 删除孤立源码前先用“当前 surface”判定

删除候选必须同时满足：CodeGraph/静态扫描无当前内部调用；不在 package scripts、公开导出、registry、README、docs、current OpenSpec、配置或维护索引中作为 current surface；测试不是唯一业务消费者。符合条件的模块连同只服务它们的测试一起删除。

当前候选包括 `communication_state_features` 和未接入的 `lidar_pillar_encoder`。`dataset_runtime.py` 只删除未消费的通用 runtime/adapters 框架，保留或迁移仍被 `target_shot_splits` 使用的轻量 row 类型。

### 3. JEPA benchmark facade 只留公开 API

`jepa_gps_shortcut_benchmark.py` 保留 runner、manifest loading、CLI 消费的公开常量/函数和稳定导入路径。测试中对 `_private_helper` 的断言迁移到 helper 所在窄模块，facade 不再把 private helper 重新导出为事实 API。

这样比继续提高 facade 行数预算更便宜；也比删除整个 facade 风险小，因为 CLI 和文档仍需要稳定公开入口。

### 4. CSI hardening 用 overlay 表达矩阵

保留 A0/A1/A2/B/C/D/E 等逻辑配置 ID 和可加载行为，但不再为每个组合维护一份完整 YAML。实现时建立一个 base config，再用小 overlay YAML、recipe table 或已有配置解析能力生成等价 resolved config。

测试应验证关键配置 ID 可加载、resolved config 中的核心字段一致、destructive negative control 仍启用 `csi_degradation` 且 D 组不启用 destructive degradation。测试不逐行比较 YAML 文本。

### 5. 架构边界测试停止镜像 prose

`tests/test_architecture_boundaries.py` 保留机器可读治理验证、路径存在性、entrypoint 双向同步、OpenSpec lifecycle、AST/import 边界、retired token guard 和本地产物边界。删除对 README、docs、OpenSpec 大段固定短语的逐字断言。

文档仍要准确，但不靠测试把自然语言冻结。需要测试的事实放入 `docs/maintainer_context_index.yaml`、inventory lifecycle 表或 OpenSpec requirement。

### 6. 重复 helper 合并而不是抽象化

两份 `OutputRegistry` 只保留一处 owner，或直接内联为一个小函数。除非已有调用方明确需要类状态，不新增通用 registry 包。

### 7. 依赖删除只处理零使用项

`thop` 和 `pytorch-model-summary` 从 dev extras 删除。若未来需要 FLOPs 或 model summary，优先用已有 PyTorch/测试输出或在对应 change 中重新证明依赖必要性。

## Risks / Trade-offs

- 脚本入口删除会破坏旧命令 → 在 README、AGENTS、docs、错误信息和最终说明中给出 console script 替代；CLI help tests 覆盖新入口。
- 外部用户可能直接 import private helper → 本仓库不把 underscore helper 当公开 API；保留公开 facade runner/API，并在变更说明中标注 private import 迁移到窄模块。
- CSI overlay 可能改变 resolved config → 添加 focused config load/equality tests，比较关键字段和 hardening/degradation 语义，而不是比较源 YAML 文本。
- 删除 orphan 模块可能遗漏隐式文档引用 → 实施前再次扫描 pyproject、README/docs、current OpenSpec、维护索引和 registry；发现 current 引用则先更新契约或保留。
- 架构测试变短可能漏掉文档漂移 → 将必须稳定的事实转入机器可读索引和 OpenSpec scenarios；自然语言只做路径和 lifecycle 级 guard。
- 工作树已有无关修改 → 实施时不回滚用户改动；若同文件冲突，先读取并按当前内容合并。

## Migration Plan

1. 更新 OpenSpec/current docs 契约：入口改为 console scripts，dataset runtime 允许等价实现，LiDAR pillar 原型标记为非当前 surface，CSI matrix 允许 base+overlay，JEPA facade 明确 public/private 边界。
2. 更新维护索引和 inventory：删除 scripts allowlist 中的 thin alias，登记 merge/delete candidates，记录 CSI matrix 新配置分类，更新热点预算和验证命令。
3. 先处理测试目标：把 private helper 测试迁移到窄模块；把架构边界测试 prose mirror 改成结构化 checks；删除只服务 orphan 模块的测试。
4. 删除源码和依赖：删除 orphan modules、thin scripts、重复 helper、未使用 dev deps；`dataset_runtime.py` 只保留仍有消费者的轻量类型。
5. 收敛 CSI 配置：新增 base+overlay/recipe 机制，删除重复完整 YAML，保证现有矩阵 ID 仍能被测试和文档解析。
6. 验证：运行 `openspec validate simplify-overengineered-surfaces --strict`、架构边界测试、CLI help tests、配置加载测试、JEPA benchmark focused tests，以及必要时全量 `conda run -n kd_mm_beam pytest -q`。

Rollback 边界按波次执行：入口迁移可通过恢复 scripts 和索引条目回滚；CSI overlay 可恢复实体 YAML；源码删除可按模块恢复。不得通过新增兼容 wrapper 回滚。

## Open Questions

- 是否保留 `scripts/` 目录中的非 Python orchestration 脚本不在本 change 范围内；本 change 只处理 Python thin aliases。
- CSI hardening overlay 采用纯 YAML anchors、recipe table，还是复用现有 virtual config resolver，由实施时按现有配置加载器最小改动决定。
