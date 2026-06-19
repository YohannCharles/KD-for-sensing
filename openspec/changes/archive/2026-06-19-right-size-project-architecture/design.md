## Context

CodeGraph 可用且索引健康：643 个索引文件、8,185 个节点、18,069 条边；其中 Python 文件 359 个、function 节点 3,420 个、import 节点 2,503 个。本地 AST 复核显示 359 个 Python 文件分布为 `src` 278、`tests` 58、`scripts` 23；`src/kd_sensing` 中复杂度主要集中在 `data`、`engine`、`diagnostics`、`models`、`cli` 和 `baselines`。

按当前架构契约，项目已经从旧根脚本和兼容聚合层迁移到 `src/kd_sensing` 包、包内 CLI、current/pending OpenSpec capability、公开 thin facade 和本地产物边界。现有结构整体方向合理：多文件拆分来自多条 current workflow、轻量导入边界、配置/数据/训练/诊断职责隔离和 retired route guard。真正需要处理的是两个相反问题：少数热点函数/owner 仍承担过多 orchestration；另一些 helper 文件只服务单 owner，继续拆分会增加 import 面和导航成本。

当前关键证据：

- `src/kd_sensing/data` 64 文件、936 个函数、443 条 import，是 dataset/sample/manifest/transform 复杂度中心。
- `src/kd_sensing/engine` 43 文件、598 个函数、374 条 import，是 trainer/evaluation/runtime 复杂度中心。
- `src/kd_sensing/diagnostics` 31 文件、612 个函数、451 条 import，包含 JEPA benchmark、visual analysis、viewer manifest、run index 和 cleanup。
- `src/kd_sensing/cli` 31 文件但仅约 2,069 行、101 个函数，主要是 thin parser/alias，不是首要合并目标。
- `src/kd_sensing/baselines/beambench` 的 Image AE+GPS 已拆成 public owner、config、datasets、models、AE cache、training、evaluation、paper split 和 reports；问题集中在 paper split/training orchestration，而不是 public owner。
- 长函数集中在 `_train_inner`、`run_mmw_town_gps_v2`、`run_image_ae_gps_paper_split_training`、`DeepSense6GDataset.__init__`、`run_jepa_gps_shortcut_benchmark`、`run_image_ae_gps_training`、`run_deepsense6g_gps_lidar_bgam`、`run_evaluation_pass` 等。

## Goals / Non-Goals

**Goals:**

- 给出完整、可分阶段实施的项目架构右尺寸化方案，解释哪些区域应该拆、合并、保留或仅监控。
- 用机器可读治理表约束热点预算、public surface policy、merge-candidate、right-size-accepted rationale、验证命令和 rollback note。
- 降低热点函数的维护风险，尤其是 dataset 初始化、训练 loop、BeamBench paper split、evaluation pass 和大型诊断 workflow。
- 降低低价值 helper 文件和 import 面，优先合并同 owner、单调用点、只服务 re-export 或无复用价值的边界。
- 保持公开 CLI/import、console scripts、配置路径、数据 split、beam label 语义、指标口径、manifest schema、run metadata 和本地产物边界兼容。

**Non-Goals:**

- 不以全局文件数、函数数或 import 数作为硬性减量 KPI；它们只作为风险定位和趋势监控指标。
- 不恢复旧根脚本、旧兼容聚合层、退役 KD/Hist/Top8/residual/G2D/CRAF/MARF/Multimodal-NF 路线。
- 不修改真实 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、历史本地产物或系统启动/认证文件。
- 不在本 change 内重写模型数值语义、训练协议、官方复现口径或 benchmark claim status。
- 不把所有大文件强制拆小；审计型 owner 若比分散 helper 更清晰，可以登记 `right-size-accepted`。

## Decisions

### 1. 使用“owner + budget + public surface policy”而不是全局文件数目标

全局 359 个 Python 文件并不能直接说明架构坏：`cli` 多文件是 console script/thin parser 的自然结果，`tests` 多文件也有助于 focused coverage。方案改用 owner 维度判定：公开 facade 必须薄；业务 owner 可以较大但要有理由；热点函数超过预算必须拆；低价值 helper 要合并或标为 keep-and-test。

替代方案是设定全局文件数上限，例如把 359 压到某个数字。该方案会诱导合并 CLI、测试和当前职责清晰的窄模块，风险高且无法证明可维护性提升。

### 2. 拆热点按稳定状态边界，不按“每 200 行一个文件”切割

拆分优先级：

- P0 治理护栏：补齐 `docs/maintainer_context_index.yaml` 中 architecture sizing baseline、wave、预算、merge-candidate 和 accepted owner，并让架构边界测试读取这些字段。
- P1 BeamBench Image AE+GPS：保留 `image_ae_gps.py` public owner；将 paper split 的 scene dataset build、AE/checkpoint reuse、feature cache setup、per-scene evaluation、summary payload/writer 收敛到已有 `datasets`、`ae`、`training`、`evaluation`、`reports` owner，避免 paper split 函数继续吸收实现细节。
- P1 DeepSense6G dataset：继续把 `DeepSense6GDataset.__init__` 中 resource reader setup、scaler/normalizer setup、target provider setup 分别迁入现有 `deepsense6g_loaders.py`、`deepsense6g_scalers.py`、`deepsense6g_targets.py` 或等价窄模块。
- P1 trainer：将 `_train_inner` 的 startup/build context、epoch loop、checkpoint coordination、final evaluation/artifact finalization 拆到 `training_state`、`training_metrics`、`checkpointing`、`trainer_runtime_helpers` 或少量新增 owner，保持 `train()` 公开入口不变。
- P2 evaluation pass：将 metric accumulation、objective auxiliary output、prediction metadata rows 和 optional modality quality report 从 `run_evaluation_pass` 中抽成 schema-safe helper。
- P2 diagnostics/visual analysis：将 `jepa_visual_analysis.py`、`run_index.py`、`runtime_artifact_cleanup.py` 纳入 hotspot inventory；优先拆 report/table/figure/cache 或 process/resource/schema writer 边界。
- P2/P3 models/config：将 `models/modular.py`、`config/canonical.py`、`data/difficulty/operators/image.py`、`data/transform_ops/csi.py` 标记为 monitor 或 keep-and-test；只有在对应功能变更时才拆，避免为了降行数破坏模型/配置审计性。

替代方案是先处理最大文件，例如直接拆 `jepa_visual_analysis.py` 或 `jepa_benchmark_runner.py`。这会绕过公开 workflow 风险和 active predictive change，容易造成大范围冲突。

### 3. 合并低价值 helper 只在同 owner 内发生

合并条件：

- 文件只被同一 owner 调用，且没有独立 public import/CLI/test 契约。
- 文件只为 re-export、常量搬运或降低单文件行数服务。
- 合并后不会引入重依赖 eager import，也不会创建跨领域 `helpers.py`。
- 合并后治理索引和架构边界测试同步更新，旧 helper 路径不新增兼容 facade。

候选类型包括单调用点 report payload、benchmark scalar/path helper、过细的 diagnostics writer helper、过窄的 script-local parser helper。`losses/jepa.py`、`losses/gps_lidar_bgam_losses.py`、`models/csi_encoder.py` 这类小而内聚的模块默认 keep-and-test，不强制合并。

替代方案是把所有 helper 合并回大 owner。该方案会降低文件数，但会让 owner 再次膨胀，并可能破坏轻量导入边界。

### 4. 保留公开 facade，但拒绝内部回流

公开 owner/facade 的原则：

- `kd_sensing.baselines.beambench.image_ae_gps`、`kd_sensing.diagnostics.jepa_gps_shortcut_benchmark`、`kd_sensing.diagnostics.viewer_manifest`、`kd_sensing.data.mmw.preparation`、`kd_sensing.models.csi` 等继续保留 public import 或 CLI 兼容语义。
- facade 文件只 re-export、做薄 orchestration 或解析公开入口；新增内部代码不得从 facade 回流导入 helper。
- 架构边界测试要检查 facade 行数、禁止片段、内部 import 路径和 public surface owner metadata。

替代方案是删除 facade 并让调用方全部改到窄模块。该方案会造成公开 import breaking change，不符合本 change 的非破坏性目标。

### 5. import 数治理关注 eager import 和跨领域依赖，不机械减少 import 语句

2,503 个 CodeGraph import 节点需要分三类处理：

- 必要的窄模块 import：保留。
- thin CLI parser import：保留，但 CLI 文件不得导入训练循环以外的重型实现，除非 run 阶段需要。
- 风险 import：公开 facade、`__init__.py`、config/path/registry 轻量模块不得 eager import dataset、model、trainer、matplotlib、pandas/scipy/skimage 或 checkpoint/weights。

验证重点是轻量导入 smoke、facade 防回流和重依赖缺失环境下的 import success，而不是把 import 语句数压低。

## Risks / Trade-offs

- 热点拆分改变内部调用顺序 → 使用 focused characterization tests 固定数据 split、checkpoint schema、summary CSV/JSON、manifest fields 和 metric rows。
- 合并 helper 误删隐性 public import → 先用 CodeGraph callers/callees 和 `rg` 字面路径扫描确认内部-only，再更新测试拒绝旧 helper 回流；不新增兼容 facade。
- Wave 与活跃 change 冲突 → P0 治理和 P1 BeamBench 可先做；JEPA predictive/benchmark 相关合并等 active predictive 语义稳定后再实施。
- 大 owner 被错误拆散导致审计更难 → `right-size-accepted` 必须包含 accepted rationale、consolidation_targets 和 focused tests；没有新增行为时不强行拆。
- 统计基线漂移 → CodeGraph 和 AST 统计只作为趋势与风险定位，不作为单独失败条件；真正失败条件来自预算、导入边界和公开行为兼容。
- 训练/评估验证耗时 → 每个 wave 绑定最小 focused tests；最终回归仍以 `conda run -n kd_mm_beam pytest -q` 为验收，无法运行时在最终说明中列出原因。

## Migration Plan

1. Wave 0：更新 OpenSpec delta、维护索引 schema、inventory、agent navigation 和架构边界测试，让治理表能表达 baseline、hotspot、merge-candidate、right-size-accepted 和 rollback note。
2. Wave 1：处理 BeamBench Image AE+GPS P1 hotspot，优先收口 `image_ae_gps_paper_split.py` 的 report/checkpoint/scene orchestration，并运行 `tests/test_beambench_image_ae_gps_direct.py` 和架构边界测试。
3. Wave 2：处理 DeepSense6G/MMW dataset 与 trainer P1 hotspot，按 resource/scaler/target/epoch/checkpoint/finalization 边界拆分，运行 dataset modality、training IO、epoch subsampling 和架构边界测试。
4. Wave 3：处理 evaluation pass、BGAM、MMW GPS v2 和 diagnostics second tier；只在 schema-safe helper 可明确时拆。
5. Wave 4：处理 diagnostics consolidation 与 JEPA benchmark owner；保留 active predictive change 期间的 accepted owner，待语义稳定后再考虑拆 predictive 子域。
6. Wave 5：合并或 keep-and-test 低价值 helper，更新 inventory、索引和测试，运行全量回归。

每个 wave 都必须可单独回滚：只回滚该 wave 新增/移动的 helper、索引项和测试，不能改变公开 CLI/import 或本地产物目录。

## Open Questions

- 是否把 `jepa_visual_analysis.py` 作为 P1 还是 P2：它是最大源码文件，但如果当前没有功能变更，先纳入治理表和测试预算即可。
- `models/modular.py` 是否需要拆出 forward preparation/metadata helper：目前它是核心模型 owner，建议等下一次模型语义改动时顺手拆。
- 是否为 CodeGraph/AST 统计新增自动化命令或只保留人工审计：建议先在 inventory 记录统计口径，避免测试对自然增长过度敏感。
