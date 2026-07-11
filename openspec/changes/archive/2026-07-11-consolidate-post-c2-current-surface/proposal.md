## Why

仓库在 post-C2 清理后仍有 84,294 行运行源码、111 个 current-spec capability 目录（71 `current`、13 `supporting`、27 `retired-tombstone`）和多组只靠测试或历史文档维持的诊断/治理表面；同时 current OpenSpec、inventory 与 README 对主线和退役路线存在冲突。继续拆文件或新增治理工具只会移动复杂度，本 change 直接收窄需求与 public surface，把 final C2 / U-MaskBeamJEPA、MMW/CSI 和必要证据链之外的低价值代码退出当前维护面。

## What Changes

- 先收口现有 active changes：在独立 closeout 中把 completed 的 `align-paper-baselines-window2` 收缩到真实 AMBER/AMR 内容或明确 abandon，不恢复 WCL CLI/config/spec 与旧 local-baseline public surface；H5/P1 正在消费的 RMBP channel-attention core 作为 temporal supporting component 保留。先修改 `add-temporal-window-missing` 的 proposal/design/spec/tasks 去除 S1-S4 与已被 focused tests/H5/P1 覆盖的旧 temporal check/launch/summary，再删除相应提前实现表面，之后才继续本 change。
- 将 final C2 / U-MaskBeamJEPA 定为唯一默认研究主线，MMW/CSI 定为保留 supporting workflow；DeepSense Image+GPS JEPA claim/mainline、Vision-Position、BeamBench、旧 Scene31 workflow、geometry prior、throughput 等只保留集中历史/拒绝语义，MMW current config 消费的 JEPA pretraining/mean context path继续保留。
- **BREAKING** 经逐项 consumer/guard 审计后折叠 24 个没有独立 guard 价值的 `retired-tombstone` specs，由 `retired-route-summary` 和参数化防回流检查统一承接旧名称、拒绝点与历史迁移说明；误分类的 `target-shot-domain-splitting`、`model-architecture-summary` 与 `local-missing-modality-baselines` 改为 supporting，不随墓碑删除。
- **BREAKING** 删除已证实零 current consumer 或已退役的运行源码与测试，包括 Scene31 summary、GPS-query 专属 downstream/evidence、geometry prior/reranker、LOSO helper、training I/O profile/throughput helper和窄 orphan facade；保留仍被当前 MMW JEPA mean-pooling、训练、评估或证据链消费的 owner。
- **BREAKING** 从 public CLI 和脚本生命周期中移除 project surface doctor、research dashboard/preview、历史 overnight launcher、被 H5/P1 覆盖的 temporal v1 脚本和 S1-S4 thin wrappers；不提供同名 alias、stub、virtual config 或兼容 facade。
- 保留 `kd-sensing-runs`、runtime cleanup/organize、paper export、U-Mask eval matrix、MMW GPS v2、MMW physics、核心 train/evaluate/preprocess 入口；runtime cleanup 的 manifest、tracked-file protection 和显式确认语义不变。
- 把 surface doctor 中唯一有价值的安全边界收敛为小型静态 guard，并缩减架构测试中的 deleted-path 镜像、固定文案断言和重复 lifecycle 表。
- 收缩 `maintainer_context_index.yaml` 与治理 specs，使其只记录无法从 pyproject、源码或 inventory 推导的任务路由，不再镜像 CLI、脚本、热点和完整源码树。
- 保留 Scene31-34 final analysis、missing-modality statistics/stress、final C2 claim docs、canonical configs、`src/kd_sensing/data/mmw/`、`src/kd_sensing/data/datasets/mmw_geometry.py`、`src/kd_sensing/models/physics/`、`src/kd_sensing/models/rmbp_mm.py`、CSI owner/tests/configs、U-Mask protected branch/loss 和用户 dirty H5/P1 launcher；本 change 不为减少行数引入配置继承、通用 launcher framework 或 `legacy/` 代码目录。

## Capabilities

### New Capabilities

无。本 change 只删除、折叠或收紧现有能力。

### Modified Capabilities

- `project-surface-cleanup`: 将 post-C2 清理从候选登记收敛为可执行删除边界，并更新 protected/current surface。
- `project-architecture`: 明确零消费者与 retired runtime 的退出规则、最小核心依赖流和禁止新兼容层的要求。
- `project-entrypoint-lifecycle`: 删除低价值诊断 CLI、历史 launcher 与 thin wrappers，保留最小 current entrypoint 集合。
- `project-health-guardrails`: 以小型结构/安全检查替代 surface doctor 产品面，并校验 active change、lifecycle 和 current wording 一致性。
- `spec-lifecycle-boundaries`: 无独立 guard 价值的 retired specs 必须折叠，不再要求 lifecycle inventory 与每个墓碑目录一一对应。
- `retired-route-summary`: 集中覆盖本轮折叠的旧能力名称、拒绝边界与允许保留的通用 helper 语义。
- `beam-distribution-shift-diagnostics`: 独立墓碑退出，由集中 retired summary 承接。
- `beambench-baseline-reproduction`: 独立墓碑退出，由集中 retired summary 承接。
- `bev-fusion-2604-reproduction`: 独立墓碑退出，由集中 retired summary 承接。
- `cxd-phase-transition-analysis`: 整项 retired analysis contract 退出。
- `dataset-reproducibility-audit`: 独立 retired audit capability 退出，不恢复其 CLI/product surface。
- `geometry-prior-beam-fusion`: 整项 geometry prior/fusion/claim/config contract 退出。
- `gps-query-effectiveness-visualization`: 整项 query evidence/visualization/claim gate contract 退出。
- `gps-query-jepa-pooling`: 整项 GPS-query pooler/config/metadata contract 退出。
- `jepa-gps-shortcut-benchmark`: 独立墓碑退出，由集中 retired summary 承接。
- `jepa-visual-analysis-suite`: 独立墓碑退出，由集中 retired summary 承接。
- `jepa-visual-architecture-sweep`: 独立墓碑退出，由集中 retired summary 承接。
- `modality-visual-diagnostics`: 独立墓碑退出，由集中 retired summary 承接。
- `predictive-jepa-robustness`: 整项 predictive model/benchmark/claim/artifact contract 退出。
- `rbma-prototype-kd-missing-workflow`: 旧独立 workflow spec 退出；U-Mask 内嵌 RBMA/prototype/teacher 分支保留。
- `real-perturbation-forward-evaluation`: 整项 retired benchmark contract 退出。
- `safe-residual-beam-rerank-fusion`: 整项 reranker model/loss/config contract 退出。
- `scenario-d-image-observability-benchmark`: 整项 D0-D7/CxD benchmark/report contract 退出，通用 image operator 保留。
- `scene31-baseline-pack`: 整项旧 baseline-pack contract 退出。
- `scene31-next-round-experiment-workflow`: 整项旧 next-round workflow contract 退出。
- `scenes31-34-subset-reliability-validation`: 旧 subset-reliability follow-up 退出，Scene31-34 final owner 保留。
- `tii-vlrg-transformer-reproduction`: 独立 retired reproduction spec 退出。
- `training-throughput-optimization`: 独立 throughput tombstone 退出，current run metadata 保留。
- `vision-position-baseline-suite`: 独立 retired baseline spec 退出。
- `wcl2025-robust-missing-modality-reproduction`: 独立 retired reproduction spec 退出。
- `mainline-experiment-documentation`: 删除 Image+GPS JEPA、Vision-Position、BeamBench 等作为 current mainline 的冲突要求，统一到 final C2 + MMW/CSI 口径。
- `maintainer-context-index`: 将机器索引限定为最小任务路由，删除可从其它权威推导的镜像字段要求。
- `research-claim-harvester`: 退役自动 dashboard/HTML claim candidate 产品面，保留人工 claim registry、paper export 和 run index 输入。
- `research-run-preview-loop`: 退役无训练 preview/budget manifest CLI 与其对其它治理入口的再包装。
- `html-evidence-dashboard`: 退役只服务 research dashboard 的静态 HTML renderer、输出契约和专属测试。
- `experiment-run-index`: 保留 artifact/run 扫描、进程/resource snapshot 与 cleanup 活跃运行保护契约，仅移除 research-harvester 专属下游命名。
- `gps-conditioned-jepa-pretraining`: 保留 current JEPA pretraining/mean-pooling 路径，删除对 GPS-query 专属 downstream 的要求。
- `jepa-downstream-extensibility`: 整项退出 current specs；仍有配置消费者的 mean context reuse 归回 `gps-conditioned-jepa-pretraining` owner。
- `component-registry`: 删除 geometry/GPS-query 等 retired component 的 current registry 期待，保留普通 unknown-name 拒绝。
- `cross-scene-loso-workflow`: 在零 current consumer 的既有条件成立后整项退役并删除 `kd_sensing.data.loso`；历史 split provenance 留在 archive/docs，当前 MMW 跨场景契约由 MMW spec 继续负责。
- `training-evaluation-runtime`: 删除对 surface doctor、research dashboard/preview 和 retired apples-to-apples module-only workflow 的 current 依赖。
- `agentic-collaboration-guardrails`: 将安全协作检查从 surface doctor/preview 产品面改为小型静态 guard 与显式命令。
- `u-mask-beam-jepa`: 明确 S1-S4 temporal-router 未进入 current contract，删除提前实现时保持既有受保护 fusion/loss 分支，包括内嵌 RBMA、beam prototype alignment 与 full-to-partial teacher stabilization。
- `overnight-branch-router-v2`: 删除已冻结训练矩阵的历史 launcher；保留 final C2 summary 仍直接消费的 read-only summary parser，并将其降为 supporting lifecycle。
- `model-architecture-extension-contract`: 删除 geometry-prior/BEV-lite 的 speculative component 与 metadata 扩展要求，保留通用组件优先和 current whole-model exception 规则。
- `modular-sequence-model`: 从 staged forward、metadata 与 architecture summary 契约中移除 geometry/safe-rerank/GPS-query 分支，保留 current modular、missing-mask、AMR/AMBER 与普通 fusion 语义。
- `observability-aware-fusion`: 删除只服务 geometry prior、predictive GPS-query 和 no-regret reranker 的要求，保留通用 reliability 与 U-Mask/Scene31 mask-weighted fusion。
- `soft-beam-label-training`: 将 DBA-aware supervised loss 契约与已退役 geometry-prior/teacher-rerank ablation 解耦。
- `automated-cache-policy`: 删除只服务 standalone training-I/O profile 和旧 JEPA perturbation benchmark 的要求，保留训练/评估/预热 cache policy。
- `model-architecture-summary`: 改为 training startup、instance parameter/trainability 与 Scene31-34 profile 的 supporting owner，删除 standalone CLI、candidate sweep、renderer 与 config preflight 表面。
- `target-shot-domain-splitting`: 纠正 lifecycle 为 MMW cross-scene supporting owner；保留 split/leakage/artifact 契约，但不恢复旧 standalone CLI。
- `reused-weight-fusion-diagnostic-metrics`: 整项退出 current specs；该能力没有实现或 current consumer，历史 CxD/GPS-query/geometry 诊断语义由 archive 保留。
- `local-missing-modality-baselines`: 改为只保护 Scene31-34 仍消费的 AMR-lite supporting contract；删除已由独立 AMBER/AMR specs 或 Scene31-34 owner覆盖的 baseline-pack、FeatureMod 和 maskfix 重复要求。
- `modality-difficulty-pipeline`: 保留通用 GPS/image operators、missing-stress、determinism 与 provenance；删除只服务已退役 Scenario-D、P0-P5 predictive、CxD/GPS-query advantage 和 shortcut benchmark cache/compatibility 的分支。
- `modality-contracts`: 保留 canonical modality 和通用 image/GPS reliability metadata 字段，移除对 Scenario-D 命名与 benchmark 专属传递的依赖。
- `experiment-workflow`: 将配置驱动实验收敛到 current train/evaluate/preprocess、JEPA pretraining/mean reuse、U-Mask、MMW/CSI，并删除旧 Scene31 BTAPA/night-grid/next-round workflow 要求。
- `openspec-document-health`: 将 GPS-query 从“current 合法语境”改为 retired wording guard，同时允许 mean JEPA、condition-id 禁用与历史说明。
- `u-mask-beam-jepa-eval-matrix`: 保留 aggregation schema，移除 claim-harvester 专属下游命名。
- `agent-context-portability`: 将 current brief/只读角色中的 dashboard/doctor 产品名改为工具中立 evidence/surface audit 语境。
- `canonical-config-resolution`: 删除依赖 surface doctor 的 config-list/doctor 产品要求；配置分类由 inventory、loader characterization 和静态引用检查承接。

## Impact

- 预期净减：约 9,000-10,500 行高置信运行源码、约 3,500-4,500 行 post-archive current OpenSpec/治理文本，以及对应测试、CLI 和脚本表面；target-shot、run-index resource protection 与 instance/startup architecture summary 不计入删除目标。
- 主要影响：`openspec/specs/`、`docs/project_surface_inventory.md`、`docs/maintainer_context_index.yaml`、README/agent context、`src/kd_sensing/diagnostics/`、JEPA downstream、geometry/LOSO/training I/O orphan、`scripts/`、`pyproject.toml` 和 focused tests。
- public breaking surface：被删除的 CLI、module-only CLI、script path 和内部 import path 不再可用；当前核心 CLI、配置路径、dataset split、beam label/metric、checkpoint、run metadata 和默认输出边界保持兼容。
- 不影响 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard、`All_models/` 或用户未提交的 `scripts/launch_h5_p1_temporal_models_v1.py` 改动。
- 不新增第三方依赖；所有项目 Python 验证继续使用 `conda run -n kd_mm_beam ...`。
