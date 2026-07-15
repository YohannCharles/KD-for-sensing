# project-entrypoint-lifecycle Specification

## Purpose
定义项目 CLI、脚本、研究入口、本地手工入口和退役入口的生命周期边界，避免临时 wrapper、旧 workflow 或本地队列脚本重新变成当前推荐入口。
## Requirements
### Requirement: 一次性研究脚本不得长期占据 current surface
只服务已完成调试结论、历史 sweep 汇总或人工复盘的一次性脚本 MUST 从当前支持面删除、归档为历史文档，或明确标为 local/manual research artifact。保留脚本时 MUST 记录 owner、输入输出边界和仍需运行的 focused 验证；删除脚本时 MUST 保留必要结论和 caveat 到 docs 或报告。

#### Scenario: 删除 CSI hardening sweep analyzer
- **WHEN** `scripts/analyze_csi_hardening_sweep.py` 或等价一次性脚本只服务历史 CSI hardening 调试结论
- **THEN** 本 change MAY 删除该脚本和只服务它的测试
- **AND** 仍有价值的结论 MUST 留在 `docs/research_notes.md`、CSI hardening 文档或对应报告中

#### Scenario: 保留当前诊断脚本
- **WHEN** 某个 `scripts/` 文件仍被 README、docs、OpenSpec current spec 或 package workflow 明确引用
- **THEN** 本 change MUST 不删除该脚本
- **AND** 若脚本保留，inventory MUST 将其分类为 research diagnostic、dataset preparation、figure helper 或 manual/local script

### Requirement: 旧实验和诊断表面必须可删除或降级
项目 MUST 对已审计为低价值的旧实验、旧诊断、孤岛 helper 和 local/manual runbook 建立删除或降级边界。候选项只有在不属于当前 package CLI、registry、canonical config、README/docs 当前入口、OpenSpec current requirement 或必要 focused test 输入时，才可删除；否则 MUST 降级为薄兼容 reader、local/manual surface 或记录删除触发条件。

#### Scenario: 删除旧 full sweep runner
- **WHEN** `cnn_hybrid_jepa_visual_prior_sweep` 的训练 runner、job graph、shell 生成和 cleanup 逻辑不再被 current docs、CLI、tests 或 OpenSpec 当前 requirement 需要
- **THEN** 本 change MUST 删除这些旧执行逻辑或将其降级为只读兼容 reader
- **AND** 当前推荐入口 MUST 指向 `jepa_visual_architecture_sweep` owner 和 manifest

#### Scenario: 删除孤岛 helper 前有证据
- **WHEN** CodeGraph 或结构检查显示某个 public helper 无当前内部调用
- **THEN** 删除任务 MUST 同时检查 pyproject、README/docs、OpenSpec current specs、tests、registry 和 package `__all__`
- **AND** 无公开契约消费时才可删除；否则 MUST 记录保留理由或降级计划

#### Scenario: 本地 runbook 不升级为当前入口
- **WHEN** 固定 GPU shell、一次性 runner 或本地 overlay YAML 只服务 Scene31/RBMA/M2Beam 本地实验
- **THEN** inventory MUST 将其标记为 local/manual、删除候选或已归档历史
- **AND** README quickstart 和 package CLI MUST 不把该脚本描述为长期推荐入口

### Requirement: 重复 CLI 脚本不得作为推荐入口
当包内 CLI 与 `tools/` 脚本提供同一工作流时，项目 MUST 以包内 CLI 或 `python -m kd_sensing.cli.<name>` 作为推荐入口。已被包内 CLI 覆盖的重复 fallback wrapper MUST 删除；仍保留的研究、数据准备或 shell 脚本 MUST 有明确生命周期分类。

#### Scenario: viewer manifest wrapper 不回流
- **WHEN** 文档或 orchestration 脚本提到旧 viewer manifest 导出
- **THEN** 对应段落 MUST 标记为退役或历史
- **AND** 项目 MUST 不再保留 `tools/visualization/export_viewer_manifest.py` fallback wrapper

### Requirement: Viewer manifest 聚合模块已退役
Viewer manifest 相关模块与 wrapper MUST 继续退役。该 guard MUST 不再把 JEPA visual analysis 或 GPS shortcut benchmark 作为 current migration owner；current diagnostics/evaluation 只指向 U-Mask matrix、MMW/CSI、Scene31-34 final analysis 和其它明确 retained owner。

#### Scenario: Viewer helper 不存在
- **WHEN** diagnostics/CLI surface 被检查
- **THEN** viewer manifest helpers/prediction exporter MUST 不存在
- **AND** 不得迁移到已退役 JEPA diagnostics

### Requirement: 兼容冗余入口已删除
项目 MUST 删除已经迁移到 canonical 模块的兼容入口。源码、测试、文档和推荐命令 MUST 不再依赖 `the builder facade module`、`the transform facade module`、`the transform aggregate module`、场景专用 dataset 兼容模块或复制旧实现的可视化脚本入口。明确保留的 console-script 入口 MUST 指向当前包内主实现。

#### Scenario: 兼容 facade 不再作为公开入口
- **WHEN** 开发者在源码、测试、README 或扩展指南中搜索已删除的兼容 facade
- **THEN** 不得出现 `the builder facade module`、`the transform facade module` 或 `the transform aggregate module` 的运行时引用
- **AND** 对应功能 MUST 通过职责明确的窄模块导入

#### Scenario: 旧入口引用检查
- **WHEN** 开发者运行架构边界检查
- **THEN** 检查 MUST 拒绝新增 `scene-specific dataset class alias`、`the scene-9 dataset-type spelling`、legacy fusion 配置路径或兼容 facade 引用
- **AND** 检查 MUST 在不读取真实数据和不加载 checkpoint 的情况下完成

#### Scenario: 可视化兼容入口已退役
- **WHEN** 开发者检查 console scripts 和包内 CLI
- **THEN** 项目 MUST 不保留 `kd-sensing-visualize-modalities` console script
- **AND** 项目 MUST 不保留 `kd_sensing.cli.export_viewer_manifest` 或等价 viewer manifest 主实现

### Requirement: 重复开发入口必须有生命周期
当包内 CLI 或 console script 已覆盖同一工作流时，项目 MUST 删除对应 `scripts/` 或 `tools/` fallback wrapper，或者在 OpenSpec 中明确其短期保留原因和删除条件。保留的研究脚本 MUST 不作为 README 推荐入口。

#### Scenario: viewer manifest fallback wrapper 删除
- **WHEN** 文档或历史记录提到 `kd-sensing-export-viewer-manifest` 或 `python -m kd_sensing.cli.export_viewer_manifest`
- **THEN** 对应上下文 MUST 标记为退役或历史
- **AND** 项目 MUST 不再要求保留 `tools/visualization/export_viewer_manifest.py` 作为 fallback wrapper

#### Scenario: 研究脚本保留边界清晰
- **WHEN** `scripts/` 或 `tools/analysis/` 中的脚本没有等价包内 CLI
- **THEN** 该脚本仅可作为研究/诊断工具保留
- **AND** 文档 MUST 不把该脚本描述为训练、评估、预处理或 manifest 导出的唯一推荐入口

### Requirement: MMW 入口生命周期 inventory 必须同步
新增或保留的 MMW Python 脚本、数据准备脚本和研究支持入口 MUST 具有可审计生命周期。固定 shell orchestration 已退役；项目表面积 inventory 与架构边界测试 MUST 同步记录入口类别、保留原因、推荐入口关系、输出产物边界和删除或收敛条件。

#### Scenario: 新增 MMW 脚本入口需要 inventory
- **WHEN** 开发者新增 `scripts/`、`scripts/mmw/` 或 `tools/analysis/` 下的 MMW Python 或 shell 入口
- **THEN** 架构边界检查 MUST 要求该入口出现在项目表面积 inventory 或等价生命周期文档中
- **AND** inventory MUST 说明该入口属于 package CLI、研究诊断脚本、数据准备脚本、config generator 或 local/manual helper 中的哪一类
- **AND** 对应测试 allowlist MUST 与 inventory 保持一致

#### Scenario: 未登记入口导致表面积检查失败
- **WHEN** 工作区中存在未登记的 MMW Python 或 shell 入口
- **THEN** 表面积回归检查 MUST 失败
- **AND** 失败信息 MUST 列出缺失登记的相对路径
- **AND** 失败信息 MUST 指向更新 inventory、删除重复入口或改为包内 CLI 的修复路径

#### Scenario: 重复 MMW orchestration 不成为推荐入口
- **WHEN** 多个本地脚本或 shell wrapper 覆盖同一 MMW quick validation 工作流
- **THEN** inventory MUST 标记推荐入口和补充 profile 的关系
- **AND** README 或 docs MUST 不把重复 wrapper 描述为唯一 canonical 入口
- **AND** 若已有包内 CLI 覆盖同一工作流，重复 wrapper MUST 删除或降级为一次性本地命令

### Requirement: 退役旧模态诊断脚本入口
旧 modality subset/perturbation scripts MUST 不作为长期入口。通用 subset/mask/difficulty 行为 MUST 由 shared evaluation、U-Mask matrix、missing-stress、MMW/CSI 或内部 helper 承载；JEPA visual/shortcut MUST 不在 allowlist。

#### Scenario: Script allowlist 使用 current owners
- **WHEN** architecture test 枚举 scripts/tools
- **THEN** 旧 modality scripts 和 JEPA visual/shortcut wrappers MUST 不存在
- **AND** retained dataset preparation、MMW/CSI、U-Mask 和 Scene31-34 entries MAY 保留

#### Scenario: 通用 subset 能力保留
- **WHEN** current evaluation 配置启用 modality subset/mask
- **THEN** shared evaluation MUST 继续工作
- **AND** 不依赖 retired scripts

### Requirement: GPS+LiDAR BGAM 包内入口已退役
GPS+LiDAR BGAM reranker 的包内入口、manifest enrich、dataset、model、loss、engine、evaluation、debug plot 和 CLI 已退役。项目 MUST 删除这些专属模块和 console scripts，并 MUST NOT 新增长期维护的顶层 `train_gps_lidar_bgam.py`、`eval_gps_lidar_bgam.py`、`datasets/gps_lidar_dataset.py`、`models/gps_lidar_bgam.py` 或包内兼容入口。

#### Scenario: BGAM console scripts 不暴露
- **WHEN** 开发者完成 editable install 并查看 `pyproject.toml` entry points
- **THEN** 项目 MUST 不暴露 GPS+LiDAR BGAM 相关 console scripts
- **AND** scripts MUST 不包含 manifest enrich、训练/评估运行或独立评估入口

#### Scenario: BGAM module CLI 不存在
- **WHEN** 用户或测试查找 `kd_sensing.cli.run_deepsense6g_gps_lidar_bgam`
- **THEN** 该 module path MUST 不存在
- **AND** 项目 MUST 不提供等价兼容 alias

#### Scenario: 不新增顶层旧入口
- **WHEN** 架构边界测试扫描退役 workflow
- **THEN** 测试 MUST 验证仓库根目录不存在新增的 `train_gps_lidar_bgam.py` 或 `eval_gps_lidar_bgam.py`
- **AND** 内部代码 MUST 不依赖顶层 `datasets.*`、`models.*` 或 `src.run_*` BGAM 入口

#### Scenario: 轻量导入边界保持稳定
- **WHEN** 开发者执行 `import kd_sensing` 或导入配置/路径轻量模块
- **THEN** 系统 MUST 不因退役 BGAM 名称 eager import torch dataset、LiDAR point cloud reader、matplotlib plotter 或训练 runtime
- **AND** BGAM 重依赖模块 MUST 不作为当前 import surface 存在

### Requirement: Hist 研究线不属于当前包结构
项目当前包结构 MUST 不再要求或暴露 HiST-Beam/Hist 专用 CLI、engine、model、evaluation 或 config 模块。`src/kd_sensing/engine` 与 `src/kd_sensing/models` MUST 保留当前主线职责模块，退役 Hist 专用文件后不得新增旧入口 facade。

#### Scenario: 包导入不要求 Hist 模块
- **WHEN** 开发者执行 `import kd_sensing`、`import kd_sensing.engine` 或 `import kd_sensing.models`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不要求 `kd_sensing.engine.hist_beam_*` 或 `kd_sensing.models.fusion.hist_beam` 存在

#### Scenario: 架构边界拒绝 Hist 旧入口
- **WHEN** 开发者运行架构边界检查
- **THEN** 检查 MUST 验证当前源码不再从 Hist 专用 engine/model/evaluation 模块导入运行逻辑
- **AND** 检查 MUST 验证没有新增 `hist_beam` 兼容聚合层

### Requirement: 退役失败实验路线不得保留源码支持面
当用户明确退役某条失败研究路线并要求不保留兼容时，项目 MUST 从当前源码支持面删除该路线的公开入口、配置、实现模块、测试和文档推荐路径。系统 MUST 不新增兼容 alias、stub CLI、薄 facade 或 registry fallback 来维持旧路线可发现性。

#### Scenario: 退役入口不可安装
- **WHEN** 开发者刷新 editable install 后检查 console scripts
- **THEN** 已退役路线的 `kd-sensing-*` 命令 MUST 不再由 `pyproject.toml` 声明
- **AND** 项目 MUST 不提供等价旧命令 alias 或兼容包装层

#### Scenario: 退役实现不可作为当前模块导入
- **WHEN** 开发者检查 `src/kd_sensing/cli`、`data`、`engine`、`models` 和 `losses`
- **THEN** 已退役路线专属模块 MUST 不再作为当前源码模块保留
- **AND** 保留主线不得从这些退役模块导入 helper

### Requirement: Top8 residual coarse BGAM viewer 退役边界
Top8 selector 训练/plot/compare、GPS coarse anchor、GPS prior residual/delta correction、camera residual、BGAM、BGAM-only TopK candidate manifest/loss 支撑、viewer manifest、Gradio viewer 和 Raymobtime s008 MUST 不属于当前包结构和推荐入口。通用 Top-K 指标、circular metrics、GPS-Rel-Polar、GPS v2、CSI 和 JEPA MAY 保留；Raymobtime、BGAM 和 viewer 旧名称只允许作为 migration guard 或退役说明出现。

#### Scenario: 保留通用指标
- **WHEN** 清理实现扫描到 `topk`、`candidate` 或 `residual` 字符串
- **THEN** 系统 MUST 按语义判断归属
- **AND** 普通 evaluation Top-K、CSI candidate ranking 和 GPS v2 自身 residual 诊断不得仅因字符串命中被删除

### Requirement: 优先退役入口不得作为 current public surface
项目 MUST 将本 change 标记的优先退役入口从 current public surface 移除。被移除的入口 MUST 不再出现在 `pyproject.toml` console scripts、README quickstart、CLI help smoke、当前 structured inventory 或 `scripts/` current allowlist 中。历史说明 MAY 保留，但 MUST 标记为 retired、historical、blocked background 或 tombstone；项目不再 MUST 为这些入口维护 `docs/maintainer_context_index.yaml` 条目。

#### Scenario: 退役 package CLI 不再声明
- **WHEN** 开发者检查 `pyproject.toml` 和安装后的 console script help smoke
- **THEN** 项目 MUST 不声明 `kd-sensing-run-amr-net-gps-image`
- **AND** 项目 MUST 不声明 `kd-sensing-run-jepa-msac`
- **AND** CLI help smoke MUST 不要求这两个命令存在

#### Scenario: 退役 script 不在 current allowlist
- **WHEN** 开发者检查脚本入口健康检查、当前 structured inventory 或保留的脚本 allowlist
- **THEN** current 入口 MUST 不包含 `scripts/mmw/visualize_gps_angle_beam_correspondence.py`
- **AND** current 入口 MUST 不包含 `scripts/mmw/visualize_gps_prediction_trajectory.py`
- **AND** current 入口 MUST 不包含 `scripts/mmw/visualize_prediction_error_label_distribution.py`
- **AND** current 入口 MUST 不包含 `scripts/run_deepsense_gps_circular_soft_label.sh`、`scripts/run_mmw_gps_circular_soft_label_ablation.sh`、`scripts/run_mmw_sunny_modal15_l5p3_h123.sh` 或 `scripts/run_mmw_sunny_modal15_l5p6_h246.sh`

### Requirement: 退役入口回流必须被架构边界测试拒绝
架构边界测试 MUST 拒绝 retired CLI/module/script/config/wrapper 回流，并 MUST 将迁移方向指向 final C2/U-Mask、retained package CLI、MMW/CSI、Scene31-34 或普通 unknown-name behavior；不得指向 JEPA visual/shortcut 等本轮删除 owner。

#### Scenario: 旧模块与脚本不回流
- **WHEN** structure guard 运行
- **THEN** retired module/script/config tokens MUST 不作为 current surface存在
- **AND** historical docs MAY 保留明确 retired wording

### Requirement: 当前推荐 workflow 排除 Top8 residual coarse 路线
README、quickstart、experiment matrix 和 inventory MUST 将 current workflow 聚焦 final C2/U-Mask、retained train/evaluate/preprocess、MMW/CSI、AMR/AMBER controls、Scene31-34 evidence 和必要 supporting owners。Top8/residual/BGAM/viewer、Image+GPS query、Vision-Position、JEPA visual/shortcut 和 geometry MUST 不作为 current 推荐面。

#### Scenario: Quickstart 使用 retained workflow
- **WHEN** current docs 被检查
- **THEN** 推荐命令 MUST 来自十个 package CLI 或 protected local/manual owner
- **AND** retired command/config MUST 只在 historical context出现

#### Scenario: Guard 不要求 retired imports
- **WHEN** quick health check 运行
- **THEN** 它 MUST 不导入 retired route
- **AND** MAY 断言 retired route不存在

### Requirement: 优先退役 workflow 不得作为当前实验入口
当前实验 workflow MUST 不再推荐、声明或验证 AMR-Net_gps_image mock/source-audit runner、JEPA-MSAC mock/paper-aligned runner、MMW GPS v2 旁支 `scripts/mmw/visualize_gps_*` 脚本，或固定 GPU/local shell orchestration 脚本。历史背景 MAY 保留，但 MUST 不提供 current 运行命令。

#### Scenario: 实验矩阵不推荐退役 workflow
- **WHEN** 开发者阅读 README 或 `docs/experiment_matrix.md`
- **THEN** 文档 MUST 不推荐运行 `kd-sensing-run-amr-net-gps-image`
- **AND** 文档 MUST 不推荐运行 `kd-sensing-run-jepa-msac`
- **AND** 文档 MUST 不推荐运行被退役的 MMW 旁支诊断脚本或固定 shell orchestration 脚本

#### Scenario: 配置加载拒绝退役配置
- **WHEN** 用户加载 `configs/baselines/amr_net_gps_image.yaml`、`configs/pretraining/jepa_msac_s32_smoke.yaml` 或 `configs/pretraining/jepa_msac_s32_paper.yaml`
- **THEN** 配置加载 MUST 失败或对应实体配置 MUST 不存在
- **AND** 错误信息或文档 MUST 说明该 workflow 已退役并指向当前 baseline、diagnostic 或 reproduction 入口

### Requirement: 当前替代入口必须清晰
退役上述 workflow 后，项目 MUST 在文档中给出当前替代入口。替代入口 MUST 是仍受支持的 package CLI、current config 或明确保留的 shell runner，不得新增旧式兼容 wrapper。

#### Scenario: MMW 诊断迁移到 package CLI
- **WHEN** 文档说明 MMW GPS v2 图表或对比
- **THEN** 文档 MUST 指向 `kd-sensing-mmw-town-gps-v2 --mode plot` 和 `kd-sensing-mmw-town-gps-v2 --mode compare`
- **AND** 文档 MUST 不要求用户直接运行退役的 `scripts/mmw/visualize_gps_*` 脚本

#### Scenario: shell runner 迁移到当前入口
- **WHEN** 文档说明 DeepSense GPS soft-label、MMW soft-label ablation 或 MMW sunny modal15 历史实验
- **THEN** 文档 MUST 将其标记为 historical 或 retired
- **AND** 当前运行建议 MUST 使用 `kd-sensing-train`、当前 package diagnostics、保留的 CSI hardening matrix runner 或明确 current 的配置

### Requirement: Local/manual 实验面必须可收敛
本地 Scene31、RBMA missing-modality、strong encoder checkpoint 复用和 M2Beam 单模态训练 overlay MUST 被分类为 local/manual experiment surface。它们 MAY 保留为人工运行材料，但 MUST 有 owner、输出边界、删除触发条件和不升级为 package CLI 的说明。

#### Scenario: 固定 GPU queue shell 删除
- **WHEN** `scripts/run_next_v3_experiments.sh`、`scripts/run_rbma_strong_encoder_4gpu_queue.sh`、`scripts/run_m2beam_single_modal_scene31_queue.sh` 或其它固定 GPU queue shell 出现在源码树
- **THEN** inventory 和架构边界检查 MUST 将其视为退役回流
- **AND** 文档 MUST 指向 `kd-sensing-train --config <yaml>`、manifest generator 或 Python diagnostic helper

#### Scenario: 统一 runner 覆盖后删除 shell
- **WHEN** 一个 local/manual runner 或文档命令已经覆盖同等配置列表、dry-run、并发和 resume 需求
- **THEN** 固定 GPU shell MAY 被删除
- **AND** 删除 MUST 同步更新 inventory、docs 和架构边界测试

#### Scenario: overlay YAML 有删除触发条件
- **WHEN** `configs/scene31/templates/`、`configs/fusion/u_mask_beam_jepa_*.yaml` 或 protected MMW/CSI YAML 被保留
- **THEN** inventory 或 docs MUST 记录其 local/manual owner 和删除触发条件
- **AND** 这些 YAML MUST 不作为 root canonical fusion 入口

### Requirement: 本地实验结论沉淀后删除临时配置
当 local/manual 实验的关键结论、指标 provenance 和 caveat 已进入 result registry、experiment matrix 或报告文档时，对应临时 queue overlay 和脚本 MUST 收敛为删除或历史记录。

#### Scenario: 结果进入 registry
- **WHEN** Scene31/RBMA/M2Beam 本地实验的 promoted 或 pending claim 已写入 `docs/result_claims_registry.md`
- **THEN** 只服务该结论的临时脚本或 overlay MAY 被删除
- **AND** 保留的复跑路径 MUST 指向 package CLI、owner module 或明确的 local/manual runner

#### Scenario: checkpoint 占位不可升级 claim
- **WHEN** strong-encoder overlay 引用本地 `outputs/scene31/best_checkpoints/*.pth` 占位
- **THEN** 该配置 MUST 保持 local/manual 或 blocked/pending 状态
- **AND** 文档 MUST 不把缺 checkpoint 的路径声明为可复现 mainline claim

### Requirement: CLI 和脚本入口健康检查
项目健康护栏 MUST 检查 CLI 和 scripts 入口不变厚。检查 MUST 基于 `pyproject.toml`、真实脚本路径、current docs、project surface inventory 或 focused tests 中的最小入口事实，拒绝未登记 current 入口、恢复 Python thin alias 和明显复制 workflow 逻辑的脚本。

#### Scenario: 新脚本缺少 owner module
- **WHEN** `scripts/`、`tools/analysis/` 或 package CLI 新增 current 入口
- **THEN** 变更 MUST 在 project surface inventory、README/docs 或 OpenSpec tasks 中登记 owner module、responsibility 和 output boundary
- **AND** 缺少登记时架构边界测试 MUST 失败

#### Scenario: 脚本入口包含训练循环 marker
- **WHEN** 保留的 `scripts/` research diagnostic、dataset preparation、config generator 或 local/manual helper 包含大段训练循环、模型 forward、optimizer step 或重复 package CLI 主逻辑
- **THEN** 健康检查 MUST 失败或要求重新分类为 owner module
- **AND** 修复路径 MUST 是委托包内实现或创建正式 package module

### Requirement: 未分类脚本和配置必须被结构检查发现
项目健康护栏 MUST 验证 current 脚本、root fusion YAML、experiment YAML 和 root 文档的分类与真实文件系统一致。检查 MUST 优先读取 pyproject、真实路径、inventory 和 OpenSpec lifecycle，而不是维护重复的大型 allowlist。

#### Scenario: root fusion YAML 未登记
- **WHEN** `configs/fusion/*.yaml` 中存在 inventory 未分类的实体 YAML
- **THEN** 架构边界检查 MUST 失败
- **AND** 失败信息 MUST 要求迁移、删除或在 inventory 中登记其 root 保留理由

#### Scenario: 新脚本未登记
- **WHEN** `scripts/` 下新增 Python/shell 文件且不属于 ignored cache
- **THEN** 架构边界检查 MUST 要求 inventory 或 current docs 记录其 lifecycle、owner 和输出边界
- **AND** 未登记脚本 MUST 不通过测试静默进入 current surface

### Requirement: 本地脚本和配置分类漂移必须被检查
项目健康护栏 MUST 检查 `scripts/`、`configs/scene31/`、local/manual experiment YAML 和诊断 manifest 的 lifecycle 分类与真实文件系统一致。新增或保留的 local/manual surface MUST 有 owner、输出边界、是否推荐、删除触发条件和 focused 验证说明。

#### Scenario: 新 local script 未登记
- **WHEN** `scripts/` 下存在新的 Python 或 shell 文件
- **THEN** 架构边界检查 MUST 要求 inventory、OpenSpec tasks 或 current docs 记录其 lifecycle 和输出边界
- **AND** 未登记脚本 MUST 不作为 current surface 静默通过

#### Scenario: 固定 GPU shell 不作为 package workflow
- **WHEN** 新增脚本只固定 GPU 映射、日志目录和一组本地 YAML
- **THEN** 健康护栏 MUST 要求删除该脚本或改为一次性本地命令
- **AND** README quickstart MUST 不把它升级为 package CLI 或长期 workflow

### Requirement: 旧实验 facade 不得回流实现
项目健康护栏 MUST 防止旧实验 facade 或兼容 reader 重新承载大段实现。若 `cnn_hybrid_jepa_visual_prior_sweep`、`jepa_gps_shortcut_benchmark` 或 MMW preparation facade 被保留，它们 MUST 只暴露必要公开入口或委托 owner 模块。

#### Scenario: 兼容 reader 重新变厚
- **WHEN** 旧 full sweep 兼容模块新增训练 runner、job graph、shell generation 或 cleanup 逻辑
- **THEN** 架构边界检查 MUST 失败
- **AND** 修复路径 MUST 是迁回当前 owner、删除旧逻辑或更新 OpenSpec 明确恢复该 workflow

#### Scenario: CLI glue 允许薄委托
- **WHEN** package CLI 只解析参数并调用当前 owner module
- **THEN** 健康护栏 MUST 允许该引用
- **AND** 允许范围 MUST 不扩展到内部 runtime 模块从 facade 导入 helper

### Requirement: Scripts are classified before retention
`scripts/` 和 `tools/analysis/` 中保留或新增的入口 MUST 明确分类为数据准备、研究诊断、config generator、figure helper、local/manual experiment helper 或 package CLI 缺口补充。重复 package CLI 的 Python thin alias 和固定 GPU queue shell MUST 删除；local/manual helper MUST 不被 README、AGENTS、docs 或 OpenSpec 写成长期推荐入口。

#### Scenario: 新脚本有 lifecycle
- **WHEN** 本 change 保留、新增或修改 `scripts/*.py`、`scripts/**/*.py` 或本地运行入口
- **THEN** inventory 或 tasks MUST 记录该脚本 owner、lifecycle、是否 local/manual、输出边界和替代 package CLI
- **AND** 架构边界测试 MUST 拒绝未分类长期脚本入口

#### Scenario: Thin alias 被删除
- **WHEN** 脚本只解析参数后调用已有 package CLI 或包内 CLI 的同名 main
- **THEN** pyproject console script 或包内 CLI MUST 成为推荐入口
- **AND** 该 thin alias MUST 删除或被明确标注为短期 local/manual helper 并登记删除条件

### Requirement: Local experiment orchestration cannot become hidden public API
本地批量实验、night-grid、next-round、seed sweep、fresh eval 汇总或类似脚本 MAY 保留为 local/manual workflow，但必须声明不作为稳定 public API。它们 MUST 写入 ignored outputs/logs，且不得提交 checkpoint、metrics、fresh eval 结果或真实运行产物。

#### Scenario: Local manual runner
- **WHEN** local/manual runner 生成或消费 Scene31、RBMA、BTAPA、night-grid 或 next-round 配置
- **THEN** 脚本 MUST 支持 dry-run 或无副作用 sanity path
- **AND** 文档 MUST 指向输出边界并说明真实训练产物不提交

### Requirement: CLI glue stays thin
Package CLI 文件 MUST 只承担参数解析、配置覆盖、轻量 IO、调用 owner module 和 user-facing exit code。真实 workflow、training loop、evaluation loop、dataset preparation、benchmark suite、report builder 或 paper table 生成主逻辑 MUST 位于 owner module。

#### Scenario: 修改 package CLI
- **WHEN** 本 change 修改 `src/kd_sensing/cli/` 下入口
- **THEN** CLI 文件 MUST 不复制训练、评估、dataset parsing、benchmark aggregation、report aggregation 或 paper table 生成主逻辑
- **AND** 对应 `kd-sensing-* --help` 或包内 CLI smoke MUST 继续可运行

### Requirement: Package console scripts 必须有生命周期锚点
`pyproject.toml` 中保留的 `kd-sensing-*` console script MUST 在机器可读 surface 清单和 inventory 中记录 lifecycle、owner、职责、输出边界和 focused validation。保留 public CLI MUST 同时具备 pyproject entry point、help smoke 或等价无副作用 smoke、inventory/docs/OpenSpec current 引用，以及 owner/output-boundary 说明。删除或降级 public CLI 时，项目 MUST 不新增同名 alias、compat wrapper、deprecation trampoline 或旧命令 fallback。

#### Scenario: public CLI 分类完整
- **WHEN** 开发者修改 `[project.scripts]`
- **THEN** 每个 `kd-sensing-*` entry point MUST 出现在 `src/kd_sensing/diagnostics/cli_surface.py`
- **AND** `docs/project_surface_inventory.md` MUST 为该命令记录 lifecycle、owner、职责、输出边界和 focused validation

#### Scenario: public help smoke 覆盖完整
- **WHEN** 开发者运行 CLI help smoke
- **THEN** `tests/test_cli_help.py` MUST 覆盖所有保留的 public console scripts
- **AND** `--help` MUST 不读取真实 `dataset/`、不加载 checkpoint、不启动训练、不写 runtime outputs

#### Scenario: 删除 public CLI 不留 wrapper
- **WHEN** 一个 `kd-sensing-*` console script 被删除或降级为 internal-only
- **THEN** current docs MUST 不再把旧命令描述为当前 public entrypoint
- **AND** 项目 MUST 不提供等价旧命令 wrapper 或 alias

### Requirement: Module-only CLI must be public or deleted
`src/kd_sensing/cli/*.py` 中可直接运行的 module-only CLI MUST 要么在 `pyproject.toml` 声明为 `kd-sensing-*` console script 并出现在 README/current docs 或 current spec 中，要么从当前支持面删除。Shared helper 例如 `cli/common.py` MAY 保留，但 MUST 不提供独立 `main()` 入口或用户可见 workflow。

#### Scenario: Hidden CLI cleanup
- **WHEN** 一个 `kd_sensing.cli.<name>` 模块包含 `main()` 或 console-style parser
- **THEN** 项目 MUST 在 `pyproject.toml` 声明对应 console script，或删除该 CLI wrapper
- **AND** current docs MUST 不推荐未声明 console script 的隐藏 `python -m kd_sensing.cli.<name>` 入口

#### Scenario: Shared CLI helper
- **WHEN** 一个 CLI 模块只提供配置加载、argparse helper 或 shared exit handling
- **THEN** 它 MAY 保留为 internal helper
- **AND** 架构边界测试 MUST 不把它当成 public runnable entrypoint

### Requirement: Local/manual scripts are removable unless explicitly retained
`scripts/` 下本地研究、固定 GPU queue、one-shot 分析和 shell orchestration MUST 具备 current lifecycle、owner、输出边界和删除条件；没有 current docs/spec/result registry 引用、没有替代价值或已有 package CLI 覆盖的脚本 MUST 删除。

#### Scenario: Script has no current owner
- **WHEN** tracked `scripts/*.py`、`scripts/**/*.py` 或 `scripts/*.sh` 不属于 dataset preparation、config generator、current research diagnostic 或 explicitly retained local/manual runner
- **THEN** 该脚本 MUST 从源码表面删除
- **AND** README、docs 和 OpenSpec MUST 不把它描述为当前入口

#### Scenario: Script duplicates package CLI
- **WHEN** 一个脚本只包装 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess` 或其它 package console script
- **THEN** 该脚本 MUST 删除
- **AND** 用户文档 MUST 指向 package console script

### Requirement: Scene31/Scene31-34 报告脚本必须分类为本地研究报告表面
Scene31-34 final analysis 的论文表格、per-scene summary、profile 和 final conclusion MAY 保留为 research diagnostic 或 local/manual reporting surface，并 MUST 在 inventory 或 current 文档中登记 lifecycle、职责和输出边界。旧 Scene31 baseline-pack/next-round/shared summary 已退役，MUST 不再作为 current report surface。

#### Scenario: Scene31-34 报告 owner 有输出边界
- **WHEN** 项目保留 Scene31-34 final analysis owner
- **THEN** inventory MUST 说明其读取本地 summary、fresh-eval 或 paper table 输入
- **AND** 输出边界 MUST 限定在 ignored `outputs/`、`logs/` 或显式用户路径

#### Scenario: 旧 Scene31 report 不升级为 package CLI
- **WHEN** README、AGENTS、OpenSpec 或 docs 描述当前推荐入口
- **THEN** 它们 MUST 不推荐旧 Scene31 baseline-pack/next-round/shared summary
- **AND** protected Scene31-34 local/manual owner MUST 不被描述为 package CLI

### Requirement: Scene31-34 encoder ablation 入口必须合并
Scene31-34 encoder ablation MUST 使用一个参数化 generator 和一个 family/manifest 驱动 runner 承担 TinyViT、PatchViT 或后续 strong encoder ablation。项目 MUST 不按 encoder family 复制同构 Python generator、shell runner 或 fixed-GPU orchestration。保留入口 MUST 被登记为 local/manual experiment surface，且不得成为 package CLI 或 current quickstart 唯一入口。

#### Scenario: TinyViT 和 PatchViT 共用生成逻辑
- **WHEN** 开发者需要生成 TinyViT 或 PatchViT Scene31-34 ablation 配置
- **THEN** implementation MUST 通过同一个 generator owner 或同一组共享 helper 生成 family-specific YAML
- **AND** 测试 MUST 覆盖至少 TinyViT 与 PatchViT 的最小 manifest 或 dry-run 输出结构

#### Scenario: 不新增 PatchViT 专用 runner
- **WHEN** PatchViT ablation 需要本地运行入口
- **THEN** implementation MUST 复用 family/manifest 驱动 runner 或当前统一 runner
- **AND** 项目 MUST 不新增 `run_scenes31_34_patchvit_ablation.sh` 或等价固定 family shell wrapper

#### Scenario: 旧重复入口删除或降级
- **WHEN** 统一 encoder ablation owner 已覆盖旧 TinyViT/PatchViT 入口
- **THEN** implementation MUST 删除旧重复脚本、将其降级为明确 historical 说明，或保留一个薄 local/manual owner
- **AND** inventory、tests 和 docs MUST 不同时推荐多个等价 encoder-family 入口

### Requirement: Final polish 与 presentation helper 必须有生命周期处置
`export_scene31_34_presentation_artifacts.py`、`run_final_scene31_34_polish.sh` 以及等价 final/presentation helper MUST 被删除或登记为 local/manual analysis helper。若保留，记录 MUST 包含 owner、输入、输出、仍需运行的交付场景、删除触发条件和 focused 验证；若删除，current docs、tests 和 OpenSpec references MUST 同步移除或改为 historical 说明。

#### Scenario: 当前论文交付仍需要 helper
- **WHEN** final polish 或 presentation helper 仍被当前论文交付、组会材料或 claim provenance 使用
- **THEN** implementation MUST 在 inventory 或等价文档中将其登记为 local/manual analysis helper
- **AND** README quickstart、package CLI 和主线训练文档 MUST 不把该 helper 描述为 canonical workflow

#### Scenario: helper 不再需要
- **WHEN** final polish 或 presentation helper 不再被 current docs、OpenSpec specs、tests、claim provenance 或交付材料引用
- **THEN** implementation MUST 删除对应脚本和只服务该脚本的测试
- **AND** 有价值的结论、caveat 或输出说明 MUST 先沉淀到 docs、paper table provenance 或报告文件

### Requirement: Historical report helper 不得冒充当前 workflow
Scene31/Scene31-34 paper table、per-scene summary、final conclusion 和一次性报告 helper MUST 明确区分 current paper export owner、local/manual report helper 与 historical artifact。当前只应保留必要的最终表格导出路径和仍有复现价值的分析 helper；重复、过期或已沉淀结论的一次性脚本 MUST 删除或标记 historical。

#### Scenario: 最终表格路径唯一
- **WHEN** 多个脚本都能导出 Scene31/Scene31-34 paper table 或 final conclusion
- **THEN** implementation MUST 指定一个 current export owner 或明确这些脚本分别属于不同 local/manual analysis surface
- **AND** docs 和 tests MUST 不把多个等价脚本同时列为推荐最终表格入口

#### Scenario: 历史报告脚本删除前沉淀结论
- **WHEN** 删除只服务历史 sweep 汇总、per-scene 复盘或一次性 conclusion 的脚本
- **THEN** implementation MUST 保留仍有价值的结果解释、限制条件和替代入口说明
- **AND** 删除 MUST 不要求重新运行历史训练或读取本地 `outputs/` 作为源码迁移步骤

### Requirement: Local/manual 入口不得升级为隐藏 public API
缺失模态主线清理后，保留在 `scripts/` 下的 local/manual runner、report helper、config generator 或 shell orchestration MUST 不作为隐藏 public API。它们 MUST 有 lifecycle 分类，并且不得被 package CLI、README quickstart、核心训练入口或 config loader 当作必需依赖。

#### Scenario: package CLI 不依赖 local/manual 脚本
- **WHEN** 用户运行 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess` 或当前 package diagnostics
- **THEN** 这些 workflow MUST 不要求导入或执行 Scene31/Scene31-34 local/manual scripts
- **AND** 删除 local/manual 脚本 MUST 不破坏核心 package CLI help

#### Scenario: fixed GPU shell 不回流
- **WHEN** 后续 change 新增固定 GPU、固定 seed 队列或单一 family shell orchestration
- **THEN** architecture/surface 检查 MUST 要求删除、合并到 manifest runner，或登记为短期 local/manual 并写明删除触发条件
- **AND** 该 shell MUST 不出现在 current README quickstart 或 package CLI smoke list 中

### Requirement: Wrapper 删除需要 focused guardrail
删除 wrapper 后，项目 MUST 通过轻量 architecture boundary 或 focused tests 防止同职责 wrapper 回流。保留的 scripts MUST 有明确 owner、输入输出边界和删除条件，且 package CLI MUST 不依赖 local/manual script。

#### Scenario: 结构检查拒绝 wrapper 回流
- **WHEN** 开发者运行 architecture boundary 或 compile check
- **THEN** 检查 MUST 拒绝 `sys.path` script-to-script import、模块全局 monkeypatch 和只转发默认参数的 wrapper
- **AND** 检查 MUST 不要求 scripts lifecycle doctor 存在

### Requirement: 本地报告脚本必须在结论沉淀后退出 current surface
项目 MUST 将一次性研究分析、论文表格排版、展示材料导出和局部结论脚本视为有生命周期的本地报告面。若其结论、输入来源和关键输出已经沉淀到 docs、claim notes、paper tables 或 retained artifact 说明，implementation MUST 删除脚本或合并到明确 owner，而不是继续把单次脚本保留为 current entrypoint。

#### Scenario: 一次性 analysis 脚本删除
- **WHEN** `scripts/analysis/` 中的脚本只复现已经沉淀的分析结论
- **THEN** 项目 MUST 删除该脚本或将其标记为非 current retained artifact source
- **AND** docs、OpenSpec current specs、tests 和 inventory MUST 不再要求该脚本路径存在

#### Scenario: 报告脚本合并不新增 wrapper
- **WHEN** 多个报告脚本只有输入 glob、标签或输出格式不同
- **THEN** 项目 MUST 收敛到一个 owner command 或 owner module helper，并通过显式参数表达差异
- **AND** 项目 MUST 不新增 alias、compat wrapper、deprecation trampoline 或同职责转发脚本

### Requirement: 删除本地报告脚本必须保留证据链
删除报告脚本前，implementation MUST 保留足够证据说明该脚本产生的结论仍可追溯。证据 MAY 是正式 docs、paper table、claim note、retained artifact manifest、或 canonical command 的输出契约。

#### Scenario: 删除前记录替代 owner
- **WHEN** 一个本地报告脚本从 current surface 删除
- **THEN** `docs/project_surface_inventory.md` 或相关 docs MUST 记录替代 owner、历史用途或 retained-with-reason
- **AND** 若输出支撑正式 claim，字段名、排序、筛选条件或 artifact 路径模式 MUST 可在替代 owner 中验证

### Requirement: 薄诊断 wrapper 必须收敛到领域 owner
项目 MUST 不长期保留只调用同一领域 owner 的 plot、compare、visualize、recommend 或 prepare wrapper。若 wrapper 没有独立输入契约、输出 schema 或 claim gate，它 MUST 合并为领域 owner CLI 的 subcommand、mode flag 或 documented command recipe。

#### Scenario: wrapper 只有转发职责
- **WHEN** 一个 CLI 或 script 只解析少量参数并调用同一 owner function
- **THEN** implementation MUST 将该行为迁到 owner CLI 或 owner module mode
- **AND** 删除旧 wrapper 时 MUST 不新增 alias、compat wrapper 或 fallback console script

#### Scenario: consolidated help 替代旧入口
- **WHEN** 旧 wrapper 被删除
- **THEN** `--help`、README、docs、OpenSpec current specs 和 inventory MUST 指向 consolidated owner command
- **AND** CLI help tests MUST 覆盖用户需要的新 mode 或 subcommand

### Requirement: 薄 wrapper 保留必须有 retained-with-reason
若某个诊断 wrapper 不能合并，项目 MUST 在 inventory 或 current spec 中记录保留理由、独立契约和删除触发条件。

#### Scenario: wrapper 仍承载独立契约
- **WHEN** wrapper 拥有独立输出 schema、claim evidence role、外部复现实验契约或不同 failure semantics
- **THEN** implementation MAY 保留该 wrapper
- **AND** retained-with-reason MUST 指明为什么 owner CLI mode 不足以替代它

### Requirement: Post-C2 public CLI 必须收敛到主线、MMW 和治理入口
项目在 post-C2 清理后 MUST 只声明十个 public console scripts：train、evaluate、preprocess、runs、runtime cleanup、runtime organize、paper export、U-Mask eval matrix、MMW GPS v2 和 MMW physics inspect。Research dashboard/preview、project surface doctor、architecture summary、training throughput、dataset/source audit 和历史复现 CLI MUST 不再作为 public console script。

#### Scenario: 删除 CLI 同步所有 current references
- **WHEN** implementation 从 pyproject 删除 dashboard、preview 或 surface doctor
- **THEN** README、docs、current specs、CLI help smoke 和 inventory MUST 同步删除 current command reference
- **AND** 删除后项目 MUST 不提供同名 console script、module alias 或 thin wrapper

#### Scenario: 保留 CLI 有生命周期锚点
- **WHEN** post-C2 清理完成
- **THEN** 十个保留命令 MUST 在 pyproject 与 inventory 中有 owner、输出边界和 focused validation
- **AND** 它们 MUST 不依赖已删除 script、dashboard 或 historical config

### Requirement: MMW 入口必须继续可发现
MMW 相关 package CLI、数据准备入口和必要 local/manual helper MUST 在 post-C2 清理中保留生命周期说明。删除其它非主线入口时 MUST 不让 MMW users 失去当前推荐运行、plot、compare、inspect 或 preparation 路径。

#### Scenario: MMW public CLI 保留
- **WHEN** implementation 更新 public CLI lifecycle
- **THEN** `kd-sensing-mmw-town-gps-v2` 和 `kd-sensing-inspect-mmw-physics` 或等价 MMW current CLI MUST 保留，除非另有独立 MMW change 替代
- **AND** README/docs MUST 继续指向 MMW current package CLI，而不是退回已退役脚本

#### Scenario: MMW local helper 不被误判
- **WHEN** `scripts/` 或 `scripts/mmw/` 中的 helper 仍服务 MMW 数据准备或 label distribution 诊断
- **THEN** inventory MUST 将其分类为 MMW dataset preparation、research diagnostic 或 local/manual helper
- **AND** 架构边界检查 MUST 不仅因其位于 `scripts/` 就要求删除

### Requirement: 一次性脚本删除必须保留结论
只服务历史 sweep、人工复盘、临时诊断或旧 runbook 的 `scripts/` 文件 MUST 被删除或降级为 historical/local manual。删除时 MUST 把仍有价值的结论、caveat、复跑方式或替代入口记录到 current docs、inventory、claim registry 或 mainline experiment history。

#### Scenario: 历史分析脚本删除
- **WHEN** implementation 删除 `scripts/analyze_*`、`scripts/summarize_*`、`scripts/diagnose_*` 或旧 Scene31 shell runbook
- **THEN** deletion ledger MUST 记录该脚本不属于主线、MMW、current docs/specs/tests 或 claim provenance
- **AND** 若脚本产出的结论仍被论文或组会材料需要，MUST 先迁移摘要到 docs 或 claim notes

#### Scenario: final C2 和主线 helper 保留
- **WHEN** `scripts/` 中的 launcher、summary 或 helper 被 final C2、当前缺失模态主线或 protected YAML/manifest 消费
- **THEN** implementation MUST 保留该脚本或先提供等价 owner
- **AND** 删除候选 MUST 标记为 protected-mainline，而不是 historical one-shot

### Requirement: Temporal 和历史 launcher 不得派生平行 script suite
H5/P1 temporal matrix 已覆盖的 check/launch/eval/summary 行为 MUST 通过现有参数化脚本使用。项目 MUST 不保留通过 `sys.path` 注入、脚本私有函数导入或模块全局变量改写派生的 S1-S4 parallel wrappers；历史 overnight launcher 在结果冻结后 MUST 退出 current script surface。

#### Scenario: S1-S4 wrapper 删除
- **WHEN** temporal router S1-S4 tasks 被 defer
- **THEN** 三个 S1-S4 wrapper 和专属 tests MUST 删除
- **AND** H5/P1 launcher 的用户改动 MUST 保留

#### Scenario: 历史 launcher 退出但 summary 保留
- **WHEN** overnight training matrix 只剩历史结果复盘价值
- **THEN** launcher MUST 删除
- **AND** 仍被 final C2 summary 消费的 read-only summary helper MAY 保留

### Requirement: On-disk script surface 必须完整分类
`scripts/` 下受控 `.py` 和 `.sh` 入口 MUST 被 lifecycle inventory 恰好覆盖一次，不论文件当前是否已被 Git 跟踪。每个 lifecycle 记录 MUST 包含 owner、保留原因、public/recommended relation、output boundary、focused validation 和 deletion condition。

#### Scenario: 未跟踪实验脚本未登记
- **WHEN** on-disk `scripts/` 出现未被任何 lifecycle 行匹配的 Python 或 shell 文件
- **THEN** architecture/compile guard MUST 失败并报告路径
- **AND** 文件 MUST 在删除或登记前不能通过 full verification

#### Scenario: Script 被多条规则匹配
- **WHEN** 某脚本同时匹配多个 lifecycle family 或精确行
- **THEN** guard MUST 失败并报告 duplicate classification

#### Scenario: 一次性 campaign 已完成
- **WHEN** local/manual script 的结论已进入 history/claim 且 deletion condition 成立
- **THEN** script MUST 从 current surface 删除或明确续期保留理由
- **AND** 系统 MUST NOT 为它新增 package wrapper

