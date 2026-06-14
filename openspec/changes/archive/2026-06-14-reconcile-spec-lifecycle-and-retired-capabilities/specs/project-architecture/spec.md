## ADDED Requirements

### Requirement: 当前架构规格遵循 lifecycle 分类
`project-architecture` spec MUST 与 OpenSpec capability lifecycle inventory 保持一致。已经标记为 `retired-tombstone` 的能力 MUST 只作为退役边界、禁止回流、migration guard 或历史背景出现；标记为 `supporting` 的能力 MUST 不被描述为 standalone 当前推荐入口。

#### Scenario: 退役能力不作为当前热点
- **WHEN** `project-architecture` 提到 HiST/Hist、Raymobtime s008、Top8 selector、GPS residual、camera residual、CRAF/MARF/G2D、Multimodal-NF 或旧 KD
- **THEN** 对应段落 MUST 明确其 retired/supporting 语义
- **AND** 文档 MUST 不要求恢复旧 CLI、旧配置、旧 facade 或旧 root script

#### Scenario: 支撑能力指向当前 workflow
- **WHEN** `project-architecture` 提到仍被当前 workflow 复用的支撑代码
- **THEN** 文档 MUST 指向实际 current workflow
- **AND** 文档 MUST 不把支撑代码所属的旧研究路线描述为当前入口

## MODIFIED Requirements

### Requirement: Active mainline 与 legacy KD 模块边界
项目 MUST 区分当前主线方法模块、supporting helper 和 legacy/retired 模块。当前主线包括 supervised beam prediction、Image+GPS JEPA query-pool downstream、paired baseline/control、Vision-Position baseline suite、DeepSense6G/MMW GPS+LiDAR BGAM、MMW GPS v2、CSI hardening、viewer manifest、JEPA visual analysis、GPS shortcut benchmark、soft-label supervised training 和通用训练/评估能力。HiST/Hist、GPS residual、camera residual、standalone Top8 selector、Raymobtime s008、CRAF/MARF/G2D、Multimodal-NF 和旧 KD MUST 不作为 active mainline 描述；若仍有通用 helper 被保留，MUST 标记为 supporting 或迁移边界。

#### Scenario: mainline 导入不触发 KD runtime
- **WHEN** 开发者导入当前主线的训练、评估、BGAM、JEPA downstream、CSI hardening、viewer 或 soft-label helper
- **THEN** 导入 MUST 不构建 frozen teacher runtime
- **AND** 导入 MUST 不解析 teacher checkpoint registry
- **AND** 导入 MUST 不要求 legacy KD baseline 模块可用

#### Scenario: 退役 Hist 不属于 active mainline
- **WHEN** 文档或测试列举 active mainline 方法
- **THEN** 列表 MUST 不包含 HiST-Beam/Hist 专用 CLI、engine、model、evaluation、LOSO executor 或 history-anchor Hist workflow
- **AND** 如提到 Hist 名称，MUST 明确为 retired-tombstone 或禁止回流边界

#### Scenario: 架构测试拒绝 KD 和退役路线回流
- **WHEN** 内部源码新增 active mainline 到 legacy KD runtime 聚合入口或退役路线专属模块的依赖
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 指向 no-KD objective、current workflow、supporting helper 或 migration guard 作为修复路径

### Requirement: 退役旧模态诊断脚本入口
项目 MUST 不再把模态失衡时期的独立模态子集和模态扰动研究脚本作为长期维护入口。通用模态 subset、mask 或 perturbation 调试能力如需保留，MUST 通过包内 CLI、配置化 evaluation pass、viewer manifest、JEPA benchmark、BGAM/CSI 当前 workflow 或明确的内部 helper 承载，并 MUST 在脚本 allowlist 和项目表面积 inventory 中体现当前边界。

#### Scenario: 脚本入口清单不包含旧诊断脚本
- **WHEN** 开发者运行架构边界测试检查 `scripts/` 与 `tools/` 入口清单
- **THEN** `scripts/eval_modality_subsets.py` 和 `scripts/eval_modality_perturbation.py` MUST 不再作为允许的长期入口存在
- **AND** 测试 MUST 继续允许当前保留的 thin CLI alias、dataset preparation、viewer manifest、MMW current workflow、BGAM、CSI hardening 和研究诊断入口

#### Scenario: 通用 subset 能力不被误删
- **WHEN** evaluation 配置启用 `evaluation.modality_subsets`
- **THEN** 系统 MUST 继续能在共享 evaluation pass 中计算配置化 subset metrics
- **AND** 该能力 MUST 不依赖被退役的独立研究脚本

### Requirement: 表面积 inventory 跟随当前主线
项目 surface inventory MUST 将当前推荐入口描述为 Image+GPS JEPA query-pool 主线、paired baseline/control、Vision-Position baseline suite、DeepSense6G/MMW BGAM、MMW GPS v2、CSI hardening、viewer manifest、JEPA visual analysis、GPS shortcut benchmark 和通用训练评估能力。已退役的模态失衡诊断脚本、KD virtual alias、HiST/Hist、Raymobtime s008、standalone Top8 selector、GPS residual、camera residual、CRAF/MARF/G2D 和 Multimodal-NF MUST 不作为新入口或健康检查要求出现。

#### Scenario: inventory 删除旧研究入口
- **WHEN** 开发者阅读 `docs/project_surface_inventory.md`
- **THEN** 文档 MUST 不再把旧模态子集/扰动诊断脚本或退役研究线列为长期维护 research diagnostic/current entry
- **AND** 文档 MUST 保留本地产物边界说明，不要求删除或迁移历史 `outputs/`、`logs/` 或 `dataset/`

#### Scenario: inventory 标注 supporting 能力
- **WHEN** 某个支撑代码仍被 BGAM、benchmark、metrics 或 migration guard 消费，但其 standalone workflow 已退役
- **THEN** inventory MUST 将其描述为 supporting 或支撑代码
- **AND** inventory MUST 不为该旧 workflow 新增 root config、console script 或 quickstart 命令

## REMOVED Requirements

### Requirement: HiST-Beam LOSO executor 热点拆分边界
**Reason**: HiST-Beam/Hist 专用 CLI、engine、model、evaluation 和 LOSO executor 已从当前支持面退役；继续把 Hist executor 作为当前热点会和 README、inventory 以及退役 specs 冲突。
**Migration**: 若未来需要新的跨场景矩阵或 LOSO workflow，MUST 通过新的 current capability 明确定义 CLI、配置、输出和防泄漏边界；现有通用 few-shot/LOSO helper 如仍保留，应归类为 `supporting`，不得恢复 Hist 专用 executor。
