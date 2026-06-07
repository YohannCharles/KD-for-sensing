## ADDED Requirements

### Requirement: Hist workflow 已从当前实验入口退役
当前训练、评估、quickstart、CLI help、run metadata 和推荐文档 MUST 不再包含 HiST-Beam/Hist LOSO 入口。旧 Hist 配置路径、console script 和 run plan 不得作为当前 workflow 兼容承诺。

#### Scenario: CLI help 不包含 Hist 保留入口
- **WHEN** 开发者执行当前推荐的 CLI help 验证
- **THEN** `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`、`kd-sensing-runs`、`kd-sensing-export-viewer-manifest` 和 `kd-sensing-visualize-modalities` MUST 正常退出
- **AND** 验证 MUST 不要求 `kd-sensing-hist-beam-loso` 存在

#### Scenario: 旧 Hist 配置路径失败
- **WHEN** 用户传入 `configs/hist_beam/quick_smoke.yaml` 或其它 `configs/hist_beam/` 路径
- **THEN** 配置加载 MUST 失败或报告路径已退役
- **AND** 系统 MUST 不生成等价 virtual config

## MODIFIED Requirements

### Requirement: 项目描述反映当前主线
项目元数据、README 和高层文档 MUST 将当前项目主线描述为多模态/少样本跨场景 beam prediction、DeepSense6G/MMW/Raymobtime supervised/adaptation、GPS candidate、residual fusion、CSI hardening、Raymobtime s008 selection 和 viewer manifest，而不是 KD-first 或 HiST-Beam-first 工作流。历史 KD 或 Hist 背景可以保留在 archive 或历史说明中，但必须标记为已退役或历史记录。

#### Scenario: pyproject 描述不再 KD 或 Hist first
- **WHEN** 开发者查看 `pyproject.toml` 的项目 description
- **THEN** description MUST 不把 knowledge distillation 或 HiST-Beam 描述为当前唯一或首要工作流
- **AND** 若提到 KD 或 Hist，MUST 表达其为 legacy、historical 或 retired

#### Scenario: 文档保留历史说明
- **WHEN** README 或 docs 提到历史 KD 或 Hist 代码
- **THEN** 文档 MUST 说明对应能力已从当前 active mainline 退役
- **AND** 文档 MUST 不提供当前推荐运行命令

### Requirement: 当前推荐 workflow 聚焦少样本跨场景主线
README、实验矩阵和 quickstart MUST 将当前推荐 workflow 聚焦于 supervised/adaptation baseline、DeepSense6G GPS candidate、Top8 selector、GPS+LiDAR BGAM、camera residual、MMW Town GPS v2、CSI hardening、Raymobtime s008 selection 和 viewer manifest。KD baseline、HiST-Beam/Hist、模态失衡诊断脚本、objective-aware auxiliary tasks 和 snapshot next-frame MUST 作为 optional、historical 或 retired workflow 描述，不得作为 few-shot cross-scene 默认主线步骤。

#### Scenario: quickstart 不推荐退役脚本
- **WHEN** 开发者阅读 README 或 `docs/experiment_matrix.md`
- **THEN** 文档 MUST 不推荐运行 `kd-sensing-hist-beam-loso`、`configs/hist_beam/*` 或已退役的独立模态诊断脚本
- **AND** 若需要当前主线实验，文档 MUST 指向仍存在的配置化 CLI 或包内 workflow

#### Scenario: optional workflow 与主线区分
- **WHEN** 文档提到 legacy KD、HiST-Beam、snapshot next-frame、occlusion、position 或 multitask objective
- **THEN** 文档 MUST 明确它们不是当前主结论的默认步骤
- **AND** 文档 MUST 不要求先运行这些支线才能执行当前 DeepSense6G/MMW/Raymobtime 主线

### Requirement: 健康检查反映保留入口
快速健康检查 MUST 覆盖当前仍支持的架构边界、包内 CLI、viewer manifest、Raymobtime、modality visual diagnostics 和当前主线 focused tests。健康检查 MUST 不要求已退役的模态失衡诊断脚本、fusion KD virtual alias 或 HiST-Beam/Hist CLI 可用。

#### Scenario: focused validation 不依赖退役入口
- **WHEN** 开发者执行本 change 的 focused 验证
- **THEN** 验证命令 MUST 使用 `conda run -n kd_mm_beam`
- **AND** 命令 MUST 不包含已退役的 Hist CLI、Hist configs 或独立模态诊断脚本
- **AND** 验证 MUST 覆盖配置加载失败、架构边界、registry 和保留 evaluation subset 能力
