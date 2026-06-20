## MODIFIED Requirements

### Requirement: 主线模型目录
项目 MUST 维护当前主线模型目录，用于集中说明每条当前主线、baseline/control 和诊断 workflow 的研究问题、模型边界、配置入口、数据口径、指标口径、运行状态和结果引用。该目录 MUST 位于 `docs/mainline_model_catalog.md` 或等价 current 文档中，并 MUST 不把 retired、historical、supporting 或 mock-only workflow 描述为当前推荐入口。

#### Scenario: 主线模型目录覆盖当前支持面
- **WHEN** 开发者阅读主线模型目录
- **THEN** 文档 MUST 至少覆盖 Image+GPS JEPA GPS-biased reuse、GPS-query pooling、supervised/random controls、Vision-Position baseline suite、Arnold22 Camera AE+GPS Direct 本地 substitute、BEV-Fusion 2604、MMW GPS v2、CSI hardening、JEPA visual analysis、GPS shortcut benchmark 和其它保留 workflow
- **AND** 每行 MUST 标明对应 config、入口命令或诊断入口、数据集/场景、split/target、metric profile、运行状态和主要 caveat

#### Scenario: 退役路线不得进入 current 主线表
- **WHEN** 主线模型目录提到 KD、HiST/Hist、Raymobtime s008、standalone Top8 selector、GPS residual、camera residual、BGAM、viewer manifest、Gradio viewer、CRAF/MARF/G2D 或 Multimodal-NF
- **THEN** 对应行 MUST 标记为 retired、historical 或 migration guard
- **AND** 文档 MUST 不提供这些路线的当前推荐训练、评估或诊断命令

### Requirement: 实验协议和参数表
项目 MUST 维护实验协议和参数表，用于将主要配置族的正式口径、smoke/debug/lowmem 口径、upper-bound 口径和历史 ablation 口径分开。该表 MUST 位于 `docs/experiment_protocols.md` 或等价 current 文档中，并 MUST 能让读者不打开多个 YAML 也能判断实验是否可横向比较。

#### Scenario: 参数表记录可比性字段
- **WHEN** 文档列出一个实验配置族
- **THEN** 表格 MUST 记录 config path、run status、dataset/scenes、split protocol、selection split、seed、epochs、batch size、learning rate、seq_len、num_pred、target source、GPS feature mode、label space、metric profile、输出目录和 focused validation 命令
- **AND** 对 smoke、debug、lowmem、upper-bound、mock 或 historical ablation 条目 MUST 显式标记状态

#### Scenario: 退役 BGAM 和 viewer 配置状态清晰
- **WHEN** 表格提到 DeepSense6G/MMW BGAM、viewer manifest、Gradio viewer、JEPA shortcut benchmark 或 difficulty profile
- **THEN** 文档 MUST 标明 BGAM 和 viewer manifest/Gradio viewer 为 retired 或 historical
- **AND** 文档 MUST 不把 BGAM 或 viewer manifest 输出写成当前正式实验产物
