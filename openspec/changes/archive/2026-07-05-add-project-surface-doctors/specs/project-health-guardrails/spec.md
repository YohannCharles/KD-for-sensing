## ADDED Requirements

### Requirement: 项目表面积 doctor
项目 MUST 提供只读表面积 doctor，用于检查 scripts、configs、hotspots 和文档引用的高风险漂移。Doctor MUST 不删除、不移动、不重写源码、配置、本地数据、输出、日志、cache 或 checkpoint。

#### Scenario: Doctor 只读运行
- **WHEN** 开发者运行项目表面积 doctor
- **THEN** doctor MUST 只读取 tracked 源码、配置、文档、OpenSpec 和必要的 git 文件清单
- **AND** doctor MUST 不修改 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或配置文件

#### Scenario: Doctor 输出可定位问题
- **WHEN** doctor 发现未分类脚本、失效 config 引用或热点超出登记边界
- **THEN** 输出 MUST 包含文件路径、问题类型、引用的权威来源和建议验证命令

### Requirement: Doctor 可纳入 quick verify
高风险表面积 doctor MUST 可作为 quick verify 的一部分运行，或至少在文档中记录为非平凡脚本/config/hotspot 改动前的推荐检查。

#### Scenario: 入口改动前运行 doctor
- **WHEN** 变更新增或修改 `scripts/`、`tools/analysis/`、`configs/` 或热点 owner
- **THEN** tasks MUST 列出对应 doctor 命令
- **AND** Python 命令 MUST 使用 `conda run -n kd_mm_beam`
