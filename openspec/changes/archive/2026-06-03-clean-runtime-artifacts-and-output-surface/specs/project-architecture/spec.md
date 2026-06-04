## ADDED Requirements

### Requirement: 语义化本地输出目录
项目 MUST 避免新脚本或默认配置继续向语义不清的兜底目录写入实验产物。长期保留的 shell orchestration、诊断脚本和 CLI 默认输出目录 MUST 包含实验族、数据集或能力名称；`outputs/other/` MAY 作为历史清理候选被扫描，但 MUST 不再作为新实验脚本的默认输出根。

#### Scenario: MMW modal15 默认输出目录可识别
- **WHEN** 用户直接运行 MMW modal15 shell orchestration 且未设置 `OUTPUT_ROOT`
- **THEN** 脚本 MUST 默认写入包含 `mmw_sunny_modal15` 或等价实验族名称的 `outputs/` 子目录
- **AND** 帮助文本 MUST 展示该语义化默认路径

#### Scenario: outputs other 不作为新默认值
- **WHEN** 架构边界测试扫描长期保留脚本和配置
- **THEN** 测试 MUST 拒绝新增默认输出根为 `outputs/other`
- **AND** 已存在的历史 `outputs/other/` 本地产物 MUST 只通过清理 manifest 管理

### Requirement: 清理流程不跨越源码边界
项目 MUST 将本地运行产物清理限定在 `.gitignore` 覆盖的本地产物范围内。清理工具、文档和测试 MUST 明确禁止删除源码、配置、文档、OpenSpec artifacts、已跟踪文件、`dataset/` 真实数据和 `All_models/` 历史复现权重。

#### Scenario: 清理 manifest 不含源码删除动作
- **WHEN** 用户生成清理候选 manifest
- **THEN** manifest MUST NOT 将 `src/`、`tests/`、`configs/`、`docs/` 或 `openspec/` 下的已跟踪文件列为可删除候选
- **AND** 如果这些路径被扫描到，manifest MUST 标记为 protected

#### Scenario: 文档说明本地产物边界
- **WHEN** 开发者阅读项目表面积 inventory 或 README
- **THEN** 文档 MUST 说明清理流程先生成 manifest
- **AND** 文档 MUST 说明真正删除需要用户显式确认
