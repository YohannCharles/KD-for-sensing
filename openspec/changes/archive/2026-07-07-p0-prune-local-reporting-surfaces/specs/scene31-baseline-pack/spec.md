## ADDED Requirements

### Requirement: Scene31 baseline pack 汇总可接入共享 summary owner
Scene31 baseline pack 的 summary 和 paper-facing export MAY 由共享 Scene31 summary owner 生成。迁移 MUST 保持 baseline pack 的输入发现、输出字段、方法标签和 artifact 命名契约，除非 current spec 同步修改。

#### Scenario: baseline pack summary owner 替代专用脚本
- **WHEN** baseline pack summary 专用脚本只复制共享读取、聚合或输出逻辑
- **THEN** implementation MAY 删除该专用脚本并改用共享 Scene31 summary owner
- **AND** baseline pack docs/tests MUST 指向共享 owner 的明确 profile 或 group

#### Scenario: baseline pack 输出保持 claim 可用
- **WHEN** baseline pack summary 迁移到共享 owner
- **THEN** paper table、claim note 或 reliability summary 所需字段 MUST 继续存在
- **AND** 迁移验证 MUST 覆盖至少一个 baseline pack artifact fixture 或 dry-run fixture
