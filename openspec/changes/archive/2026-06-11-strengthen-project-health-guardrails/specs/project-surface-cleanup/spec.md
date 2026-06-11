## ADDED Requirements

### Requirement: 实验配置支持面分类
项目 MUST 对当前支持的配置文件进行生命周期分类，至少区分 canonical/root 推荐入口、实验复现配置、debug/smoke 配置、dataset preparation 配置、diagnostics 配置和已退役历史记录。新增、迁移或删除配置时，项目 MUST 同步更新 inventory、引用文档和相关架构 guardrail。

#### Scenario: 实验子目录配置有归属
- **WHEN** 开发者在 `configs/` 下新增实验特化 YAML
- **THEN** 该配置 MUST 位于语义明确的子目录或被 inventory 分类说明
- **AND** README、docs、OpenSpec 或脚本中的引用 MUST 指向真实存在的路径
- **AND** 该配置 MUST 不通过 root `configs/fusion/` 混入长期 canonical 入口，除非 inventory 明确将其列为当前推荐入口

#### Scenario: 配置引用漂移被发现
- **WHEN** 架构边界测试扫描 README、docs、scripts 和当前 OpenSpec specs 中的配置路径引用
- **THEN** 测试 MUST 能发现指向不存在配置文件的当前支持面引用
- **AND** 历史 archive 或明确标记为退役记录的引用 MUST 不被误判为当前入口

### Requirement: Root 文档支持面分类
项目 MUST 对仓库根目录和 `docs/` 中的长期文档、复现报告、研究笔记和历史方案进行生命周期分类。当前 README MUST 保持快速上手和主 workflow；长期需求与架构约束 MUST 留在 OpenSpec；研究/复现文档 MUST 标明用途和产物边界。

#### Scenario: Root 文档有生命周期
- **WHEN** 开发者查看项目表面积 inventory
- **THEN** inventory MUST 分类说明 README、README_REPRODUCE、环境/数据/报告文档、研究笔记和历史方案文档的当前用途
- **AND** 未分类 root 文档 MUST 被架构边界测试发现或要求补充说明

#### Scenario: 文档不推荐退役入口
- **WHEN** README 或长期 docs 描述当前可运行 workflow
- **THEN** 文档 MUST 不把已退役 KD/HiST/Top8/residual/camera residual 路线描述为当前推荐入口
- **AND** 如需保留历史背景，文档 MUST 明确标记为历史或退役记录
