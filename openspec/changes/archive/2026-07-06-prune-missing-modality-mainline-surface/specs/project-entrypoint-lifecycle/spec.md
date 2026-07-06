## ADDED Requirements

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
