## ADDED Requirements

### Requirement: Ponytail follow-up 表面必须按证据收口
项目在处理 ponytail audit follow-up 时 SHALL 将每个候选分类为 delete、merge、keep 或 local-clean；对 delete 和 merge 候选，MUST 记录调用方证据、owner 替代路径、公开 API 风险、OpenSpec/inventory 消费关系和最小验证命令。

#### Scenario: 跟踪的历史清理清单不留在源码根目录
- **WHEN** 根目录 JSON 文件只是历史清理、审计或迁移 manifest，且没有当前 CLI、测试、OpenSpec 规格或文档索引消费它
- **THEN** 该文件 SHALL 从跟踪源码中删除，后续可再生成清单 MUST 写入本地运行产物位置或 ignored 路径

#### Scenario: 本地生成元数据不作为源码删除
- **WHEN** `src/` 下存在 ignored 的 `*.egg-info`、`__pycache__` 或 pytest/cache 元数据
- **THEN** 清理动作 SHALL 作为 local-clean 处理，MUST NOT 与源码删除、模型权重删除、数据删除或实验输出删除混同

#### Scenario: 低价值 package facade 通过 owner 导入收缩
- **WHEN** 包级 `__init__.py` 只重导出 owner 模块符号，且内部源码或测试可以直接导入 owner 模块
- **THEN** 该 facade SHALL 收缩为轻量 marker 或明确公共入口，内部调用点 MUST 改为 owner 模块导入

#### Scenario: 未登记脚本不进入 current surface
- **WHEN** `scripts/` 下脚本没有被 inventory、README、OpenSpec 或当前工作流引用，且与现有入口重复
- **THEN** 该脚本 SHALL 删除，或将唯一可复用逻辑合并到已登记入口后再更新 inventory

#### Scenario: 根目录历史笔记必须删除或归档
- **WHEN** 根目录 Markdown 文档只记录历史研究背景，且不属于当前快速上手、架构、OpenSpec 或维护导航
- **THEN** 该文档 SHALL 删除、归档，或压缩迁移到合适文档位置，并从 current surface 中移除
