## ADDED Requirements

### Requirement: Post-C2 清理必须保护主线与 MMW
项目在 post-C2 表面积清理中 MUST 先建立 protected inventory，再删除源码、配置、脚本、测试或文档入口。protected inventory MUST 至少覆盖 final C2 / U-MaskBeamJEPA 缺失模态主线、当前 claim/evidence 输入、MMW/CSI 数据集与 workflow、主线 YAML/manifest、以及 U-MaskBeamJEPA 已存在 fusion 分支实现。

#### Scenario: protected inventory 阻止误删
- **WHEN** implementation 准备删除任意 `src/`、`configs/`、`scripts/`、`tests/` 或 docs 文件
- **THEN** 删除候选 MUST 先在 protected inventory 或 deletion candidate ledger 中分类
- **AND** 属于 final C2、U-MaskBeamJEPA 主线、MMW/CSI、current claim/evidence 或主线 YAML/manifest 的文件 MUST 不进入删除任务

#### Scenario: MMW 支线保留
- **WHEN** implementation 执行 post-C2 表面积清理
- **THEN** `src/kd_sensing/data/mmw/`、`src/kd_sensing/data/datasets/mmw*.py`、MMW GPS v2、physics-informed MMW、CSI hardening、MMW/CSI configs、MMW/CSI tests 和相关 package CLI MUST 保留
- **AND** 文档 MUST 将 MMW 标记为 future dataset workflow 或 current supporting workflow，而不是删除候选

### Requirement: 主线 YAML 和 manifest 删除必须有证据
项目 MUST 保护仍被主线使用的 YAML、CSV、JSON manifest 或 config reference。删除或生成化配置前，implementation MUST 证明该文件不被 final C2、current Scene31/Scene31-34 evidence、claim registry、experiment matrix、OpenSpec current spec、focused tests 或用户标记的主线输入消费。

#### Scenario: current evidence config 被保护
- **WHEN** YAML 或 manifest 被 claim registry、experiment matrix、mainline docs、OpenSpec current spec、final C2 launcher、focused tests 或用户标记引用
- **THEN** implementation MUST 保留该文件
- **AND** 若后续必须删除，MUST 先把 provenance 更新到等价 generator、base config 或 manifest 输入

#### Scenario: historical generated config 可删除
- **WHEN** YAML 或 manifest 只服务历史 sweep，且可由 generator/template 无损重建，并且不属于 current claim/evidence
- **THEN** implementation MAY 删除实体文件
- **AND** deletion candidate ledger MUST 记录 generator、替代入口、验证命令和回滚方式

### Requirement: U-MaskBeamJEPA fusion 分支本轮不得删除
Post-C2 表面积清理 MUST 不删除或修改 U-MaskBeamJEPA 中已存在的 fusion、router、forward 或 loss 分支实现。未胜出分支 MAY 在 inventory 中标记为后续审计候选，但删除 MUST 通过独立 OpenSpec change 再实施。

#### Scenario: fusion 分支保持源码存在
- **WHEN** implementation 触碰 `src/kd_sensing/models/u_mask_beam_jepa.py` 或 U-Mask loss/config helper
- **THEN** 本 change MUST 不删除 `pcpg`、`bprr`、`raw_conf_gate`、`weighted_sum`、`concat_mlp`、`supervised_router` 或既有 branch/loss 开关
- **AND** focused tests MUST 继续覆盖保留分支的至少 smoke 或 registry/config 行为，除非后续独立 change 明确退役

### Requirement: 删除必须按可回滚波次执行
Post-C2 清理 MUST 分波次实施，每波只处理一个稳定边界：保护扫描、入口/文档收口、一次性脚本删除、非主线源码测试删除、历史配置收缩、guardrail 收口。每波 MUST 记录验证命令和失败回滚方式。

#### Scenario: wave 完成后验证
- **WHEN** 某个删除 wave 完成
- **THEN** implementation MUST 运行该 wave 对应的 focused validation
- **AND** 若验证失败，implementation MUST 能恢复本 wave 删除的文件，而不是继续叠加下一波删除

#### Scenario: 不创建兼容入口
- **WHEN** 某个非主线入口被删除
- **THEN** 项目 MUST 不新增同名 wrapper、stub CLI、virtual config、package facade 或 compatibility alias
- **AND** 文档 MUST 指向保留主线入口、MMW 入口、historical note、普通 unknown-name 错误或 retired tombstone
