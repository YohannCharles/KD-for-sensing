## ADDED Requirements

### Requirement: Conformal 波束路线保持最小公共表面和本地产物边界
Conformal 诊断与筛选 MUST 作为受协议约束的本地实验模块与工具实现，不得新增 public CLI、canonical MMW recipe、历史 route 或兼容入口。诊断产物 MUST 写入 `outputs/conformal_beam_diagnostic/`，筛选产物 MUST 写入 `outputs/conformal_beam_screen/`；两者 MUST 不被源码或 current spec 读取为运行依赖。

#### Scenario: 无本地实验产物的源码检查
- **WHEN** 仓库在没有 `outputs/conformal_beam_diagnostic/` 与 `outputs/conformal_beam_screen/` 的环境中加载 canonical recipe、CLI help 或执行架构边界测试
- **THEN** 当前 U0、AMBER-Full、RMBP-MM 与 DeepSense6G 的现有公共工作流 MUST 继续可用

#### Scenario: 目标运行目录已存在
- **WHEN** 筛选运行器发现目标 `outputs/conformal_beam_screen/` 子目录已存在
- **THEN** 它 MUST fail closed，MUST 不隐式覆盖或把不完整目录当作完成结果

### Requirement: Conformal 路线复用既有冻结表征而不新增缓存边界
运行器 MUST 复用 `outputs/router_observability/cache` 下已审计的冻结 U0 表征缓存与既有拓扑 manifest，MUST 不构建新的数据集缓存根、不修改 `outputs/cache/MMW`、不删除或覆盖现有缓存。

#### Scenario: 表征缓存缺失或哈希不符
- **WHEN** 表征缓存不存在，或其绑定的 U0 checkpoint SHA256 与筛选配置不一致
- **THEN** 运行器 MUST fail closed，MUST 不以重新训练或重建骨干作为回退

### Requirement: 筛选运行不启动骨干训练或 outer test
Conformal 筛选 MUST 不创建 optimizer、不训练 encoder/fusion/prototype/router、不启动 multi-seed 骨干、不修改系统启动项或凭证文件。它 MUST 不读取 outer test 的样本、标签、统计量或结果。

#### Scenario: 长时间运行的状态记录
- **WHEN** 筛选跨多个种子、切分粒度与 alpha 连续运行
- **THEN** 它 MUST 只向本地 runtime 产物记录简洁状态，MUST 不终止无关 GPU 进程

#### Scenario: 门槛全部失败
- **WHEN** C2--C4 均未通过预注册门槛，或 C5 通过了有效性与条件性门槛
- **THEN** 运行器 MUST 产出负结果报告并停止，MUST 不自动启动 multi-seed 骨干、outer test 或下一轮筛选
