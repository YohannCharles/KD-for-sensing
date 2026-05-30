## Context

本 change 来自一次项目表面积和 MMW sensor-assisted 工作流巡检。当前已确认的硬问题是架构边界测试会因为新增 MMW 数据准备脚本和 shell orchestration 未纳入 inventory/allowlist 而失败；同时，sensor-assisted target adaptation 已记录部分 sensitive-field flags，但 labeled target 的 radio/path auxiliary supervision、summary eligibility 和 quick conclusion 排除规则还没有形成端到端约束。

另一个结果风险来自 metric horizon 聚合：主验证指标已经支持 selected metric horizons，但 force-mask subset scalar 仍可能使用 first valid slot，导致同一 run 的主指标与 subset 指标不可直接比较。MMW preflight 还存在训练 executor 调用 dataset 私有 helper 物化 radar CSV 的边界问题，增加了训练入口副作用和维护成本。

本 change 是稳定化与防回归方案，不改变公开训练/评估/预处理 CLI，不要求迁移本地数据，也不把 outputs、logs、cache 或 checkpoint 纳入源码变更。

## Goals / Non-Goals

**Goals:**

- 让项目表面积检查恢复通过，并使新增 MMW 脚本入口具有可审计生命周期。
- 为 sensor-assisted run 形成统一 `main_conclusion_eligible` 与 `eligibility_reasons` 语义，覆盖不允许输入、target sensitive supervision、prototype no-op 和缺失对比 run。
- 让普通 validation、force-mask subset validation 和 standalone evaluate 复用一致的 selected horizon 聚合口径。
- 将 MMW split/radar CSV materialization 暴露为公开数据准备或 split utility，训练 preflight 不再导入 dataset 私有 helper。
- 将继续增长的 HiST-Beam LOSO executor 按 preflight、stage execution、summary/conclusion 和 matrix metadata 拆分出明确迁移路径。

**Non-Goals:**

- 不重新设计 HiST-Beam 模型、loss 数学形式、MMW 数据格式或实验矩阵规模。
- 不删除用户本地数据、训练输出、日志、cache、checkpoint 或已下载 zip。
- 不强制移除所有研究脚本；本 change 只要求保留脚本具备 inventory 分类和边界说明。
- 不把 target radio/path auxiliary supervision 作为默认主结论能力；允许它作为显式 opt-in 的补充/诊断路径，并必须被 eligibility metadata 标记。

## Decisions

### 1. eligibility 作为 run-level 事实字段，而不是只在 conclusion 阶段推断

每个 sensor-assisted adaptation/evaluation run 都写出机器可读字段：`main_conclusion_eligible`、`eligibility_reasons`、`used_target_*_for_training`、`sensitive_field_policy` 和 `sensor_assisted_profile`。训练阶段能确定的泄漏或不合规输入应尽早失败；只有配置显式允许但不适合作为主结论的 target auxiliary supervision，才允许 run 完成并标记不可用于主结论。

备选方案是在 quick conclusion 汇总时临时扫描 flags 并排除。该方案改动小，但会让单个 run artifact 本身无法解释为什么不可比较，也容易被其它分析脚本绕过。

### 2. target sensitive guard 区分 split、budget 和 opt-in policy

unlabeled target 和 label_budget=0 的 target training loss 访问 beam、beam_power、CSI/channel、path、radio 等真实 target supervision 字段时必须失败。label_budget>0 的 labeled subset 可以继续使用 supervised beam loss；path/radio auxiliary supervision 必须有单独 opt-in，并默认让 run 不进入主结论，除非对应 spec 明确该实验族允许。

备选方案是简单禁止所有 labeled target auxiliary 字段。它最保守，但会阻断 V6/V8 作为补充诊断或 ablation 的用途，不利于后续定位 auxiliary branch 是否真的生效。

### 3. metric horizon 聚合收敛到同一 helper

selected horizon 解析和聚合应集中在训练指标模块的共享 helper 中，普通验证和 subset validation 都传入同一组 `metric_horizons`。subset payload 可以保留逐 horizon 诊断，但 top-level `top1`、`top3`、`top5`、`adba` 必须和主验证使用同一聚合规则。

备选方案是只修 subset top1 的局部逻辑。这样能修当前表现，但后续 Top-3、Top-5、ADBA 或 standalone evaluate 仍可能再次漂移。

### 4. MMW materialization 从训练 preflight 下沉到公开数据准备/split utility

MMW split 和 radar CSV materialization 属于数据准备职责，应由公开 preprocessor、manifest/split builder 或等价包内 utility 提供。HiST-Beam LOSO executor preflight 只检查 artifact 是否存在、是否可复用，或调用公开 utility；不得导入 `_ensure_*` 这类 dataset 私有 helper。

备选方案是在 executor 中继续维护小型私有补丁函数。短期容易，但会让训练入口继续承担写数据副作用，并且和 MMW dataset 内部实现耦合。

### 5. LOSO executor 拆分以保持外部入口兼容为优先

`hist_beam_loso_execution` 可先保留公开 facade 和 CLI 行为，把实现逐步迁移到窄模块，例如 `hist_beam_loso_preflight`、`hist_beam_loso_stages`、`hist_beam_loso_summary` 和 `hist_beam_loso_matrix`。拆分必须先有 characterization tests，确保 run metadata、summary JSON、quick conclusion 和 checkpoint reuse 语义不变。

备选方案是一次性重写 executor。该方案清理更彻底，但对正在使用的实验入口风险高，也会扩大本稳定化 change 的验证成本。

## Risks / Trade-offs

- sensitive-field policy 过严导致既有补充实验失败 -> 通过显式 opt-in 与 `main_conclusion_eligible=false` 保留诊断运行能力。
- 新增 eligibility 字段可能让旧分析脚本忽略不可比状态 -> 保持既有指标字段不变，同时在 summary 和 quick conclusion 中强制消费 eligibility。
- metric horizon 统一后 subset 指标数值会变化 -> 在任务中加入 focused test，明确这是从 first-valid-slot 漂移修正为 selected horizon 口径。
- executor 拆分可能引入循环导入或产物字段漂移 -> 先提取纯 helper 和 writer，再迁移 stage orchestration，并用现有 focused tests 约束输出。
- MMW materialization 公共化可能暴露现有本地产物缺失 -> preflight 错误信息必须指向可运行的公开准备命令或 utility，而不是静默写半成品。
