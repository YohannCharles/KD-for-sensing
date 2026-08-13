# DeepSense6G 跨数据集三模型面板实施计划

**目标：** 在不借用 MMW protocol、历史 beam index 或测试集调参的前提下，将 Prototype-only 四模态模型、AMBER-Full-local 和 RMBP-MM-local 以同一预算迁移到 DeepSense6G Scene31–34，并自动完成 15-mask 与 TBCP-3 评测。

**固定实验契约：** 读取 `deepsense6g_twc_secondary_v1` 已审计的过滤后 train/test CSV；四模态、5 帧、64 类、40 epoch、batch 64、seed 1/2/3、last checkpoint。DeepSense6G 没有兼容的 5 帧 validation，因此训练不访问 test、不做 validation 选择；训练完成后只进行一次官方 test 评测。Prototype-only 使用 `cyclic_index_v1` 标签空间邻接，不宣称 MMW ULA 物理拓扑。

## Task 1：固化 OpenSpec 与配置边界

- 更新 active change 的 proposal/design/tasks 和 delta spec，记录 DeepSense6G secondary panel、固定 split、过滤清单、无 validation 与 one-shot test 语义。
- 将 topology predictor 的数据集校验窄化为 MMW 或 DeepSense6G：MMW 保留现有 audit/test-seal；DeepSense6G 只允许 `cyclic_index_v1`、空物理 audit、last checkpoint 和四场景 pooled 数据。
- 增加 focused config tests，确保不能把 DeepSense 配置伪装成 ULA audit，也不能放宽 MMW 契约。

## Task 2：生成并预检九个独立训练配置

- 从一个 tracked DeepSense panel recipe 生成 `prototype_only/amber_full/rmbp_mm × seed1/2/3` 的 resolved configs。
- 每个任务使用独立 run/output/log；绑定 protocol manifest、过滤后 CSV SHA/count、四场景 train/test identity。
- 运行 config load、数据首 batch、模型 forward/loss smoke；确认 test loader 未在训练 smoke 中构建。

## Task 3：启动八卡队列

- GPU0–7 各启动一个单卡进程；第九个任务等待首张空闲卡后自动启动。
- 任务失败只记录一次并停止，不无限重试；训练输出写入新的 ignored root，不覆盖历史失败产物。
- 启动后检查数分钟：GPU 利用率、首 batch、metrics/run_status/log 与输出目录。

## Task 4：自动评测

- 每个 40-epoch last checkpoint 生成官方 test 的 15-mask Direct/Top-3/Top-5 evidence。
- topology likelihood 只从过滤后的 pooled train beam-power 拟合；test 只用于最终 TBCP-3 requested-only replay。
- 三种模型共享相同 test identity、K=3、full covariance、noiseless probing；输出按缺失模态数和 15 mask 汇总。

## Task 5：验证与交付

- 运行 OpenSpec strict、focused config/model/evaluator tests、compile；长时任务启动后核验进程与日志。
- 明早只汇报完整的 seed/方法；失败或未完成项显式列出，不静默删 seed，不把 local baseline 称作论文官方复现。
