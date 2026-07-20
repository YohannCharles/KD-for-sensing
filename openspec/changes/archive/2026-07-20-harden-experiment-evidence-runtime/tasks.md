## 1. 实验身份与筛选证据

- [x] 1.1 为 H4 设计筛选配置实现可重算的规范化摘要，并在计划复用时校验候选、画像、inner split 与配置身份。
- [x] 1.2 将 batch probe 结果绑定到候选、GPU、训练画像和配置摘要，拒绝不匹配或缺失的探针证据。
- [x] 1.3 强化 inner split 审计，覆盖稳定样本身份、分组身份与资源引用，并对缺失字段或跨 split 交集 fail closed。
- [x] 1.4 校验筛选候选相对于 matched control 的 resolved-config 差异只包含声明 allowlist。
- [x] 1.5 为 checkpoint/evaluate 画像验证重算规范化配置摘要和训练画像摘要。

## 2. 评估契约与指标一致性

- [x] 2.1 让 fixed-mask all-weather evaluator 复用 checkpoint 训练画像与归一化 artifact 校验，禁止静默重拟合。
- [x] 2.2 为部分评估结果记录覆盖范围，并让汇总器拒绝不完整或混杂实验身份的输入。
- [x] 2.3 统一 DBA/aDBA 的 top-k 与 circular/linear 距离语义，并输出可校验的 metric profile。
- [x] 2.4 为 MMW CMA 正样本使用带 domain/split 上下文的稳定样本身份。

## 3. 模型与数据运行时正确性

- [x] 3.1 修正 AMBER 的模态 token padding mask，使填充 GPS token 不参与注意力或 pooled feature。
- [x] 3.2 冻结或排除 T2 非活跃分类头参数，并在 reliability_mean 下跳过零权重 router/oracle 训练分支。
- [x] 3.3 缓存 DeepSense6G future-beam 标签读取，并修正确定性 CUDA 配置在训练与评估间的一致性。
- [x] 3.4 对 MMW 输入 schema、标签范围和相对资源路径增加 fail-closed 校验。

## 4. 入口与调度稳健性

- [x] 4.1 让训练、评估和预处理 CLI 拒绝未知参数及未知 dotted override。
- [x] 4.2 扩展 MMW launcher preflight 覆盖训练需要的标签、BS GPS 与雷达派生输入。
- [x] 4.3 为筛选 launcher 的 manifest/status 写入实现原子化和子进程启动失败清理。
- [x] 4.4 修正文档入口、维护路由和验证目标中的失效或重复工作流说明。

## 5. 验证

- [x] 5.1 添加覆盖 H4 配置/outer-test 隔离、配置摘要、probe 身份和 inner split 审计的回归测试。
- [x] 5.2 添加覆盖评估溯源、指标语义、AMBER token mask、非活跃头与数据/CLI fail-closed 行为的回归测试。
- [x] 5.3 使用 `conda run -n kd_mm_beam` 运行相关 pytest、`make verify-quick`、`make verify-cli-config`、`make verify-compile` 与 OpenSpec 严格校验。
