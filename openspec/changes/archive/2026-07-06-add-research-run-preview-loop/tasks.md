## 1. Happy path 入口

- [x] 1.1 选择 research run preview 的入口形态：package CLI、Make target 或已登记 thin script。
- [x] 1.2 编排已有无副作用命令：OpenSpec validate、architecture quick check、surface doctor、run index/research dashboard、paper table/checklist consistency。
- [x] 1.3 默认模式确保不启动真实训练、不读取真实 dataset、不加载 checkpoint、不写训练产物。

## 2. 静态预览 QA

- [x] 2.1 定义 HTML/CSV/figure/table/checklist/conclusion 的必需字段和 caveat 检查。
- [x] 2.2 实现 synthetic fixture focused tests，覆盖空输入、candidate-only、pending、escaping、远程依赖、缺列和空数据。
- [x] 2.3 如已有 HTML evidence dashboard 输出，复用其 summary/renderer，不重复扫描 outputs。

## 3. 实验预算 manifest

- [x] 3.1 定义 budget manifest schema 或 dry-run summary 字段。
- [x] 3.2 在长跑或多 seed workflow 前输出 GPU/时间/输出 root/数据读取/checkpoint/cache/停止条件。
- [x] 3.3 文档说明真实 manifest 实例写 ignored outputs 或用户显式路径，不提交源码。

## 4. 环境和 run recipe

- [x] 4.1 记录 smoke/dev 与 GPU/full training 环境边界。
- [x] 4.2 给主要 package CLI 增加安装诊断或 `python -m kd_sensing.cli.<owner>` fallback 提示。
- [x] 4.3 确认 recipe 不包含本地路径、凭证、平台启动文件修改或 checkpoint。

## 5. 文档和验证

- [x] 5.1 更新 README、docs/experiment_matrix、agent navigation 或 inventory 的短入口和职责说明。
- [x] 5.2 运行 `openspec validate add-research-run-preview-loop --strict`。
- [x] 5.3 运行 `openspec validate --all --strict`。
- [x] 5.4 运行相关 focused tests，例如 architecture boundaries、dashboard/paper export/preview QA tests。
