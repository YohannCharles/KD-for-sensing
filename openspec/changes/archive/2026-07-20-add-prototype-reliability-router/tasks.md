## 1. 共享模型组件

- [x] 1.1 公开并复用 beam topology position helper，避免 Router 复制 topology 定义
- [x] 1.2 实现共享的 PATR 时序原型证据、H2R 帧健康门控和 CoRe leave-one-out 共识组件
- [x] 1.3 将 `router_variant` 接入 UMaskBeamJEPA，并保持 Current 默认 state-dict/forward 与共享输出契约
- [x] 1.4 实现候选 detached state 的统一重新路由接口及 Router-only冻结/eval策略

## 2. 配对数据与效用监督

- [x] 2.1 实现240-entry固定 Joint Drop/Corrupt训练panel、checksum和精确平衡审计
- [x] 2.2 扩展传感器 corruption以仅修改选定 `[B,T]` cell，并保持旧推理API行为不变
- [x] 2.3 将GPS corruption scaler metadata与future beam-power target开关解耦
- [x] 2.4 实现 label-topology/normalized beam-power连续utility、macro calibration、direct gain和无元数据配对单调loss
- [x] 2.5 将drop-control/joint view和统一候选重新路由接入现有UMaskBeamJEPA训练扩展
- [x] 2.6 修复AMP下线性beam-power归一化下溢，并以真实功率量级回归测试锁定float32策略

## 3. 初始化与配置契约

- [x] 3.1 实现与严格resume互斥的模型初始化checkpoint、SHA/key allowlist和load provenance
- [x] 3.2 扩展候选Router与监督配置解析，拒绝未知、不兼容或泄漏配置
- [x] 3.3 实现只优化候选Router参数的严格optimizer与冻结模块状态校验

## 4. 八卡筛选运行时

- [x] 4.1 实现8个seed1 resolved config、固定GPU映射、source/panel身份预检和manifest生成
- [x] 4.2 扩展inner-only机制评估以比较Uniform、train-fit prior、Current、Dynamic和Oracle
- [x] 4.3 维护运行状态和预注册Gate汇总，不将seed1候选标记为outer claim

## 5. 验证与执行

- [x] 5.1 使用 `conda run -n kd_mm_beam pytest` 完成Current兼容、四variant机制、梯度和输出契约测试
- [x] 5.2 使用 `conda run -n kd_mm_beam pytest` 完成Joint panel、selective corruption、utility、初始化和配置fail-closed测试
- [x] 5.3 运行 `openspec validate add-prototype-reliability-router --strict` 及项目快速验证
- [x] 5.4 使用 `conda run -n kd_mm_beam` 完成batch64单步GPU smoke并核对显存
- [x] 5.5 在GPU0--7启动八个seed1候选，核验PID、日志、manifest和预计完成时间
- [x] 5.6 终止数值策略错误的四个Power任务，记录源码SHA并在GPU1/3/5/7启动修复版子集
