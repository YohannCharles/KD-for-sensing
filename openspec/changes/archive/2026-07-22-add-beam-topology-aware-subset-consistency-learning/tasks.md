## 1. 身份与固定协议

- [x] 1.1 定位并记录上游 encoder/F1 validation-best checkpoint、resolved config、cache、prototype、temperature、topology 与 SHA
- [x] 1.2 生成并校验 14-pattern train/validation manifest、全部 nested pair 和 topology sector manifest
- [x] 1.3 实现在线 encoder-tail 捕获与 cache/F1 parity preflight，禁止字段和 split overlap 必须 fail closed

## 2. 模型与损失

- [x] 2.1 实现 encoder-tail low-rank residual、确定性 Regime A/B/C、missing fusion residual 和 Full 物理 bypass
- [x] 2.2 实现只读当前可用模态的 AER auxiliary prototype head 与逐样本模态平均 loss
- [x] 2.3 实现 topology risk、stop-gradient NTM、16/8/4-sector SCFC 与对应诊断
- [x] 2.4 实现 V0--V5 结构公平、基础 F1/prototype 冻结、参数分组与固定 train-batch loss 校准

## 3. Stage A 与训练评测

- [x] 3.1 实现六个在线 raw-input subset specialist、encoder/fusion 参数分组和 subset validation-loss checkpoint 选择
- [x] 3.2 实现 V0--V5 的固定 schedule 训练、相邻 nested pair、group-balanced validation selection 与 early stopping
- [x] 3.3 实现 15 路 pattern、全部 nested pair、weather/sector/error-distance、表示/probe 与效率评测
- [x] 3.4 实现 specialist headroom、主表、七项 success gates、25 个必答问题与唯一推荐方向汇总

## 4. 启动脚本与测试

- [x] 4.1 新增 Stage A、Stage B GPU0--5 失败隔离启动脚本，保存 PID、日志、resolved config、状态与退出码
- [x] 4.2 新增 Full bypass、14-pattern、冻结、AER、NTM、SCFC、manifest 和禁止输入 focused tests
- [x] 4.3 使用 `conda run -n kd_mm_beam pytest tests/test_bt_subset_consistency.py -q` 运行 focused preflight

## 5. 验证与实验

- [x] 5.1 使用 `openspec validate add-beam-topology-aware-subset-consistency-learning --strict`、`make verify-quick` 和 `conda run -n kd_mm_beam python scripts/verify_compile.py` 校验
- [x] 5.2 在 GPU0--5 完成 Stage A 并保存六个 specialist 结果，不使用 outer test
- [x] 5.3 已记录终止 Stage B V0--V5：未启动，且该方向已被 CMSBL 主线取代

## 6. Reproducibility Repair

- [x] 6.1 保留 legacy 产物并审计上一轮 V0--V5 的 launch/config/log/PID/status/checkpoint/metrics，输出受限状态、根因和是否需要重训
- [x] 6.2 从 Availability Fallback U0 直接引用锁定 canonical F1 与 split/sample/mask/metric hash，双次精确复现 Full+14 patterns 并完成历史对账
- [x] 6.3 使用独立 canonical F1 对象和统一 development IDs 重评六个既有 specialist，输出有效性、真实 headroom 与训练诊断
- [x] 6.4 已记录取消 repair V1--V5 实现：current source 不再保留 AER/NTM/C2F/BT-SCL runtime
- [x] 6.5 已记录取消对应 smoke：未授权也未启动 GPU0--5 训练
- [x] 6.6 已记录终止 repair V0--V5：不生成新的实验产物或 claim
