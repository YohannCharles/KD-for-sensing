## 1. Cache 与融合

- [x] 1.1 实现只加载 inner-train/inner-validation clean cache的 schema、identity与 forbidden-field审计
- [x] 1.2 实现 C0 prior等价的 modality evidence、full/missing logits、evidence oracle和 residual target
- [x] 1.3 实现全部 validation full重建与冻结 C0 fixed-subset missing重建 gate

## 2. Probe 训练

- [x] 2.1 实现共享 train-only input normalization、单层 Linear与预注册小型 MLP
- [x] 2.2 实现 evidence/residual loss、固定 batch order、AdamW、early stopping和 validation-recovery checkpoint
- [x] 2.3 实现 mean evidence与 train-only分批 nearest-neighbor基线

## 3. 指标与产物

- [x] 3.1 实现 predictability、最终 beam、oracle-gap recovery和参数/时间审计
- [x] 3.2 实现 sunny/rainy/foggy与8 beam sector分层、每模态判定和唯一方向总结
- [x] 3.3 生成 `outputs/missing_evidence_probe/` manifest、notes、状态、CSV、checkpoint、figure目录和运行脚本

## 4. 启动与验证

- [x] 4.1 实现 GPU0--3独立 launcher，设置物理GPU后内部使用 `cuda:0`并隔离失败
- [x] 4.2 增加15项 preflight定向测试并使用 `conda run -n kd_mm_beam pytest` 保存结果
- [x] 4.3 使用 `openspec validate`、聚焦 pytest和 compile检查验证实现

## 5. 本地可行性运行

- [x] 5.1 通过 cache/model重建 gate后运行四模态共16个轻量任务
- [x] 5.2 汇总 recovery、predictability、oracle gap、weather/sector和唯一建议后停止
