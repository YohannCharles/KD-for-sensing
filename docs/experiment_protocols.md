# MMW 实验协议

当前协议固定为 MMW 15-domain、image/radar/gps/lidar 四模态、40 epoch 和 `last.pth`。比较方法仅为 T2、S1、AMBER-Full、RMBP-MM；同一比较必须共享 split、样本身份和缺失 mask identity。

T2 的可解释消融仅包括 BPA/CMA、prototype head 和已登记的 head/fusion 控制。hyperparameter screening 是 development evidence，不能单独升级 claim。真实训练、评估和汇总产生的产物只留在 `outputs/`，当前文档只登记其 provenance 和结论状态。
