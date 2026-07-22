# 数据集实验协议

MMW 协议固定为 15-domain、image/radar/gps/lidar 四模态、40 epoch 和 `last.pth`。比较方法仅为 T2、S1、AMBER-Full、RMBP-MM；同一比较必须共享 split、样本身份和缺失 mask identity。

DeepSense6G 协议仅允许 Scene31–34、同一四模态输入和 future-beam 64 类硬标签。当前只提供 T2 recipe；其 scene、CSV split 与 mask provenance 必须独立记录，不能和 MMW matrix 混合。

T2 的可解释消融包括 BPA/CMA、BCACL U2 和 CMSBL M1--M3。CMSBL 当前是 development evidence，不能单独升级 claim。真实训练、评估和汇总产生的产物只留在 `outputs/`，当前文档只登记其 provenance 和结论状态。
