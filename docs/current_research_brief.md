# Current Research Brief

当前研究主线是 MMW 与 DeepSense6G 共享四模态 T2 上的 CMSBL：以 BCACL U2 的 modality-private/shared Beam 监督缓解模态失衡，再以 train-only capacity gap 和困难 availability-mask loss 处理模态缺失。CMSBL 只改变训练 objective；推理继续使用 T2 supervised router 与融合 Beam prototype。

MMW 仍保留 T2、S1、AMBER-Full、RMBP-MM 的固定比较协议；DeepSense6G 仅保留 Scene31--34 T2 路径。BCACL relation teacher/quality/two-stage、PCER、PGCD、动态 Router、PR-SQDF、missing residual、feature/prototype fusion、availability fallback 和 BT-SCL 均已退役。

所有正式 claim 必须来自 tracked recipe、真实 run artifact 和对应数据集 summary。CMSBL 当前保持 inner/development、claim-ineligible，未授权自动 outer test 或 multi-seed。
