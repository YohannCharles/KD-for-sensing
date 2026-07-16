# Current Research Brief

当前研究问题是在 MMW 15-domain、四模态、固定 40 epoch 和 fixed-last checkpoint 协议下，评估 T2 对时序缺失的鲁棒性。

方法集合固定为 T2、S1、AMBER-Full、RMBP-MM。T2 使用 supervised router、BPA、embedded teacher CE 和 same-model superset consistency；S1 仅关闭 consistency。BPA/CMA 消融用于解释 prototype 目标的贡献，不扩展为新主线。

所有 claim 必须来自 tracked recipe、真实 run artifact 和对应的 MMW summary。退役路线的结果不参与当前 claim。
