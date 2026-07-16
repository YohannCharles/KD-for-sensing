# 退役路线说明

本文件保留已删除代码的研究用途，避免将历史路线重新引入 current runtime。实现、配置、CLI、测试和兼容 stub 均不保留；需要精确版本时使用 Git history 或 `openspec/changes/archive/`。

| 路线 | 曾有用途 | 退役原因 |
| --- | --- | --- |
| 外部 teacher、full-to-partial KD、weak-pattern KD、latent probe、H6/H7 teacher 调参变体 | 早期 teacher-student 和辅助预测实验 | T2 只保留 embedded teacher CE 与 same-model superset consistency，其他分支没有 tracked recipe 消费。 |
| DeepSense6G 的 CSI/mmWave/input-beam/soft-label/cache 分支、Scene9/23、final C2、H5 | 旧数据集和缺失模态实验主线 | 当前仅保留重写后的 Scene31–34 四模态 future-beam T2 路径；不恢复这些历史输入、目标、缓存或别名。 |
| CSI、mmWave channel、physics-informed MMW | 通道重建、物理先验和 CSI hardening 研究 | 不属于四模态 T2/baseline 闭包，且增加无条件 import 与数据契约。 |
| Town10 skybridge zip 解包、channel index、beam-power 派生和专属 split builder | 早期 Town10 MMW 数据准备 | 当前 T2/baseline 仅使用 Town03 all-weather prepared sequence；保留 Town03 雷达图和 manifest split，不保留 Town10 专属入口。 |
| 数据 cache/descriptor/protocol-split 兼容、RSU-local GPS/yaw 路径 | 旧数据准备与 all-weather 路由辅助逻辑 | 当前 MMW 使用 tracked recipe、prepared sequence 和统一 GPS 表示，不保留 fallback 或 route-specific 特例。 |
| GPS-only v2、AMR、旧 fusion/PCPG/BPRR/RBMA | 单模态控制和旧 router/fusion 试验 | 当前基线集合固定为 S1、AMBER-Full、RMBP-MM，T2 router 只保留当前 supervised-router 路径。 |
| Image+GPS JEPA、预训练和 snapshot/多任务目标 | 预训练、定位、遮挡等支线 | 不服务当前 beam 主任务与固定矩阵协议。 |
| run index、paper export、viewer、一次性诊断 CLI | 本地管理和展示工具 | public workflow 收敛为 train/evaluate/preprocess，证据脚本保留为仓库内 MMW helpers。 |

历史文档不构成当前支持承诺。重新引入任一族必须先创建新的 OpenSpec change，并恢复完整实现、配置、验证和文档，而不是增加 alias 或迁移层。
