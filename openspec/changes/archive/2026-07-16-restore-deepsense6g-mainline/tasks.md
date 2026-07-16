## 1. DeepSense6G 数据边界

- [x] 1.1 新建严格的 Scene31–34 四模态 DeepSense6G dataset，读取标准 CSV、资源文件与 future beam 硬标签。
- [x] 1.2 复用当前 image/radar/GPS/LiDAR transform，并接入训练 GPS scaler、稳定 metadata 和 split 选择。
- [x] 1.3 将 dataset 注册到共享 registry 和 data factory，保持 MMW domain 构建路径不变。

## 2. 配置与运行时契约

- [x] 2.1 扩展配置校验以显式接受 `mmw` 与 `deepsense6g`，并校验 DeepSense6G 场景和四模态/64 类约束。
- [x] 2.2 添加 DeepSense6G T2 canonical base/recipe，不新增模型、CLI 或历史兼容层。
- [x] 2.3 更新架构边界和配置测试，使双数据集主线为受检约束。

## 3. 证据与文档

- [x] 3.1 以临时合成资源测试 DeepSense6G 样本、future beam 标签、split 与 GPS scaler 行为。
- [x] 3.2 更新 README、导航、当前研究说明和 retired-route 文档，说明 DeepSense6G 的受限主线范围。
- [x] 3.3 使用 `conda run -n kd_mm_beam` 运行相关 pytest、`make verify-quick`、`make verify-cli-config`、`make verify-compile`、`make verify-full` 与 `openspec validate restore-deepsense6g-mainline --strict`。
