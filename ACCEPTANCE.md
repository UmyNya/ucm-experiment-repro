# 离线验证

此仓库的验证范围是配置解析、容量数值换算、进程生命周期、模拟客户端、事件解析与报告生成逻辑。单元测试不加载模型或占用 GPU/NPU，不能证明实机性能或容量校准结果。

发布副本的离线检查结果：57 项单元测试中 56 项通过，1 项因未配置外部 UCM dispatcher 源码而跳过。合成 pipeline 报告生成 16 张图、48 个图片文件、2 页请求图和 2 页层级图，75 个相对链接全部有效。报告原始文件与测试日志保留在本地，不随仓库分发。

发布文件检查排除了运行目录、历史审计、对话快照和本地配置；未发现个人主目录路径、邮箱、设备 UUID、内网地址、常见凭据格式或私钥。后续修改和运行产物需要重新检查，检查结果不替代未来内容审阅。

```bash
python3 -B -m unittest discover -s tests -v
python3 -B scripts/plan.py --profile swe_w24_n3379_performance
python3 -B tests/check_pipeline_report.py
```

基础配置和多数单元测试只需 Python 标准库。报告测试需要 Matplotlib；原生 trace 测试需要 C++ 编译器。读取实际 UCM dispatcher 的适配测试，在未配置外部 UCM 源码时会明确跳过。`tests/check_*clients.py` 等客户端集成脚本还需要外部客户端环境和使用者提供的输入，不包含在基础单元测试覆盖声明中。

容量测试使用 `tests/fixtures/capacity-layouts.json` 的最小数值字段，保留整数取整、预留节点和无效输入检查，不依赖私有实验报告。

服务构建、模型加载、实际 GPU/NPU 推理、设备 profiling、并行容量布局和 loopback 挂载均需要在对应容器中另行验证。执行命令及结果不会自动上传。
