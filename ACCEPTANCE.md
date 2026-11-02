# 离线验证

此仓库的验证范围是配置解析、容量数值换算、进程生命周期、模拟客户端、事件解析与报告生成逻辑。单元测试不加载模型或占用 GPU/NPU，不能证明实机性能或容量校准结果。

当前代码的离线检查结果：指定参考 UCM 源码后，80 项单元测试全部通过，无跳过。合成 pipeline 报告生成 16 张图、48 个图片文件、2 页请求图和 2 页层级图，75 个相对链接全部有效。报告原始文件与测试日志保留在本地，不随仓库分发。

配置测试覆盖原生 vLLM 扩展字段、嵌套对象、数组、显式关闭开关、旧格式兼容、独立环境变量文件的快照、公共参数冲突检查及新增引擎参数的校准身份绑定。另外使用 vLLM v0.28.0/v0.30.0 官方配置加载器配合模拟参数定义做了读取验证，未导入或启动真实引擎；该检查不代表模型、所有原生选项或 UCM 与目标引擎版本已完成集成验证。

原生 UCM 配置测试覆盖新旧格式等价、顶层及 connector 字段、数组索引覆盖、容量与路径冲突拒绝、原始配置快照和实际配置落盘。实际 `ucm/utils.py` 的 `Config` 读取器与 `ucm/metrics_dispatcher.py` 累计统计适配均在内存中测试，未加载模型、编译原生代码或修改参考源码。

发布文件检查排除了运行目录、历史审计、对话快照和本地配置；未发现个人主目录路径、邮箱、设备 UUID、内网地址、常见凭据格式或私钥。后续修改和运行产物需要重新检查，检查结果不替代未来内容审阅。

```bash
python3 -B -m unittest discover -s tests -v
python3 -B scripts/plan.py --profile swe_w24_n3379_performance
python3 -B tests/check_pipeline_report.py
```

基础配置和多数单元测试只需 Python 标准库。报告测试需要 [requirements.txt](requirements.txt) 中的 NumPy、Matplotlib；原生 trace 测试需要 C++ 编译器。实际 UCM 配置读取器测试还需要测试解释器中已有 PyYAML。没有可用源码时，两项源码测试会明确跳过；缺少 PyYAML 时读取器测试会明确跳过；显式指定了错误的源码路径则测试失败。`tests/check_*clients.py` 等客户端集成脚本还需要外部客户端环境和使用者提供的输入，不包含在基础单元测试覆盖声明中。

容量测试使用 `tests/fixtures/capacity-layouts.json` 的最小数值字段，保留整数取整、预留节点和无效输入检查，不依赖私有实验报告。

自动 HBM 的离线检查覆盖 CUDA/Ascend 各 profile 的启动参数、固定容量模式回归、校准模式选择、L1 校准身份检查、模拟服务实测容量记录，以及报告按实际容量分组和排除缺失观测的结果。测试使用合成 worker 数据，不代表目标设备的可用容量。

服务构建、模型加载、实际 GPU/NPU 推理、设备 profiling、并行容量布局和 loopback 挂载均需要在对应容器中另行验证。执行命令及结果不会自动上传。

## 使用参考 UCM 源码做离线检查

```bash
UCM_SOURCE_ROOT=/容器内/UCM仓库 python3 -B -m unittest discover -s tests -v
```

将占位路径替换为源码根目录；应包含 `ucm/utils.py` 和 `ucm/metrics_dispatcher.py`。不设置该变量时，测试使用 `config/experiment.json` 的 `paths.ucm_repository`。显式指定的路径错误会导致测试失败，不会静默跳过。原生配置读取器测试使用已有 PyYAML；该依赖由目标 UCM 环境提供，测试不会自动安装。测试只读取源码并在内存中隔离 logger/原生指标依赖，不构建 UCM、不启动模型、不修改原仓库。
