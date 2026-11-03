# 离线验证

此仓库的验证范围是配置解析、容量数值换算、构建参数与原生依赖获取流程、进程生命周期、模拟客户端、事件解析与报告生成逻辑。测试不加载模型或占用 GPU/NPU，不能证明实机性能或容量校准结果。

离线测试覆盖合成 pipeline 报告的图表、分页及相对链接。运行结果以执行测试时的输出为准；测试日志、报告及环境记录由使用者保存在本地，不随仓库分发。

配置测试覆盖原生 vLLM 扩展字段、嵌套对象、数组、显式关闭开关、旧格式兼容、独立环境变量文件的快照、公共参数冲突检查及新增引擎参数的校准身份绑定。另外使用 vLLM v0.28.0/v0.30.0 官方配置加载器配合模拟参数定义做了读取验证，未导入或启动真实引擎；该检查不代表模型、所有原生选项或 UCM 与目标引擎版本已完成集成验证。

原生 UCM 配置测试覆盖新旧格式等价、顶层及 connector 字段、数组索引覆盖、容量与路径冲突拒绝、原始配置快照和实际配置落盘。实际 `ucm/utils.py` 的 `Config` 读取器与 `ucm/metrics_dispatcher.py` 累计统计适配均在内存中测试，未加载模型、编译原生代码或修改参考源码。

发布文件检查排除了运行目录、历史审计、对话快照和本地配置；未发现个人主目录路径、邮箱、设备 UUID、内网地址、常见凭据格式或私钥。后续修改和运行产物需要重新检查，检查结果不替代未来内容审阅。

```bash
python3 -B -m unittest discover -s tests -v
python3 -B scripts/plan.py --profile swe_w24_n3379_performance
python3 -B tests/check_pipeline_report.py
```

基础配置和多数单元测试只需 Python 标准库。报告测试需要 [requirements.txt](requirements.txt) 中的 NumPy、Matplotlib；原生 trace 测试需要 C++ 编译器。原生依赖构建测试需要 CMake、Git、C++ 编译器和 make，wheel 打包测试还需要已有 pip、setuptools、wheel；缺少这些工具时相应测试明确跳过，不自动安装。实际 UCM 配置读取器测试还需要测试解释器中已有 PyYAML。没有可用源码时，三项源码测试会明确跳过；缺少 PyYAML 时读取器测试会明确跳过；显式指定了错误的源码路径则测试失败。`tests/check_*clients.py` 等客户端集成脚本还需要外部客户端环境和使用者提供的输入，不包含在基础单元测试覆盖声明中。

容量测试使用 `tests/fixtures/capacity-layouts.json` 的最小数值字段，保留整数取整、预留节点和无效输入检查，不依赖私有实验报告。

自动 HBM 的离线检查覆盖 CUDA/Ascend 各 profile 的启动参数、固定容量模式回归、校准模式选择、L1 校准身份检查、模拟服务实测容量记录，以及报告按实际容量分组和排除缺失观测的结果。测试使用合成 worker 数据，不代表目标设备的可用容量。

可选 `block_size` 的检查覆盖省略/null 不生成引擎参数、显式值校验、校准文件绑定实际布局、运行时容量对齐、AISBench 延后准备输入、重启后布局变化与准备失败时的服务清理，以及独立客户端读取观测和报告按实测值分组。CPU HTTP 模拟服务验证了实际启动配置保持省略、运行计划记录实测值、校准依据冲突时停止并释放资源；这些结果不证明目标引擎最终会选择哪一个 block size。

完整 UCM 服务构建、模型加载、实际 GPU/NPU 推理、设备 profiling、并行容量布局和 loopback 挂载均需要在对应容器中另行验证。执行命令及结果不会自动上传。

构建环境的 PATH 回归测试使用临时可执行文件和真实子进程，覆盖私有 venv 中的 CMake 查找、优先于继承的同名工具、保留原有工具路径及不修改父进程环境。测试覆盖 CUDA/Ascend 环境设置，不安装依赖或构建 UCM。

wheelhouse 预检测试覆盖未配置、目录缺失、路径为文件、空目录和无读取权限的情况，以及仅生成计划时允许先导出依赖清单。入口测试确认无效目录会在源码导出及子进程启动前失败，并保留失败状态；包的版本与平台兼容性仍由 pip 解析。

原生依赖功能测试保留默认离线行为，并验证显式 CMake 参数能够覆盖默认值。测试覆盖缺少 vendor 源码时失败、已有源码时离线编译，以及从临时本地 Git 仓库获取依赖后完成 CMake 配置、C++ 编译和程序运行。实际 UCM 的四份 `dep-*.cmake` 与 `helper.cmake` 也参与测试，仅将源码地址替换为本地测试仓库，验证获取分支及后续目标可用。测试进程禁止 Git 网络协议，不访问外部下载站点。

wheel 测试实际执行 `pip wheel --no-index --no-deps --no-build-isolation`，由测试包的构建后端调用 CMake 获取并编译依赖，运行程序，再检查 wheel 内容。它验证打包调用链，不构建完整 UCM，也不验证真实四项依赖的源码兼容性、目标网络连通性或 Ascend/CANN 链接；这些仍须在目标容器中验证。

## 使用参考 UCM 源码做离线检查

```bash
UCM_SOURCE_ROOT=/容器内/UCM仓库 python3 -B -m unittest discover -s tests -v
```

将占位路径替换为源码根目录；应包含 `ucm/utils.py`、`ucm/metrics_dispatcher.py` 和 `ucm/shared/vendor/dep-*.cmake`、`helper.cmake`。不设置该变量时，测试使用 `config/experiment.json` 的 `paths.ucm_repository`。显式指定的路径错误会导致测试失败，不会静默跳过。原生配置读取器测试使用已有 PyYAML；该依赖由目标 UCM 环境提供，测试不会自动安装。测试只读取参考源码；Python 测试在内存中隔离 logger/原生指标依赖，CMake 测试在临时目录编译测试依赖，不构建完整 UCM、不启动模型、不修改原仓库。
