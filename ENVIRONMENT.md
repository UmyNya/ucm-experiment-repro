# 环境与迁移清单

在已有 Docker 容器中执行，不构建或启动容器。先将 `config/experiment.json` 的 `paths` 改为容器内实际路径；模型/tokenizer、三份仓库、输出目录和 L2 所在磁盘都必须在容器内可见。脚本不复制或修改权重。模型变化时同时检查 `config/vllm.json` 的 `hf_overrides`，历史的 14 层、关闭 MTP 设置不应自动沿用到任意新模型。

服务、AISBench、SWE 和报告可以使用不同 Python。环境准备时，`paths.server_python` 指向容器内具备服务依赖的基础 Python；执行 `prepare_environment.py --build` 成功后，将其改为构建目录 `service-environment.json` 中记录的 `server_python` 路径。AISBench 和 SWE 可在 `paths.aisbench_python`、`paths.swe_python` 中直接使用各自已有环境。完整保留 `venv/bin/python` 路径，不能把它解析成系统 Python 后再调用。具体命令见 [准备服务环境](USAGE.md#准备服务环境)。

## 按 requirements 准备依赖

依赖清单按运行环境分别维护：

| 清单 | 用途与安装位置 |
|---|---|
| 本仓库 [requirements.txt](requirements.txt) | 本包结果解析、报告和离线测试的 NumPy、Matplotlib 直接依赖，使用 Python 3.12 下验证过的版本。不是完整传递依赖锁文件。 |
| UCM 源码的 `requirements-build.txt`、`requirements.txt` | UCM 构建与运行依赖。开启 `build.install_dependencies` 时，环境准备脚本读取所选 UCM 提交中的这两份文件。 |
| AISBench / SWE-rebench 自身的 requirements 或平台对应安装说明 | 各自客户端 Python 环境的依赖；不能用本仓库 requirements 代替。 |
| 服务镜像与设备软件栈 | vLLM、PyTorch、vllm-ascend、torch-npu、CUDA/CANN、驱动及系统编译工具链，按目标平台和版本准备。 |

报告/离线测试的 Python 按本仓库清单准备。AISBench 结果解析在 `paths.aisbench_python` 中执行，该环境也需要 NumPy，通常由 AISBench 自身依赖提供；应以客户端的版本约束为准。

以下是使用者在目标环境执行的命令示例，普通服务/实验入口不会自动运行它们：

```bash
# 在报告/离线测试的 Python 环境中，按清单安装。
python3 -m pip install -r requirements.txt

# 无网络时，改用已准备好的对应平台 wheels。
python3 -m pip install --no-index --find-links /离线wheels目录 -r requirements.txt
```

UCM 的依赖由自己的 requirements 管理；开启安装时仍使用离线 wheelhouse：

```bash
python3 -B scripts/prepare_environment.py --build \
  --set build.install_dependencies=true \
  --set 'build.wheelhouse="/离线wheels目录"'
```

依赖安装默认关闭；缺少依赖时构建会报错，不会自动联网。其他 Python 版本或平台应先核对可用 wheel 与组件约束，再调整相应清单；不要把报告环境的版本固定项直接用于替换服务镜像的软件栈。

## 各组件环境

| 环境 | 来源及要求 |
|---|---|
| 服务 | 目标镜像已有 vLLM/torch；Ascend 还需对应的 vllm-ascend、torch-npu、CANN/驱动。`prepare_environment.py --build` 在新的私有 venv 构建并安装 UCM，继承镜像依赖，不替换镜像里的 vLLM/torch。 |
| AISBench | `paths.aisbench_repository` 的 `src/aisbench_ucm` 通过只读路径使用；实际 AISBench 客户端应可导入 `ais_bench.benchmark.cli.main`。离线验证使用 ais-bench-benchmark 3.1.20260630、Python 3.12。原仓库 `env/bench.lock.txt` 和 `docs/environment.md` 是历史环境参考。 |
| SWE | `paths.swe_repository` 下的 `swe_bench` 通过只读路径使用。直接依赖见该仓库 `requirements.txt`，完整历史版本见 `requirements.lock.txt`；离线验证使用 Python 3.12。 |
| 报告 | 运行 `generate_report.py` 的 Python 需要 Matplotlib；报告代码与表格/配置解析均为本包代码，图片不依赖浏览器、CDN 或远程 Markdown 模板。 |
| CUDA profiling | 配置 `profiling.nsys`，需要容器可访问相应 GPU/驱动和 Nsight 所需权限；CPU 生命周期测试使用 Nsight Systems 2025.5.1，设备 trace 尚未完成实机验证。 |
| Ascend profiling | 使用目标镜像自带的 `torch_npu.profiler`；原生 UCM 采集头不依赖 CUDA。CANN/torch-npu trace schema 尚未完成实机验证。 |

AIS 历史完整 lock 是 Linux x86_64/CUDA 环境记录，其中包括 HTTP 客户端不需要的本地推理包，且原说明已记录部分依赖元数据冲突。**不能把整份历史 lock 当作 Ascend/ARM 的通用安装清单。** 使用目标上现成的 AISBench 客户端环境；若需重新准备，按其目标平台官方包及离线 wheel 清单安装到新的私有 venv，再执行本包输入准备和 HTTP 验证。不要运行原仓库会改写其目录的 setup/restore-cache 命令。

SWE 若需要新的客户端环境，可在本包的新目录中创建，使用目标平台已准备的离线 wheels；以下路径均为示意，需要替换。创建目录必须是新目录：

```bash
python3 -m venv /本包目录/新SWE环境
/本包目录/新SWE环境/bin/python -m pip install --no-index \
  --find-links /离线wheels目录 -r /SWE仓库/requirements.lock.txt
```

UCM 构建默认 `--no-index`、`--no-deps`、`--no-build-isolation`，缺依赖就失败并保留日志；`build.wheelhouse` 和 `install_dependencies` 提供显式的离线依赖来源。`build.jobs` 控制编译并行数。CUDA 的原生库和编译缓存不能直接用于 Ascend，应在对应平台重新构建。每份私有环境保存 source commit、补丁与 wheel 哈希、Python/原生文件清单，服务启动时复核。

迁移脚本包至少保留 `config/`、`scripts/`、`native/`、`reference/patches/`、`requirements.txt` 与使用说明；运行容量示例和离线测试时还需保留 `reference/layouts/` 与 `tests/`。已有结果需要连同引用它们的报告目录层级一起迁移，保留 `REPORT.md` 相对路径所指向的图表和原始数据。

参考实现接口：[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[Ascend PyTorch Profiler](https://www.hiascend.com/document/detail/en/mindstudio/2600/TITools/ascend_pytorch_profiler/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md)、[Ascend trace flow 说明](https://www.hiascend.com/document/detail/en/mindstudio/700/TITools/Profiling/atlasprofiling_16_0059.html)。文档不替代目标版本的运行验证。
