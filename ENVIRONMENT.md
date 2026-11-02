# 环境与迁移清单

在已有 Docker 容器中执行，不构建或启动容器。先将 `config/experiment.json` 的 `paths` 改为容器内实际路径；模型/tokenizer、三份仓库、输出目录和 L2 所在磁盘都必须在容器内可见。脚本不复制或修改权重。模型变化时同时检查 `config/vllm.json` 的 `hf_overrides`，历史的 14 层、关闭 MTP 设置不应自动沿用到任意新模型。

服务、AISBench、SWE 和报告可以使用不同 Python。环境准备时，`paths.server_python` 指向容器内具备服务依赖的基础 Python；执行 `prepare_environment.py --build` 成功后，将其改为构建目录 `service-environment.json` 中记录的 `server_python` 路径。AISBench 和 SWE 可在 `paths.aisbench_python`、`paths.swe_python` 中直接使用各自已有环境。完整保留 `venv/bin/python` 路径，不能把它解析成系统 Python 后再调用。具体命令见 [准备服务环境](USAGE.md#准备服务环境)。

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

迁移脚本包至少保留 `config/`、`scripts/`、`native/`、`reference/patches/` 与使用说明；运行容量示例和离线测试时还需保留 `reference/layouts/` 与 `tests/`。已有结果需要连同引用它们的报告目录层级一起迁移，保留 `REPORT.md` 相对路径所指向的图表和原始数据。

参考实现接口：[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[Ascend PyTorch Profiler](https://www.hiascend.com/document/detail/en/mindstudio/2600/TITools/ascend_pytorch_profiler/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md)、[Ascend trace flow 说明](https://www.hiascend.com/document/detail/en/mindstudio/700/TITools/Profiling/atlasprofiling_16_0059.html)。文档不替代目标版本的运行验证。
