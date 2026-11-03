# vLLM + UCM 服务与实验脚本

在已有容器内独立启动 vLLM + UCM 服务，使用 AISBench 或 SWE-rebench 测试，或执行包含编译校准、协议预热、缓存观测和重复轮的单配置实验。支持配置 CUDA/Ascend 平台、设备与并行数、HBM/L1/L2 容量，以及 pipeline 采集补丁和本地 Markdown 报告。

这是经过离线验证的脚本包。真实 GPU/NPU 推理、容量校准和设备 trace 的兼容性需要在实际模型及容器环境中验证。

## 从哪里开始

1. 阅读 [环境要求](ENVIRONMENT.md)，准备容器中可用的模型、源码仓库和 Python 依赖。本包结果解析、报告与离线测试的直接依赖见 [requirements.txt](requirements.txt)，服务和客户端依赖按各自组件准备。
2. 修改 [主配置](config/experiment.json) 中的示例路径、平台、设备和容量。`/workspace/...`、`/models/your-model` 是占位路径。
3. 按 [准备服务环境](USAGE.md#准备服务环境) 构建 UCM，并配置生成的服务 Python 路径。
4. 根据目的选择 [独立服务与客户端](USAGE.md#独立启动顺序) 或 [单配置完整实验](USAGE.md#单配置完整实验与报告)。完整参数解释见 [主配置参数参考](USAGE.md#主配置参数参考)。

```bash
# 仅检查最终配置和启动计划，不启动服务。
python3 -B scripts/plan.py --profile swe_w24_n3379_performance

# 运行不需要模型服务的单元测试。
python3 -B -m unittest discover -s tests -v
```

## 文件位置

| 路径 | 内容 |
|---|---|
| `config/experiment.json` | 公共路径、平台、容量、负载和实验流程 |
| `config/vllm.json` | vLLM 原生服务参数；新增参数可直接写入，JSON 参数直接使用对象 |
| `config/vllm-env.json` | 公共、CUDA、Ascend 服务环境变量 |
| `config/ucm.json` | UCM connector 与缓存策略 |
| `config/aisbench.json`、`config/swe-rebench.json` | 客户端输入和负载 |
| `scripts/` | 服务、校准、客户端、实验与报告入口 |
| `native/`、`reference/patches/` | 原生采集支持与 UCM 观测补丁 |
| `reference/layouts/` | 历史 CUDA 数值布局示例，换模型或后端需要重新测量 |
| `tests/` | 离线测试、匿名数值样例及可选客户端集成检查 |
| `requirements.txt` | 本包结果解析、报告与离线测试使用的 NumPy、Matplotlib 版本 |

vLLM 参数统一采用原生参数名，例如 `tensor_parallel_size`、`quantization`、`attention_config`、`additional_config`。写法、环境变量示例和公共配置的分工见 [vLLM 参数参考](USAGE.md#vllmjson-参数参考)。每次服务启动前生成 `vllm-effective.yaml`，通过 `vllm serve --config` 加载；完整命令和环境覆盖保存在该次运行的 `plan.json`。

`vllm.json` 中的 `block_size` 可以省略或设为 `null`，由引擎决定；脚本不会补默认值。启动后记录实际 worker block size，AISBench 使用实测值准备输入。容量换算和独立客户端的要求见 [省略 block_size](USAGE.md#省略-block_size)。

UCM 配置采用原生顶层字段与 `ucm_connectors[].ucm_connector_config` 结构。源码引用和启用开关在主配置的 `ucm_runtime` 中，容量和私有缓存路径由公共配置生成；写法与兼容规则见 [UCM 参数参考](USAGE.md#ucmjson-参数参考)。

构建 UCM 还需要 fmt、spdlog、pybind11、zlib 等原生依赖源码，它们不由 Python wheelhouse 提供。默认离线；联网准备方式通过 `build.cmake_args` 显式开启，见 [原生依赖源码](ENVIRONMENT.md#原生依赖源码)。

## 环境探测与容量布局

以下命令在已有容器中、本仓库目录下执行。`--output` 指定新建结果，相对路径从命令所在目录计算；重复执行时使用新名称。

```bash
python3 -B scripts/probe_environment.py \
  --output runs/target-container.json \
  --model /models/your-model --disk-path /cache

# 此命令仅用示例布局换算，不测量设备。
python3 -B scripts/capacity.py \
  --layout reference/layouts/historical-cuda-d21.json \
  --hbm-tokens 320000 --l1-tokens 1000000
```

### 如何测量并生成自己的 layout JSON

完成服务环境准备后，用 `scripts/calibrate_capacity.py` 启动模型读取真实 worker 布局。HBM 使用 `auto` 时由引擎自动规划，否则使用配置中的 HBM 物理预算；L1 使用物理预算。测量后停止服务，操作步骤与 tokens 换算见 [容量说明](USAGE.md#hbml1l2-容量)。脚本不会通过反复尝试搜索最大可用容量。

设置 `capacity.hbm.mode="auto"` 可按设备与模型运行时占用自动规划 KV cache，预算比例由 `config/vllm.json` 中的 `gpu_memory_utilization` 控制。实际分配量在服务启动后记录，配置示例见 [自动 HBM 容量](USAGE.md#自动-hbm-容量)。

## 数据与验证范围

仓库只包含代码、配置模板、文档、补丁和最小测试数据。运行目录、对话归档、真实请求、设备标识和原始实验报告不随仓库分发。未来运行产生的配置快照、日志和请求记录保存在本地；分享结果前需自行检查内容。

[验证说明](ACCEPTANCE.md) 记录离线检查方法与范围；[实验协议参考](FINDINGS.md) 说明默认参数的技术背景；[第三方说明](THIRD_PARTY_NOTICES.md) 列出补丁涉及的第三方许可。
