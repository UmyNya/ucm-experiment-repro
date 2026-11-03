# 使用说明与配置索引

本包在已有容器内执行，不配置或启动 Docker。默认配置沿用历史实验作为示例，首次使用时先按服务器情况修改路径、平台、设备与容量。验证范围为离线测试；服务构建、推理与设备采集需要在实际运行环境中验证。

## 配置文件位置

| 文件 | 修改内容 | 逐字段说明 |
|---|---|---|
| [config/experiment.json](config/experiment.json) | 容器路径、平台、设备、HBM/L1/L2 容量、公共负载、实验流程、采集与构建 | [主配置参数参考](#主配置参数参考) |
| [config/vllm.json](config/vllm.json) | vLLM 原生服务参数、模型覆盖、TP/PP/DP、调度与 KV cache | [vLLM 参数参考](#vllmjson-参数参考) |
| [config/vllm-env.json](config/vllm-env.json) | 公共、CUDA、Ascend 服务环境变量 | [服务环境变量](#vllm-envjson-参数参考) |
| [config/ucm.json](config/ucm.json) | 原生 connector、Early/HRRN/READY、整块缓存、wait/spill、POSIX I/O | [UCM 参数参考](#ucmjson-参数参考) |
| [config/aisbench.json](config/aisbench.json) | 输入来源、N、长度、前缀比例、顺序、预热数据、缓存 salt | [AISBench 参数参考](#aisbenchjson-参数参考) |
| [config/swe-rebench.json](config/swe-rebench.json) | W、正式请求数、seed、workload 来源、tokenizer 重生成、轨迹数 | [SWE 参数参考](#swe-rebenchjson-参数参考) |
| [config/history-sources.example.json](config/history-sources.example.json) | 可选历史审计的目录映射，不参与新实验配置 | [历史材料映射](#history-sourcesexamplejson-参数参考) |

主配置 `config_files` 引用组件参数及服务环境变量文件，可以改为其他路径；同一 section 不允许同时内联和外部引用。相对数据路径按所属配置文件目录解释，公共 paths 按主配置目录解释。读取主配置的运行入口支持 `--config` 和 `--set dotted.key=JSON值`；覆盖只对该次运行生效，不改写文件。每次运行封存原始配置、哈希、合并后实际值和执行脚本。辅助工具的参数见各脚本的 `--help`。

`load.concurrency` 是客户端并发，`swe.session_pool` 是活跃轨迹池，`server.max_num_seqs` 是引擎序列上限。三者分别配置，不相互替代。profile 的名字只是模板名，W32 或其他请求数以实际配置为准。

首次配置主要修改 `paths`、`platform`、`capacity`、`load` 和 `execution.repetitions`，并同步检查组件文件中的模型、并行数和 UCM 策略。各文件的字段、示例默认值、单位与生效条件见上表链接。组件 JSON 的顶层不再包一层 `server` / `ucm` / `aisbench` / `swe`；这些前缀用于合并配置和 `--set`，例如在 vllm.json 中写 `tensor_parallel_size`，命令行使用 `--set server.tensor_parallel_size=2`。

## 选择运行方式

根据服务是否需要长期运行、每轮是否需要重建缓存，选择以下一种方式。两种方式共用公共配置和客户端实现，分别完成各自流程，不需要先后执行。

| 比较项 | 独立启动服务与客户端 | 单配置完整实验 |
|---|---|---|
| 适用场景 | 保持服务运行，按需单独调用 AISBench 或 SWE-rebench；测试已有服务 | 按完整实验协议，测试一个配置的性能或 pipeline |
| 使用入口 | `serve.py` 管理服务；`run_aisbench.py` / `run_swe.py` 运行客户端 | `experiment_*.py` 或 `run_experiment.py --profile ...` |
| 服务生命周期 | 用户启动并停止；客户端结束后服务继续运行 | 编排器为每个校准轮和正式轮启动新服务，并在该轮结束后停止、清理 |
| 编译校准 | 独立客户端不执行完整负载编译校准编排，即使配置中 `compile_calibration=true` | 默认先执行完整同 N 负载的编译校准，再重启服务进入正式轮；次数由配置控制 |
| 每轮协议预热 | 执行相应客户端的 warmup | 执行相同的客户端 warmup |
| 缓存起点 | 使用已有服务及缓存，状态受之前请求影响；重复轮之间不重置 | 每轮使用新服务、独立 L1 命名空间和空 L2；只复用该次实验的编译缓存 |
| `--repetitions 3` | 在同一服务上连续测试三轮 | 执行三轮独立正式测试；编译校准轮另计 |
| 服务端观测 | `--service-run` 自动启用；仅 `--base-url` 默认只验证客户端，`--observe` 需要服务提供对应接口 | 自动执行容量、读写排空和编译观测；按配置判定有效性 |
| 数据与报告 | 保存独立客户端目录中的原始结果和指标，再调用 `generate_report.py` | 保存校准轮、正式轮及其服务记录，再调用 `generate_report.py`；校准轮不计入正式成绩 |

例如，在默认启用一次编译校准的情况下，完整实验指定 `--repetitions 3` 会先执行一轮编译校准，再执行三轮正式测试，共使用四次新服务；独立客户端指定同样参数则在已有服务上执行三轮，期间服务保持运行。

两种方式中的 AISBench 都执行无关预热与 N 个串行目标预热，保留自然 HBM 残留；SWE 都执行无关预热并按轨迹依赖回放。完整实验的“新服务/空缓存”指协议预热之前的起点，正式计时开始时已经包含预热产生的缓存状态。独立客户端的结果不能直接声明满足每轮新服务的历史冷启动条件。

首次使用本包管理服务时，先完成 [准备服务环境](#准备服务环境)，然后选择 [独立启动顺序](#独立启动顺序) 或 [单配置完整实验与报告](#单配置完整实验与报告)。仅连接外部已有服务的独立客户端可使用已有客户端环境，无需构建本包的服务环境。

## 准备服务环境

此步骤供独立服务与单配置完整实验共用。它准备源码副本、UCM 运行环境及观测补丁，不启动模型服务，也不执行实验或容量校准。环境与配置匹配时可以复用；更换 UCM 提交、平台或构建补丁后需重新准备。

命令从本目录执行。`paths.server_python`、`paths.aisbench_python`、`paths.swe_python` 可以分别指向三个环境。启动命令前面的 `python3` 用于运行管理脚本；`paths.server_python` 指定管理脚本实际启动 vLLM 时使用的解释器，两者可以不同。

准备环境时，`paths.server_python` 可以是容器中的 `python3`。服务启动时，该字段必须填写本包成功构建的环境中的 `venv/bin/python` 路径：服务入口会检查构建清单、UCM 源码和补丁身份，以及资源隔离和指标采集接口。这是本脚本包的启动要求，并非 vLLM 本身要求使用虚拟环境。新环境通过 `--system-site-packages` 访问基础解释器已有依赖，并在自己的目录安装构建的 UCM。

```bash
# 检查渲染后的计划，不启动服务。
python3 -B scripts/plan.py --profile swe_w24_n192_performance
```

构建入口为 `scripts/prepare_environment.py --build`。默认不安装 Python 依赖，也不联网获取原生依赖；只有新建 venv 可访问的基础环境已具备构建依赖、导出的 UCM 源码已包含所需 vendor 源码时，才能直接使用默认命令：

```bash
python3 -B scripts/prepare_environment.py --build
```

Python 依赖和原生依赖源码需要分别准备。Python wheels 通过 `build.wheelhouse` 提供；fmt、spdlog、pybind11、zlib 的源码由 UCM 的 CMake 配置获取。原仓库中未跟踪或被忽略的 vendor 目录不会随 `git archive` 导出，具体条件见 [原生依赖源码](ENVIRONMENT.md#原生依赖源码)。vLLM、PyTorch、CUDA/CANN 等平台环境仍由目标容器提供。

### 准备 Python wheels

`build.install_dependencies=true` 只从已有的本地目录安装 Python 包，**不会下载 Python wheels，也不会自动准备 `dependency-wheels/`**。已准备好 wheels 时，继续使用它们的实际目录，无需重新下载；不要将已有的 `UCM_WHEELHOUSE` 改为示例目录。

首次准备时，在目标容器中使用与 `paths.server_python` 相同的 Python 和系统架构下载。以下假设 `python3` 就是该解释器，否则将下载命令中的 `python3` 替换为对应路径。先导出所选 UCM 提交的源码和依赖清单，不执行构建或安装：

```bash
python3 -B scripts/prepare_environment.py --set build.install_dependencies=false
```

将输出目录填入 `UCM_PREPARED_RUN`。如果已有构建失败目录且其中存在 `source/requirements-build.txt` 和 `source/requirements.txt`，可直接使用该目录，无需重新导出。确认它对应当前要构建的 UCM 提交。

```bash
UCM_PREPARED_RUN="runs/实际准备目录"
export UCM_WHEELHOUSE="$PWD/dependency-wheels"

python3 -m pip download --only-binary=:all: \
  --dest "$UCM_WHEELHOUSE" \
  -r "$UCM_PREPARED_RUN/source/requirements-build.txt" \
  -r "$UCM_PREPARED_RUN/source/requirements.txt"
```

此命令需要可用的 Python 包索引，只下载包及其依赖，不安装到当前环境。目录由 pip 创建；仅执行 `mkdir` 得到空目录并不能完成准备。使用 `$PWD` 生成实际路径，使 pip 的下载位置与构建脚本读取的位置一致；配置中的相对路径以主配置文件目录为基准，而 pip 的相对路径以当前工作目录为基准。下载全部成功后，再执行构建。

### 通过命令参数构建

确认 `UCM_WHEELHOUSE` 已指向准备好的目录。该变量由使用者在当前 shell 中设置，不由脚本自动定义。目标容器能够访问 UCM 声明的源码地址时，可从本地 wheelhouse 安装 Python 依赖，同时显式开启原生源码下载：

```bash
: "${UCM_WHEELHOUSE:?请先准备 Python wheels 并设置其实际目录}"

python3 -B scripts/prepare_environment.py --build \
  --set build.install_dependencies=true \
  --set "build.wheelhouse=\"$UCM_WHEELHOUSE\"" \
  --set 'build.cmake_args=["-DDOWNLOAD_DEPENDENCE=ON","-DFETCHCONTENT_FULLY_DISCONNECTED=OFF"]'
```

`--set` 只对本次运行生效，不修改配置文件。两个 CMake 选项需同时设置，显式值覆盖默认离线设置。若原配置中还有其他 `build.cmake_args`，请一并保留在命令的数组中，因为该覆盖会替换整个数组。Python 依赖仍从本地 wheelhouse 安装，不从 Python 包索引下载；原生源码按所选 UCM 提交声明的地址和版本获取。

实际构建会在导出源码和创建 venv 前检查 wheelhouse，目录不存在、不可读或为空时直接报出配置路径并记录失败。目录检查不代替 pip 的依赖解析；如果目录有效但缺少所需版本或目标平台对应的包，需要按同一份 UCM 依赖清单补齐。`--build` 不会因为目录缺包而自动切换为联网安装。

### 将构建参数写入配置文件

需要重复使用相同构建方式时，将以下字段合并到 `config/experiment.json` 已有的 `build` 对象中。保留其他配置字段；若已有其他 CMake 参数，也要保留在数组中。`wheelhouse` 填写实际目录，路径规则与命令参数相同。

```json
{
  "build": {
    "install_dependencies": true,
    "wheelhouse": "../dependency-wheels",
    "cmake_args": [
      "-DDOWNLOAD_DEPENDENCE=ON",
      "-DFETCHCONTENT_FULLY_DISCONNECTED=OFF"
    ]
  }
}
```

保存后直接执行：

```bash
python3 -B scripts/prepare_environment.py --build
```

每次构建都会创建新的环境目录和 venv，已有 Python wheels 可以重复使用。脚本自动将新 venv 的 `bin` 放在构建子进程 `PATH` 首位，供构建后端查找 `cmake` 等工具，原有路径保留在后，无需手动修改 `PATH`。最终路径与 CMake 参数分别记录在 `build-plan.json` 的 `environment_changes.PATH` 和 `environment_changes.BUNDLE_CMAKE_ARGS` 中。

### 使用构建后的服务 Python

构建成功后，命令打印环境目录，默认形式为 `runs/experiment_service_environment_时间戳_唯一标识/`。从该目录的 `service-environment.json` 读取应填写的完整路径，将命令中的占位目录替换为实际输出目录：

```bash
python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["server_python"])' \
  /实际构建输出目录/service-environment.json
```

将打印值填入 `config/experiment.json` 已有的 `paths.server_python` 字段，保留其余配置。路径形式为 `/容器内脚本包目录/runs/实际构建目录/venv/bin/python`；也可以填写相对主配置目录的 `../runs/实际构建目录/venv/bin/python`。保留完整的 `venv/bin/python`，无需 `source activate`，也不要将其符号链接解析为系统 Python。容器中的脚本包路径可能与宿主机不同，应以构建输出为准。

不带 `--build` 的 `prepare_environment.py` 仅导出源码和构建计划，不创建可启动服务的环境。只有构建成功后生成的 `service-environment.json` 才是有效的解释器路径来源。

## 独立启动顺序

完成服务环境准备后，先启动服务，再在另一终端选择客户端命令执行。AISBench 和 SWE-rebench 客户端可以按需分别调用，每次调用都会准备输入、执行协议预热并记录结果；服务由用户决定何时停止。

```bash
# 配置 paths.server_python 后启动服务；命令前面的 python3 可以保持不变。
python3 -B scripts/serve.py start --profile swe_w24_n192_performance

# 另开终端；run_dir 使用上一步输出的新目录。
python3 -B scripts/serve.py status runs/服务运行目录

# 以下客户端命令按测试需求选择执行。
python3 -B scripts/run_aisbench.py --service-run runs/服务运行目录
python3 -B scripts/run_swe.py --service-run runs/服务运行目录 \
  --profile swe_w24_n192_performance --repetitions 1
python3 -B scripts/run_swe.py --service-run runs/服务运行目录 \
  --profile swe_w24_n3379_performance

# 主动停止这一个服务。启动终端 Ctrl-C 也会清理。
python3 -B scripts/serve.py stop runs/服务运行目录
```

客户端不会停止或重启服务。独立客户端的 repetitions 会复用该服务及其缓存，因此不声明满足历史 fresh-service 冷缓存协议。AIS 客户端保留无关预热、N 个串行目标预热、自然 HBM 残留；SWE 保留无关预热和严格因果闭环顺序。下述完整实验入口负责全负载编译校准后重启及每轮独立缓存。

已有其他方式启动的服务也可测试：

```bash
python3 -B scripts/run_aisbench.py --base-url http://127.0.0.1:18290
python3 -B scripts/run_swe.py --base-url http://127.0.0.1:18290
```

`--base-url` 默认只验证客户端；配置中的服务设置不等同对端实际设置。加 `--observe` 要求对端提供本包 BundleObserver RPC，核对容量和物理写回排空。`--service-run` 自动启用 observe，并使用已启动服务的实际模型、并行数、容量等设置，客户端 tokenizer 必须一致。任何此类外部客户端测试都不会擅自重启服务。独立 pipeline 客户端要求服务也使用 pipeline profile 启动，采集后的分析步骤见后文。

AISBench 配置省略 `block_size` 时，使用 `--service-run` 读取服务的实测值，或通过 `--observe` 查询 BundleObserver 后准备输入。只有普通 OpenAI 接口时无法读取该值，需要在客户端配置中明确填写已知的实际 `block_size`；脚本不会推测。详见 [省略 block_size](#省略-block_size)。

## 单配置完整实验与报告

完成 [准备服务环境](#准备服务环境) 并配置 `paths.server_python` 后，直接执行完整实验入口。该入口自行启动服务并调用公共客户端流程，无需预先执行 `serve.py start` 或另外调用 `run_aisbench.py` / `run_swe.py`。它不接管已有独立服务，配置中的服务端口和设备资源应可供新服务使用。

每次只运行下面某一个命令；所有命令读取同一份公共配置及其组件引用。按相同模型、TP、容量和 UCM 策略运行不同类型的实验时，配置一次后选择对应入口即可，无需随入口切换反复修改这些公共参数。客户端、测量模式、请求数及上下文长度按所选 profile 和组件配置生效。需要测试其他参数时，再修改配置或使用单次命令行覆盖；脚本不会自动遍历对照矩阵。

```bash
# 选择一个实验入口，直接使用配置文件中的容量、策略与重复次数。
python3 -B scripts/experiment_aisbench_performance.py
python3 -B scripts/experiment_aisbench_pipeline.py
python3 -B scripts/experiment_swe_n192_performance.py
python3 -B scripts/experiment_swe_n3379_performance.py
python3 -B scripts/experiment_swe_n192_pipeline.py

# 与 experiment_swe_n192_performance.py 等价的通用入口。
python3 -B scripts/run_experiment.py --profile swe_w24_n192_performance

# 测试完成后，明确选择一组或几组本包运行目录生成报告。
python3 -B scripts/generate_report.py runs/实验目录甲 runs/实验目录乙
```

公共示例配置使用 L1 physical 模式、19 GiB、一次正式测量。历史 N40 A/D21 的 L1 为 18 GiB，这是特定历史组的参数，不是 AISBench 入口的固定要求。只有需要对齐该历史容量或测试其他容量时，才需要调整；可以修改 `config/experiment.json`，也可以使用下面的单次覆盖。命令行覆盖优先于配置文件，仅影响该次运行，不会改写配置文件，生效值会保存在实验记录中。

```bash
# 可选：本次使用 L1 物理容量 18 GiB，正式测试三轮。
python3 -B scripts/experiment_aisbench_performance.py \
  --set 'capacity.l1.mode="physical"' --set capacity.l1.gib=18 --repetitions 3

# 可选：本次使用 W32、客户端并发 16；其余参数仍取公共配置。
python3 -B scripts/run_experiment.py --profile swe_w24_n192_performance \
  --set swe.session_pool=32 --set load.concurrency=16
```

按历史组完整复现还需核对源码提交、UCM 开关、packed 布局、模型、平台、输入和协议；仅设置 18 GiB 不会将当前配置切换成历史 A/D21。历史组条件见 [实验协议与迁移参考](FINDINGS.md)。

编排顺序为：冻结/校验输入一次 → 完整同 N 负载编译校准 → 停止校准服务 → 每轮启动新的服务、L1 命名空间和空 L2 → 协议预热 → 物理 drain 与容量核验 → 正式窗口 → 物理尾部 drain → 关闭服务及本次资源。只复用**本次 campaign 的编译缓存**。`execution.compile_calibration`、`calibration_repetitions`、`repetitions` 控制次数；关闭编译校准后仍须通过正式窗口编译检查。失败轮保留，不自动重试或挑选成绩。

`compilation_watch.py` 观察 TorchDynamo、可用 Triton cache-hit listener、设备 graph capture 和已加载 TorchAir GeGraph.compile，并配合缓存文件/日志审计。无法观察当前编译器 API 时会明确失败；其覆盖范围限于这些接口。Triton-Ascend/TorchAir 的 hook 尚未完成实机验证，需要适配时可使用自定义 worker extension。

报告是本地 `REPORT.md`，不使用外链模板。`RESULTS.json`、`performance.csv`、`ALL_CONFIGURATIONS.json`、`figures/` 与原始运行目录通过相对路径关联。输出格式由 `profiling.output_formats` 控制，默认 PNG/SVG/PDF。校准、失败、输入/配置不相容以及未经通过的 pipeline 不会混入正式统计；模拟数据默认排除，只有测试时显式加 `--include-validation` 才单独显示。

## Pipeline 采集与补丁

`profiling.backend=auto` 按平台选择 CUDA/Nsight 或 Ascend/torch_npu profiler。`profiling.nsys` 指定 Nsight 可执行文件。服务私有构建默认注入 `reference/patches/native-pipeline.patch`、`python-pipeline.patch` 与 `native/profile_nvtx.h`；原生 trace 使用 monotonic 时钟，不依赖 NVTX/CANN 头文件。`patch_files` 追加本地补丁，`extra_python_paths` 和 `worker_extension` 支持当前版本的额外采集适配。所有补丁只作用于导出的 UCM 副本。

采集开始前完成预热和 drain，采集结束覆盖响应后的真实 I/O 尾部。Python 请求 SHA256、实际任务/block/layer、原生读写跨度通过时钟锚点关联设备 trace；CUDA 使用 CUPTI correlation ID，Ascend 使用 trace 中的 async_npu/HostToDevice flow。缺失或歧义不会用邻近时间猜测归属。报告保留五阶段总览、逐请求分页、各层 H2D 完成、READY、credits、等待/spill、队列和请求性能图。`request_page_size` 默认 8；不会只画前几个请求。

托管 pipeline 实验在服务关闭后自动导出并分析。独立客户端保留对服务的控制权，按如下步骤手动分析；**Nsight 需要服务所有者先停止服务，让 profiler 完整落盘**：

```bash
python3 -B scripts/serve.py start --profile swe_w24_n192_pipeline
python3 -B scripts/run_swe.py --service-run runs/服务目录 --profile swe_w24_n192_pipeline
python3 -B scripts/serve.py stop runs/服务目录
python3 -B scripts/analyze_pipeline.py --plan runs/客户端目录/plan.json \
  --attempt runs/客户端目录/attempts/001
python3 -B scripts/generate_report.py runs/客户端目录
```

pipeline 的服务 warmup/客户端协议按相应类型执行。采集增加开销，与无 profiling 的性能组分别统计。Ascend trace 的 PID/TID、flow schema 和 copy 字节字段受版本影响；无法识别会保留原始数据并失败。Ascend 采集格式的支持范围尚需通过实际设备 trace 验证。

## HBM/L1/L2 容量

全部位于 **config/experiment.json → capacity**。

| 层 | 自动规划 | 按 tokens 配置 | 按物理容量配置 |
|---|---|---|---|
| HBM | hbm.mode=auto，使用 vllm.json 的 gpu_memory_utilization | hbm.mode=tokens，hbm.tokens；calculation.method=bytes_per_token 并指定 bytes_per_token，或 calibration | hbm.mode=physical，hbm.bytes |
| L1 | 不支持 | l1.mode=tokens，l1.tokens；calculation.method=layout 指定 node_bytes/nodes_per_block/reserved_nodes，或 calibration | l1.mode=physical，l1.gib（整数 GiB） |
| L2 | 不支持 | 不使用 tokens | l2.gib（整数 GiB），另选 enforcement |

### 自动 HBM 容量

将 `config/experiment.json` 中已有的 `capacity.hbm.mode` 改为 `auto`，并在 `config/vllm.json` 中设置 `gpu_memory_utilization`。例如 `0.9` 表示让引擎按每张卡总显存的 90% 规划该实例的模型运行预算；KV cache 的实际分配还需扣除模型权重、激活和运行时等占用，并按引擎 block 取整。这不是将总显存的 90% 全部分配给 KV cache，也不保证与其他进程共享设备时一定可用。

```bash
# 查看自动 HBM 计划；不启动服务。
python3 -B scripts/plan.py --profile swe_w24_n3379_performance \
  --set 'capacity.hbm.mode="auto"' --set server.gpu_memory_utilization=0.9

# 运行完整实验时也可使用相同覆盖；需要先准备目标服务环境。
python3 -B scripts/experiment_swe_n3379_performance.py \
  --set 'capacity.hbm.mode="auto"' --set server.gpu_memory_utilization=0.9
```

auto 模式下，脚本省略 `--kv-cache-memory-bytes`，由 vLLM/对应 Ascend 后端规划容量。`hbm.tokens`、`hbm.bytes` 和 HBM 换算系数不生效；旧校准文件只有在 L1 使用校准换算时才会读取。无需删除其他模式的配置字段，默认模式仍为 `tokens`。

启动前计划中的 `kv_cache_memory_bytes` 和 `hbm_target_tokens` 为 `null`，表示尚未测得实际容量。启动后 `plan.json` 的 `capacity.observed_hbm`、`startup-workers.json` 和 `capacity-validation.json` 记录每个 worker 的实际 KV 字节数及 tokens，并继续检查张量/block 一致性和 L1 容量。自动分配出的实际容量不同的记录会在报告中分组统计，缺少有效容量观测的受管服务记录不计入有效结果。需要严格固定容量做对照时，应使用 `physical` 或校准后的 `tokens` 模式。

CUDA/Ascend 的计划生成均支持 auto；实际自动规划能力仍以目标 vLLM/后端版本的运行结果为准。

### 固定容量与布局校准

HBM tokens 必须按实际 block size 对齐。配置中显式指定时提前检查；省略且使用 `bytes_per_token` 换算时，在服务启动后检查实际布局。校准换算式为 `tokens / block_size × 实测每block字节`；它指 KV 物理 block 总量，可供请求分配的上界还要扣保留 block。TP/PP/DP 下按各 worker 观察，不盲目乘除卡数。

`hbm.bytes` 配置的是设备上 KV cache 的字节预算。模型权重、激活、通信缓冲和运行时也会占用 HBM，服务器所需总显存应计入这些部分；实际卡的总 HBM 由硬件决定。L1 是 UCM 的主存缓存预算，L2 是配置的存储目录/文件系统容量。

L1 普通 tokens 为 `floor((floor(GiB×2^30/node_bytes)−reserved_nodes)/nodes_per_block)×block_size`。元数据额外占用内存；整数 GiB 未必能精确表达目标。`l1.rounding=at_least` 向上选最小可覆盖容量，`exact` 无法精确表达时失败。

历史默认 HBM 320000 tokens / 5372160000 bytes、L1 19 GiB 仅适用于历史测得布局，不能直接作为 Ascend 或其他模型/TP 的容量证据。

```bash
# 默认用 hbm.bytes、l1.gib 启动校准；hbm.mode=auto 时保留引擎自动规划。
python3 -B scripts/calibrate_capacity.py --profile swe_w24_n192_performance

# 不预先指定 KV 字节预算，由引擎自动规划后测量实际布局。
python3 -B scripts/calibrate_capacity.py --profile swe_w24_n192_performance \
  --set 'capacity.hbm.mode="auto"' --set server.gpu_memory_utilization=0.9
```

校准会实际启动服务、加载模型和分配缓存，读取布局后停止本次服务。HBM 为 auto 时保留自动规划；其他 HBM 模式使用 `hbm.bytes` 作为启动预算。L1 始终使用 `l1.gib`，校准过程中不读取已有校准文件。脚本不会通过反复尝试搜索最大容量。查看新目录中的 capacity-calibration.json、startup-workers.json、capacity-validation.json。将 `capacity.calibration_file` 指向汇总校准文件，将需要使用实测换算的 HBM/L1 `calculation.method` 改成 `calibration`，再设置 `mode=tokens` 和目标值；也可继续保留 HBM auto。脚本不会自动改写配置。校准文件绑定模型配置哈希、HF overrides、后端、TP/PP/DP、block size、KV dtype 及 UCM 布局参数，使用校准换算时身份不同则拒绝复用。

同时生成 `layouts/worker-000.json`、`worker-001.json` 等单 worker 布局，文件内记录实际 rank/PID，可直接传给 `capacity.py --layout`。例如 `--layout runs/实际校准目录/layouts/worker-000.json`；目录名需替换为脚本打印的路径。汇总文件与单 worker layout 的完整用法见 [README 的测量说明](README.md#如何测量并生成自己的-layout-json)。

L2 的 `enforcement`：

- `ucm_gc_budget`：默认，沿用历史 UCM GC 容量预算；不是物理硬配额。
- `filesystem`：将 paths.l2_root 指向已有受限文件系统，设置 filesystem_limit_gib；验证整个文件系统的容量上限，明确它可能与其他目录共享。普通大磁盘上的子目录不会被宣称为硬配额。
- `loopback`：在运行目录创建 filesystem_limit_gib 大小的 ext4 镜像并挂载到私有 L2 目录，退出时验证 backing file 再卸载清理。需要目标容器已有 mount 权限；脚本不会修改容器权限。元数据占用会降低可用数据容量。真实挂载尚未验证。

## 环境、数据与结果位置

目标镜像、客户端依赖、历史锁文件和迁移要求见 [ENVIRONMENT.md](ENVIRONMENT.md)。客户端可以继续使用各自已安装的环境，无需与服务共用 Python。

`prepare_environment.py` 只从 UCM 指定 commit 导出副本，build.jobs 控制构建并行数。Python 依赖须已可用，或在 build.wheelhouse 配置离线 wheels 并开启 install_dependencies；原生依赖需要导出源码中的 vendor 目录，或显式开启 CMake 获取。构建副本补丁记录在独立环境目录：固定提交标识、CMake 参数注入、禁止清理其他服务的旧共享内存、协调文件/SHM 使用本次命名空间，以及独立累计原生指标缓冲区。后者避免指标消费者禁用/清空造成错误的零 I/O 判断。`profiling.patch_files` 指定的附加补丁也只应用于副本。原仓库不会 checkout、reset 或 commit。

`prepare_inputs.py --profile ...` 只准备和验证输入，不启动模型或发送请求。单独准备 AISBench 输入时需显式配置已知的 `server.block_size`；省略时应通过完整实验或带服务观测的独立客户端准备输入。SWE 输入准备不依赖该字段。每次运行自动建立 runs/experiment_...，其中 config-sources/ 和 configuration.json 保存配置，executed-code/ 保存执行脚本，inputs/ 保存文本及精确 token 向量，attempts/ 保存客户端原始输出和 performance.json/request-samples.json。服务目录保存 server.log、startup-workers.json、实际容量和 resource-cleanup.json。

以下命令在已有容器内、本脚本包目录下执行。模型与缓存路径填写容器内可见的实际路径；探测命令使用服务所用的 Python，若服务使用独立虚拟环境，将 `python3` 替换为配置中 `paths.server_python` 的值。`--output` 指定新建的探测结果文件，记录脚本所在环境及该 Python 的信息。

```bash
python3 -B scripts/probe_environment.py \
  --output runs/target-container.json \
  --model /容器内模型目录 \
  --disk-path /容器内缓存目录
python3 -B scripts/prepare_inputs.py --profile aisbench_n40_performance
python3 -B scripts/prepare_inputs.py --profile swe_w24_n192_performance
python3 -B -m unittest discover -s tests -v
```

普通服务停止后只清理本次进程、记录在案的共享内存、私有 L2；配置、日志、输入、指标和编译缓存保留。身份变化或进程尚存时保留相关对象并记录失败，避免误删。测试中的 mock HTTP 结果始终标记为合成数据，不能用于真实性能结论。

离线验证覆盖配置解析、输入哈希、普通目录清理、进程身份检查及 CPU 模拟请求流程。示例镜像 vLLM-Ascend v0.30.0-a5 与参考 UCM 版本的组合尚未完成实机启动验证；该 UCM 的版本专用 patch 列表未列 v0.30.0，是否需要适配须依据构建和启动结果判断。逐项验证范围见 [ACCEPTANCE.md](ACCEPTANCE.md)。

离线源码适配测试可以通过 `UCM_SOURCE_ROOT` 指定参考 UCM 仓库；不设置时使用主配置的 `paths.ucm_repository`。这只选择测试读取的源码，不修改实验配置或原始仓库。具体命令和依赖见 [验证说明](ACCEPTANCE.md)。

## 主配置参数参考

`config/experiment.json` 管理运行环境、公共容量和实验流程，组件的具体行为由 `config_files` 引用的文件管理。下面的默认值对应随包提供的示例配置。JSON 不支持注释；说明集中保存在本节。

### 基础信息与组件引用

| 字段 | 默认值 | 含义 |
|---|---|---|
| `schema_version` | `1` | 本脚本包的配置格式版本；保持为 `1`。不是 vLLM 或 UCM 的版本。 |
| `label` | `early-hrrn-wait` | 实验标签，用于结果标识。名称不会启用 HRRN、wait 或其他策略，策略由 UCM 配置决定。 |
| `config_files.server` | `vllm.json` | vLLM 参数文件，包括 TP/PP/DP、序列上限、block size、模型 overrides 等。 |
| `config_files.server_env` | `vllm-env.json` | 服务环境变量文件；读取后合并到 `server.environment` 与 `server.platform_environment`，也支持不引用此文件。 |
| `config_files.ucm` | `ucm.json` | 原生 UCM 顶层参数、connector 列表与存储配置。 |
| `config_files.aisbench` | `aisbench.json` | AISBench 请求数、输入长度、前缀比例和数据来源。 |
| `config_files.swe` | `swe-rebench.json` | SWE 轨迹池、N192/N3379 请求数、输入来源等参数。 |

组件文件路径相对于主配置所在目录。组件字段使用合并后的名字覆盖，例如 `--set server.tensor_parallel_size=2`。UCM 提交由 `ucm_runtime.source_ref` 决定；默认 `HEAD` 会随源码仓库变化，需要固定实验版本时应填写提交哈希。

### ucm_runtime：实验脚本的 UCM 设置

这两个字段放在 `experiment.json`，用于选择构建源码和服务集成方式，不传给 UCM 原生配置。

| 字段 | 示例默认值 | 含义及生效条件 |
|---|---|---|
| `ucm_runtime.enabled` | `true` | 控制启动计划是否配置 UCM KV connector。受管服务和完整实验要求 `true`，设置为 `false` 时服务入口会拒绝启动；同步 UCM 对照应关闭 Early 预取，不能通过移除 connector 替代。 |
| `ucm_runtime.source_ref` | `HEAD` | 从 `paths.ucm_repository` 导出的 Git 提交引用。可填写提交哈希或本地已有的 tag/分支；脚本不自动 fetch 或切换原仓库。固定实验时填写完整提交哈希，更改后重新构建服务环境。 |

### paths：容器内路径

| 字段 | 含义与设置方法 |
|---|---|
| `paths.ucm_repository` | UCM 源码仓库。环境准备脚本从指定提交导出副本进行构建。 |
| `paths.aisbench_repository` | 包含 `src/aisbench_ucm` 的 AISBench 实验工具仓库。AISBench 运行依赖另由客户端 Python 环境提供。 |
| `paths.swe_repository` | 包含 `swe_bench`、工作负载及输入材料的 SWE-rebench 仓库。 |
| `paths.model` | 模型目录，默认示例为 `/models/your-model`。更换模型时同步检查 `vllm.json` 中的 `hf_overrides`。 |
| `paths.tokenizer` | 客户端分词器目录，必须与服务的实际分词行为一致。默认与模型目录相同。 |
| `paths.server_python` | 默认 `python3`，环境准备时用作基础解释器。服务启动时须填写成功构建目录的 `service-environment.json` 中 `server_python` 的值，即完整 `venv/bin/python` 路径；裸命令名不满足服务入口的环境清单定位要求。读取命令见 [准备服务环境](#准备服务环境)。 |
| `paths.aisbench_python` | AISBench 客户端解释器，默认 `python3`；需具备 AISBench 及相应依赖。 |
| `paths.swe_python` | SWE 客户端解释器，默认 `python3`；需具备 SWE 回放依赖。 |
| `paths.output_root` | 实验、日志、配置快照等输出的根目录。默认 `../runs`，相对 `config/experiment.json` 所在目录解析，即脚本包的 `runs/`。 |
| `paths.l2_root` | L2 临时缓存的父目录。默认 `null`，使用每个服务运行目录下的 `scratch/`；指定路径后仍会创建该次服务专属的 L2 子目录，停止时清理该子目录。 |

以上路径均填写容器内可见路径。路径支持 `~` 和已定义的环境变量。Python 填写裸命令名时按 `PATH` 查找；填写虚拟环境路径时保留 `venv/bin/python`。模型、仓库、结果和临时缓存使用各自的目录。

### platform：平台与可见卡

| 字段 | 默认值 | 含义 |
|---|---|---|
| `platform.kind` | `ascend` | `ascend` 使用 Ascend NPU，`cuda` 使用 NVIDIA CUDA。决定构建平台、设备环境变量和自动 profiling 后端。 |
| `platform.visible_devices` | `["0"]` | 服务可见设备的标识列表，例如 `["0", "1"]`；CUDA 设置 `CUDA_VISIBLE_DEVICES`，Ascend 设置 `ASCEND_RT_VISIBLE_DEVICES`。 |

可见卡数与并行数分别设置。增加可见卡不会自动增加 TP；需要同步修改 `vllm.json` 中的 `tensor_parallel_size`、`pipeline_parallel_size`、`data_parallel_size`。配置检查要求可见卡数不少于 `TP × PP × DP`；满足数量检查不代表模型和后端已支持该并行组合。

### capacity：HBM、L1 与 L2

`capacity` 中的容量预算会写入生成的服务配置。HBM 表示设备 KV cache 预算；L1 表示 UCM 主存缓存预算；L2 表示 UCM 存储缓存预算。HBM 不包含模型权重、激活和通信缓冲，L1 的管理元数据也会额外占用内存。`GiB = 2^30 bytes`。

| 字段 | 默认值 | 含义及生效条件 |
|---|---|---|
| `capacity.calibration_file` | `null` | 汇总 `capacity-calibration.json` 路径。HBM 或 L1 使用 tokens + calibration 时读取，并检查模型、平台、并行数及布局身份；未使用校准换算时不读取。 |
| `capacity.hbm.mode` | `tokens` | `auto` 使用 vLLM 的 `gpu_memory_utilization` 自动规划 KV 容量；`tokens` 从目标 token 数换算字节；`physical` 直接使用 `hbm.bytes`。 |
| `capacity.hbm.tokens` | `320000` | token 模式的 KV 物理 block 总容量，需为 `server.block_size` 的整数倍；请求可用容量还要扣除引擎保留 block。 |
| `capacity.hbm.bytes` | `5372160000` | physical 模式的 KV 字节预算。容量校准时，HBM 非 auto 才使用此值作为启动预算。 |
| `capacity.hbm.calculation.method` | `bytes_per_token` | token 模式下，`bytes_per_token` 使用指定系数，`calibration` 使用校准文件中的每 block 字节数。 |
| `capacity.hbm.calculation.bytes_per_token` | `16788` | `bytes_per_token` 方法的换算系数；目标字节数为 `tokens × bytes_per_token`。仅适用于对应模型、KV dtype、平台及并行布局。 |
| `capacity.hbm.calculation.basis` | 历史 CUDA 布局说明 | 换算依据的说明文字，保存在配置中，不参与计算。 |
| `capacity.l1.mode` | `physical` | `physical` 使用 `l1.gib`；`tokens` 由布局推导满足目标的 GiB 预算。 |
| `capacity.l1.gib` | `19` | physical 模式的 L1 数据预算，整数 GiB。容量校准脚本也使用此值。 |
| `capacity.l1.tokens` | `1128320` | token 模式的普通缓存容量目标，不包含加载预留节点。默认 physical 模式下，此值不决定实际容量。 |
| `capacity.l1.rounding` | `at_least` | token 模式下，`at_least` 取能覆盖目标的最小整数 GiB；`exact` 要求精确等于目标，否则报错。 |
| `capacity.l1.calculation.method` | `layout` | token 模式下，`layout` 使用下列显式布局；`calibration` 使用汇总校准文件中的 worker 布局。 |
| `capacity.l1.calculation.node_bytes` | `1093632` | `layout` 方法中，每个实际 L1 节点的数据字节数。 |
| `capacity.l1.calculation.nodes_per_block` | `1` | `layout` 方法中，存放一个完整 token block 所需的 L1 节点数。 |
| `capacity.l1.calculation.reserved_nodes` | `1024` | `layout` 方法中，从普通缓存区扣除的加载预留节点数。应与 `ucm.ucm_connectors[0].ucm_connector_config.cache_load_exclusive_buffer_number` 一致；修改换算值不会自动修改 UCM 的预留配置。 |
| `capacity.l1.calculation.basis` | 历史 packed 布局说明 | 布局依据的说明文字，不参与计算，也不会启用 UCM packed 布局。 |
| `capacity.l2.gib` | `128` | 写入 UCM `posix_capacity_gb` 的缓存/GC 预算，整数 GiB。 |
| `capacity.l2.enforcement` | `ucm_gc_budget` | 容量约束方式，见下方说明。 |
| `capacity.l2.filesystem_limit_gib` | `null` | `filesystem` 或 `loopback` 模式的文件系统容量限制，必须为不小于 `l2.gib` 的正整数 GiB；`ucm_gc_budget` 模式不使用。 |

`l2.enforcement` 的三种取值：

- `ucm_gc_budget`：使用 UCM 缓存预算，不为目录创建文件系统硬配额。
- `filesystem`：使用预先准备的受限文件系统，脚本检查其总容量不超过 `filesystem_limit_gib`。它不会自动为普通目录配置 quota。
- `loopback`：创建指定大小的 ext4 镜像并挂载到专属 L2 目录；需要容器已有挂载能力和相关工具，真实挂载尚未验证。文件系统元数据会占用部分空间。

HBM 与 L1 可以独立选择模式。默认组合是 **HBM 按 tokens 换算，L1 直接使用 19 GiB**。不生效的字段不会叠加，也不是启动失败时的备用预算。HBM auto 可与 L1 的任一种模式组合；容量校准时 L1 使用 `l1.gib`，HBM 非 auto 时使用 `hbm.bytes`。

上述换算系数来自历史 CUDA GLM14L TP1 布局。`ucm.json` 的 `posix_packed_layer_bytes` 默认是 `null`，L1 换算字段不会替它恢复历史 packed 布局。Ascend、其他模型、并行数或存储布局可配置 HBM auto 或可启动的物理预算，再按 [容量校准步骤](#hbml1l2-容量) 获取实际布局。

### load：客户端负载

| 字段 | 默认值 | 含义 |
|---|---|---|
| `load.concurrency` | `32` | 客户端最大并发请求数。SWE 同时受轨迹依赖约束，实际在途请求数可能小于该值。 |
| `load.request_rate` | `0` | AISBench 请求发送速率，单位 requests/s；`0` 表示不额外限速，仍受并发限制。SWE 当前协议要求为 `0`。 |
| `load.stream` | `true` | 使用流式响应。SWE 当前协议要求为 `true`。 |
| `load.output_tokens` | `1` | 正式请求的最大输出 token 数。默认 1 用于历史性能负载；AISBench 的协议预热单独固定为输出 1 token。 |

这里不设置总请求数。AISBench 的总数来自 `aisbench.n`；SWE 根据 profile 的 `subset` 使用 `swe.request_limit` 或 `swe.full_request_limit`。`load.concurrency`、`server.max_num_seqs` 和 `swe.session_pool` 分别控制客户端并发、引擎序列上限和输入轨迹池。

### execution：次数、编译校准、超时和采样

所有 `_s` 字段均以秒为单位。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `execution.repetitions` | `1` | 正式测试轮数，可由 `--repetitions` 覆盖。完整实验每轮重启服务并创建独立缓存；独立客户端重复使用已有服务。 |
| `execution.compile_calibration` | `true` | 完整实验在正式轮之前，以相同请求数执行编译校准，然后用新服务开始正式测试，复用该次实验的编译缓存。独立客户端入口不负责此编排。 |
| `execution.calibration_repetitions` | `1` | 编译校准轮数，只有启用 `compile_calibration` 时执行，不计入正式成绩。 |
| `execution.startup_timeout_s` | `1800` | 等待服务健康就绪的超时。 |
| `execution.request_timeout_s` | `3600` | SWE 请求及 AISBench 协议预热请求的超时。当前脚本未将此值传入 AISBench 正式推理模型配置，正式请求还受 AISBench 自身超时控制。 |
| `execution.benchmark_timeout_s` | `14400` | 单次客户端进程的最长等待时间，也用于输入准备和部分分析子进程；不是全部正式轮合计时限。 |
| `execution.drain_timeout_s` | `300` | 等待缓存读写及异步写回排空的超时，也用于部分 profiling RPC。 |
| `execution.shutdown_timeout_s` | `30` | 停止受管进程时等待退出的时间。 |
| `execution.health_poll_s` | `2` | 等待服务启动时的健康检查间隔。 |
| `execution.metrics_interval_s` | `0.5` | 启用服务观测时，正式请求窗口内服务指标采样的间隔。更短间隔会增加采样开销。 |

**编译校准与容量校准是两个流程。** `compile_calibration` 运行负载以预热编译缓存；`calibrate_capacity.py` 加载模型并读取缓存布局，不发送 benchmark 请求。每轮客户端协议中的 warmup 又是独立步骤，不计入正式请求计时。

### validation：实验有效性校验

| 字段 | 默认值 | 含义及限制 |
|---|---|---|
| `validation.require_input_identity` | `true` | 要求输入身份校验。当前配置解析器要求固定为 `true`。 |
| `validation.require_exact_capacity` | `true` | 服务及观测边界的容量检查不通过时中止。设为 `false` 仍保留检查结果，并不证明容量符合目标。 |
| `validation.require_cache_drain` | `true` | 要求读写排空协议，当前配置解析器要求固定为 `true`。 |
| `validation.reject_compilation_during_measurement` | `true` | 完整实验的正式窗口出现编译或无法通过编译审计时，拒绝该轮。关闭后结果不能声称满足正式窗口无编译条件。 |
| `validation.require_profile_attribution` | `true` | pipeline 要求请求与采集事件的有效关联，当前配置解析器要求固定为 `true`。 |
| `validation.strict_source_clean` | `true` | 环境准备及服务启动时要求原 UCM 工作树干净。构建始终从 `source_ref` 的提交导出；关闭此检查也不会将未提交修改自动带入构建。 |

这些默认值用于保留实验可比性。仅用 `--base-url` 连接外部服务且未开启 `--observe` 时，服务端容量、排空和编译观测不可用；配置为 `true` 不会让外部服务自动获得观测接口，此类结果会记录其观测边界。

### profiling：采集与报告

| 字段 | 默认值 | 含义及生效条件 |
|---|---|---|
| `profiling.instrument_native` | `true` | 准备 UCM 环境时加入原生/Python pipeline 采集支持。pipeline 要求开启，服务启动时核对其与构建环境一致；改变后需要重新构建。此值不等于每次性能测试都开启设备 trace。 |
| `profiling.backend` | `auto` | `auto` 按平台选择 CUDA 的 `nsys` 或 Ascend 的 `torch_npu`；也可显式填写这两个值，必须匹配平台。 |
| `profiling.patch_files` | `[]` | 用户补丁文件路径列表，按顺序在环境准备阶段应用到 UCM 源码副本并记录哈希。变更补丁后需要重新构建。 |
| `profiling.extra_python_paths` | `[]` | 附加服务端 Python 模块搜索目录，例如自定义采集模块所在目录。 |
| `profiling.worker_extension` | `null` | 默认使用 `runtime_observer.BundleObserver`；自定义时填写可导入的类路径，并实现脚本要求的观测 RPC 接口。 |
| `profiling.nsys` | `nsys` | CUDA pipeline 使用的 Nsight Systems 命令名或可执行文件路径。Ascend pipeline 不使用此命令。 |
| `profiling.required_stages` | 五阶段列表 | `L2_read`：存储读取；`H2D`：主存到设备；`device_compute`：设备计算；`D2H`：设备到主存；`L2_write`：写回存储。当前要求保留全部类别，实际没有发生的事件可为零。 |
| `profiling.request_page_size` | `8` | pipeline 请求图每页显示的请求数，所有请求按页保留，不影响负载。 |
| `profiling.output_formats` | `["png", "svg", "pdf"]` | 报告图片格式，可选择其中一种或多种；同时用于性能与 pipeline 报告。 |
| `profiling.keep_raw_trace` | `true` | 保留原始采集数据，当前要求固定为 `true`。 |

是否实际采集设备 trace 由所选 profile 的 `mode` 决定：`performance` 执行性能流程，`pipeline` 执行带采集的流程。pipeline 的编译校准轮使用 performance 模式，正式轮再采集。

### profiles：实验入口模板

`profiles` 是可选模板集合，一次只执行 `--profile` 选中的模板，不会遍历全部模板。

| 模板字段 | 含义 |
|---|---|
| `client` | `aisbench` 或 `swe`，决定客户端和预热协议。 |
| `mode` | `performance` 或 `pipeline`，决定是否执行 pipeline 采集与分析。 |
| `max_model_len` | 服务最大上下文长度的模板值。`vllm.json` 中 `max_model_len` 非 `null` 时优先使用组件值；该字段不是实际输入 token 数。 |
| `subset` | SWE 使用：`small` 读取 `swe.request_limit`，`full` 读取 `swe.full_request_limit`。 |

| 默认模板 | 请求数来源 | 默认上下文上限 |
|---|---|---|
| `aisbench_n40_performance` | `aisbench.n`，默认 40 | 40960 |
| `aisbench_n40_pipeline` | `aisbench.n`，默认 40 | 40960 |
| `swe_w24_n192_performance` | `swe.request_limit`，默认 192 | 131072 |
| `swe_w24_n3379_performance` | `swe.full_request_limit`，默认 3379 | 131072 |
| `swe_w24_n192_pipeline` | `swe.request_limit`，默认 192 | 131072 |

模板名中的 N40、W24、N192、N3379 是示例名称；修改请求数或 `swe.session_pool` 后，实际值以合并配置及运行计划为准。

### build：UCM 环境准备

| 字段 | 默认值 | 含义 |
|---|---|---|
| `build.jobs` | `2` | UCM 构建并行度。按 CPU 和主存资源调整，不是设备卡数或请求并发。 |
| `build.cmake_args` | `[]` | 额外 CMake 参数数组，追加在默认参数之后，同名 `-D` 设置以最后一个为准。默认离线；同时传入 `-DDOWNLOAD_DEPENDENCE=ON` 和 `-DFETCHCONTENT_FULLY_DISCONNECTED=OFF` 可显式启用原生依赖下载，见 [原生依赖源码](ENVIRONMENT.md#原生依赖源码)。 |
| `build.wheelhouse` | `null` | 离线 Python 依赖包目录；启用 `install_dependencies` 时必须填写，实际构建前须可读且非空。首次下载见 [准备 Python wheels](#准备-python-wheels)。 |
| `build.install_dependencies` | `false` | `false` 依赖基础 Python 已安装的包；`true` 在新建虚拟环境中从 wheelhouse 安装 UCM 的构建和运行依赖。 |
| `build.image_label` | `quay.io/ascend/vllm-ascend:v0.30.0-a5` | 记录镜像名称的说明字段，不拉取镜像、不启动容器、不安装此版本，也不核验镜像 digest。 |

这组参数由 `prepare_environment.py` 使用；只有加 `--build` 才实际执行构建和安装。基础容器仍需具备匹配平台的 vLLM、PyTorch 和驱动等环境，参见 [环境说明](ENVIRONMENT.md)。

## vllm.json 参数参考

[config/vllm.json](config/vllm.json) 的顶层字段合并到 `server`。下表使用合并后的字段名，命令行覆盖也使用这些名字，例如 `--set server.max_num_seqs=32`。表中的值是随包示例值；后端支持的参数组合以所选 vLLM / vllm-ascend 版本为准。

### 统一写法与新增参数

`vllm.json` 只保存服务参数，不需要 `args`、`options` 等额外嵌套层。参数名使用 vLLM 原名，将命令行的 `--` 去掉，推荐将连字符写成下划线。例如 `--api-server-count 1` 写成 `"api_server_count": 1`。也兼容 `api-server-count`，但同一参数的两种写法不能同时出现。新增原生参数无需修改脚本白名单。

下面是可以合并到 `vllm.json` 的 Ascend 参数片段，用于说明写法，不是完整配置，也不代表默认启用或适用于所有模型：

```json
{
  "api_server_count": 1,
  "quantization": "ascend",
  "trust_remote_code": true,
  "enable_auto_tool_choice": true,
  "tool_call_parser": "glm47",
  "reasoning_parser": "glm47",
  "kv_cache_dtype": "fp8",
  "attention_config": {
    "indexer_kv_dtype": "fp8"
  },
  "additional_config": {
    "enable_flashcomm1": true,
    "enable_cpu_binding": true,
    "multistream_overlap_shared_expert": true,
    "enable_dsa_cp": true
  }
}
```

JSON 参数直接写对象，不需要再转义成字符串；内部字段名保持目标引擎要求的写法。`--attention_config.indexer_kv_dtype fp8` 对应上面的嵌套对象。数组直接写 JSON 数组。新增可选顶层参数设为 `null` 表示不传给引擎，使用引擎默认值；`block_size` 也可省略或设为 `null`，`max_model_len: null` 则使用选中 profile 的上下文长度。其余已有的并行数、容量相关字段及布尔字段仍按下表要求检查类型，需保留。数组及对象由 vLLM 原生配置加载器处理；空数组遵循原生“不追加参数”的行为。

布尔开关使用 `true` / `false`。`true` 生成原生正向开关，`false` 显式生成 `--no-参数名`，避免某些 vLLM 版本通过 `--config` 加载时忽略 `false`。目标版本必须支持该负向开关；只有正向开关、默认关闭的选项，可删除该字段或设为 `null` 以使用默认值。未知参数、拼写错误以及不支持的开关由目标 vLLM 在启动时报告，离线 `plan.py` 不等于目标引擎兼容性检查。

临时覆盖同样支持新增参数和嵌套字段：

```bash
python3 -B scripts/plan.py --profile swe_w24_n3379_performance \
  --set 'server.api_server_count=1' \
  --set 'server.attention_config.indexer_kv_dtype="fp8"' \
  --set 'server.additional_config={"enable_flashcomm1":true,"enable_cpu_binding":true}'
```

以下参数已由公共配置管理，不在 `vllm.json` 中重复填写；重复定义会报错：

| vLLM 参数或设置 | 配置位置 |
|---|---|
| `model` / 命令行模型目录 | `experiment.json` → `paths.model` |
| `kv_cache_memory_bytes` | `experiment.json` → `capacity.hbm`；`auto` 时不传显式字节预算 |
| `kv_transfer_config` | `ucm.json` 与公共容量配置共同生成 |
| `worker_extension_cls` | `experiment.json` → `profiling.worker_extension` |
| `config` | `experiment.json` → `config_files.server`；脚本生成最终 vLLM 配置 |
| `CUDA_VISIBLE_DEVICES` / `ASCEND_RT_VISIBLE_DEVICES` | `experiment.json` → `platform.visible_devices` |

每次服务目录中保存 `vllm-effective.yaml`，其中包含最终模型路径、容量、UCM connector、observer 和该次上下文长度。文件采用 JSON 语法（同时是有效 YAML），由所选 `paths.server_python` 执行 `-m vllm.entrypoints.cli.main serve 模型目录 --config 本次目录/vllm-effective.yaml`。这与 `vllm serve` 使用同一个 CLI 入口，支持 `api_server_count` 等服务参数。`plan.json` 同时记录 `vllm_configuration`、`vllm_config_file`、`server_command` 和 `server_environment`。`plan.py` 仅展示内容，不创建示意路径 `PLAN_ONLY` 下的配置文件；实际服务启动时才写入该次运行目录。

原有内联 `server`、旧版 `vllm.json` 中的 `environment` / `platform_environment` / `extra_args` 仍可读取。新配置使用独立环境变量文件和直接的原生字段；`extra_args` 仅兼容旧的长选项 argv 数组，不允许覆盖已有原生字段或公共配置生成的参数，也不允许短选项别名。同一环境变量字典不能同时放在两个文件中。新增加的引擎参数参与实验分组和容量校准身份检查，更改后不能直接复用旧的布局校准文件。

### 服务地址与模型设置

| 字段 | 示例默认值 | 含义及生效条件 |
|---|---|---|
| `server.host` | `127.0.0.1` | 服务监听地址。配置校验允许 `127.0.0.1`、`localhost`、`0.0.0.0`；后者监听容器内所有 IPv4 接口。本包受管服务的客户端地址仍使用回环地址和下方端口。 |
| `server.port` | `18290` | HTTP 服务端口，整数 1–65535。与机器上已有服务避免冲突。 |
| `server.served_model_name` | `glm52-mixed` | API 对外暴露的模型名称，也是客户端请求中的 `model` 值。权重目录由 `paths.model` 指定。 |
| `server.seed` | `20260911` | vLLM 引擎启动的随机种子。AISBench 请求还会单独携带 `aisbench.seed`；SWE 的 `swe.seed` 用于负载交错顺序。 |
| `server.hf_overrides` | 下列两个模型字段 | 传给 `--hf-overrides` 的 Hugging Face 模型配置覆盖对象。换模型时检查这些字段是否适用；使用模型原始配置时可设为 `{}`。 |
| `server.hf_overrides.num_hidden_layers` | `14` | 覆盖模型隐藏层数。此值来自历史 14 层实验，不会自动裁剪或转换模型权重。 |
| `server.hf_overrides.num_nextn_predict_layers` | `0` | 历史 GLM 模型的 next-token prediction / MTP 层数覆盖，示例用 0 关闭这部分。该字段是否被识别取决于模型实现。 |
| `server.max_model_len` | `null` | 服务最大上下文长度，单位 tokens，需覆盖输入和输出。`null` 时使用选中 profile 的 `max_model_len`；填写正整数时覆盖所有 profile 的模板值。它不改变客户端输入长度。 |

### 并行、调度与 KV cache

| 字段 | 示例默认值 | 含义及生效条件 |
|---|---|---|
| `server.tensor_parallel_size` | `1` | 张量并行度 TP；模型张量在多个设备间分片。改变 TP 后需重新核对 KV 布局和容量换算。 |
| `server.data_parallel_size` | `1` | 数据并行度 DP；多个模型副本处理请求。它不等于客户端并发。 |
| `server.pipeline_parallel_size` | `1` | 流水线并行度 PP；模型层分配到多个阶段。可见设备数至少为 TP × PP × DP，且 UCM connector 必须支持所选组合。 |
| `server.max_num_seqs` | `16` | 引擎每轮调度允许的最大序列数。它与 `load.concurrency`、`swe.session_pool` 分别配置。 |
| `server.max_num_batched_tokens` | `8192` | 单次调度迭代的 token 预算，覆盖当轮 prefill/decode 工作；不是单条请求长度或整个实验 token 总数。 |
| `server.block_size` | `64`（历史示例） | 可省略或设为 `null`，由引擎决定；显式填写时必须为正整数。用于 HBM 对齐和 AISBench 前缀取整，启动后核对 worker 实际值。详见下节。 |
| `server.gpu_memory_utilization` | `0.92` | 引擎实例的设备显存使用比例，本包校验范围为 `(0,1]`。是否用于自动规划 KV 容量取决于最终命令是否指定 `--kv-cache-memory-bytes`；显式 KV 字节预算优先。字段名在 Ascend 配置中也保持不变。 |
| `server.kv_cache_dtype` | `auto` | KV cache 数据类型；`auto` 由引擎按模型及后端选择。其他取值须为目标引擎支持的类型，改变后需重新核对每 token/block 的字节数。 |
| `server.enable_prefix_caching` | `true` | 启用 vLLM 的前缀 KV 复用。此开关与 UCM Early 预取开关分别控制。 |
| `server.enable_chunked_prefill` | `true` | 允许长输入的 prefill 按调度预算分块执行，影响计算与加载的交错。 |
| `server.enable_expert_parallel` | `true` | 对支持的 MoE 模型启用专家并行相关路径；效果依赖模型、后端及 TP/DP 布局。它不是另一个设备数量字段。 |
| `server.enforce_eager` | `false` | `true` 强制 eager 执行；`false` 允许引擎使用其图执行路径，具体取决于版本和配置。完整实验仍单独审计正式窗口的编译行为。 |
| `server.async_scheduling` | `false` | 引擎异步调度开关。它不等于 UCM 的 `use_request_async_load`，也不控制客户端异步 HTTP 并发。 |

模型目录和客户端 tokenizer 分别在 `experiment.json` 的 `paths.model`、`paths.tokenizer` 中配置。默认未设置服务的 `tokenizer` 参数，服务从模型目录加载 tokenizer；如果在 `vllm.json` 中显式添加 `tokenizer`，也需同步检查客户端的 `paths.tokenizer`，保证分词行为匹配。新增原生字段中的文件路径按 vLLM 自身语义处理；本包的服务工作目录是本次运行目录，建议这些路径填写容器内绝对路径。

### 省略 block_size

从 `vllm.json` 删除 `block_size`，或设置 `"block_size": null`，均表示不向引擎传递该参数。也可用 `--set server.block_size=null` 临时覆盖，脚本不会填入 `64`、`128` 或其他值。默认配置仍保留历史示例的 `64`。

```bash
# 仅检查计划；HBM 自动规划、L1 使用已有物理容量。
python3 -B scripts/plan.py --profile aisbench_n40_performance \
  --set server.block_size=null --set 'capacity.hbm.mode="auto"' \
  --set 'capacity.l1.mode="physical"'

# 在准备好环境的目标容器中执行单配置完整实验。
python3 -B scripts/experiment_aisbench_performance.py \
  --set server.block_size=null --set 'capacity.hbm.mode="auto"' \
  --set 'capacity.l1.mode="physical"'
```

服务启动后从 BundleObserver 读取各 worker 的 `capacity_layout.block_tokens`，要求均为相同的正整数。`plan.json` 的 `observed_block_size` 和 `model_identity.block_size` 记录实测值，`configuration.server.block_size` 保留 `null`。报告展示配置值与实测值，实际值不同的记录分别分组；受管或已关联服务的记录缺少该观测时不计为有效结果。

AISBench 完整实验先启动第一轮服务，取得实测值后准备一次输入，再进行协议预热和测量。后续校准轮、正式轮复用该输入；重启后 block size 改变会停止本次实验并清理该服务。独立客户端通过 `--service-run` 或 `--observe` 获取实际值；单独离线执行 `prepare_inputs.py` 没有服务观测，需显式配置已知值。

容量配置按以下规则处理：

| 容量方式 | 省略 block_size 时的处理 |
|---|---|
| HBM `auto` / `physical`，L1 `physical` | 可直接生成启动计划，服务启动后验证实际布局。 |
| HBM `tokens` + `bytes_per_token` | 先按配置系数计算字节预算；启动后检查实际 block 对齐及目标 tokens。系数仍须与目标模型、后端匹配。 |
| HBM/L1 `tokens` + `calibration` | 使用校准文件记录的实际 block size 换算；不把该值写入引擎启动参数。实际服务必须与校准值一致，否则停止并要求重新校准。 |
| L1 `tokens` + `layout` | 无法在未知 block size 时计算启动所需 GiB；需显式指定 block size、使用校准换算或改为物理容量。 |

`calibrate_capacity.py` 同样接受 `--set server.block_size=null`，并将实测值写入校准文件的 `identity.block_size`。校准文件仍校验模型、后端、并行数及其余布局身份。不同 worker 的 block size 不一致、无法观测或与容量换算依据冲突时，即使放宽 `require_exact_capacity` 也会停止，避免继续使用错误的换算或输入对齐。

## vllm-env.json 参数参考

[config/vllm-env.json](config/vllm-env.json) 保存服务进程环境，通过 `config_files.server_env` 引用。文件顶层使用 `environment`、`platform_environment`，不包一层 `server`；读取后合并到以下字段，命令行覆盖仍使用 `server.` 前缀。

| 字段 | 示例默认值 | 含义及生效条件 |
|---|---|---|
| `server.environment` | `{"ENABLE_UCM_PATCH":"1"}` | 所有平台共用的服务环境变量字典，值必须是字符串。 |
| `server.environment.ENABLE_UCM_PATCH` | `"1"` | 启用所选 UCM 源码提供的 vLLM 集成补丁；不代表自动安装 vLLM，也不同于本包构建阶段的 profiling 补丁。支持的引擎版本由 UCM 实现决定。 |
| `server.platform_environment` | `cuda`、`ascend` 两个字典 | 只应用 `platform.kind` 对应的平台条目。同名键覆盖 `server.environment` 中的值。 |
| `server.platform_environment.cuda.GLM51_LAYERWISE_SYNC_FIX` | `"1"` | CUDA 历史 GLM layerwise 同步适配开关，由所选 UCM 补丁代码读取；换模型或 UCM 版本需核对适用性。 |
| `server.platform_environment.ascend` | `{}` | Ascend 专属环境变量，默认没有额外条目，可按目标后端补充。 |

继承进程环境后，本包依次应用公共变量、所选平台变量及受管变量。设备可见性、离线模型访问、运行目录、编译缓存和 UCM 配置路径等受管变量由脚本最后设置，应修改相应结构化配置。不要在会被快照保存的环境变量字典中填写凭据。

例如，将以下内容合并到文件中现有的 `platform_environment.ascend` 对象，环境变量值均写成字符串：

```json
{
  "HCCL_OP_EXPANSION_MODE": "AIV",
  "HCCL_BUFFSIZE": "400",
  "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
  "TASK_QUEUE_ENABLE": "1",
  "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
  "VLLM_ASCEND_ENABLE_PREFETCH_MLP": "1"
}
```

这些只是可选写法示例，默认 Ascend 环境字典仍为 `{}`。8 卡可见性在主配置中设置 `platform.visible_devices` 为 `["0","1","2","3","4","5","6","7"]`，并在 `vllm.json` 中按实际需要设置 `tensor_parallel_size` 等并行参数。

## ucm.json 参数参考

[config/ucm.json](config/ucm.json) 使用 UCM 原生结构：connector 行为放在顶层，存储参数放在 `ucm_connectors[0].ucm_connector_config`。格式对应 UCM 源码中的 `examples/ucm_config_example.yaml`，由 `ucm/utils.py` 的 `Config` 读取；保留 JSON 文件扩展名，UCM 的 YAML 读取器也能读取该 JSON 语法。

以下是结构示意，已有文件中的其他参数仍保留：

```json
{
  "use_layerwise": true,
  "enable_early_prefetch": true,
  "ucm_connectors": [
    {
      "ucm_connector_name": "UcmPipelineStore",
      "ucm_connector_config": {
        "store_pipeline": "Cache|Posix",
        "io_direct": true,
        "posix_io_engine": "aio",
        "cache_dump_capacity_policy": "wait"
      }
    }
  ]
}
```

文件不再包含本包自定义的 `launch`、`store`、`enabled`、`source_ref` 包装。启用开关和源码版本在主配置的 `ucm_runtime.enabled`、`ucm_runtime.source_ref` 中设置。默认运行行为保持原有参数值。

| 原生结构字段（合并配置路径） | 内容 |
|---|---|
| `ucm.ucm_connectors` | connector 数组。本包的容量、清理和观测协议要求恰好一个 `UcmPipelineStore`，当前使用 `Cache\|Posix` 存储链；原生格式支持传递字段，不表示已适配任意存储后端或多 connector 实验。 |
| `ucm.ucm_connectors[0].ucm_connector_name` | 默认 `UcmPipelineStore`，对应 UCM 存储工厂中的 connector 名称。 |
| `ucm.ucm_connectors[0].ucm_connector_config` | L1/L2、I/O 和缓存策略对象，各字段见下面的存储参数表。 |

新增 UCM 参数可直接放在原生位置，无需修改脚本白名单；对象和数组保留类型及嵌套结构，例如目标版本支持的顶层 `metrics_config_path` 或存储配置中的 `store_health` 对象。参数是否生效需以目标 UCM 源码和实际观测为准，离线计划检查不会验证所有原生参数。额外文件路径建议使用容器内绝对路径，因为服务工作目录是本次运行目录。

支持原生配置路径的临时覆盖，数组索引用点号加数字：

```bash
python3 -B scripts/plan.py --profile swe_w24_n3379_performance \
  --set 'ucm.enable_early_prefetch=false' \
  --set 'ucm.ucm_connectors.0.ucm_connector_config.cache_dump_capacity_policy="spill"' \
  --set 'ucm_runtime.source_ref="HEAD"'
```

旧版 `launch` / `store` 文件仍可读取，旧的 `--set ucm.launch.字段=值`、`--set ucm.store.字段=值`、`ucm.enabled`、`ucm.source_ref` 也保留为别名。读取后统一转换为原生结构；一个文件不能混用新旧结构，`enabled` / `source_ref` 不能同时定义在旧 UCM 文件与主配置中。

### 顶层参数：connector 与 Early 预取

下列范围和兼容条件按本包参考 UCM 实现说明；更换源码提交后应核对对应实现。Early 表示提前准备外部 KV；owner 表示负责某批预取任务的 worker；READY 表示预取达到可交付状态；HRRN 是结合等待时间与剩余工作量的请求优先级策略。

| 字段 | 示例默认值 | 含义及生效条件 |
|---|---|---|
| `ucm.use_layerwise` | `true` | 使用逐层 KV 加载/保存 connector；`false` 选择对应的 direct 路径。与模型、PP 及存储布局共同决定实际 connector。 |
| `ucm.use_request_async_load` | `false` | 请求级异步外部 KV 加载。参考实现只对原生 `UCMDirectConnector` / `EarlyDirectConnector` 生效，在 layerwise 等路径会关闭并给出日志。 |
| `ucm.enable_early_prefetch` | `true` | 启用 Early connector，提前准备将被消费的外部 KV。参考实现适用于原生 direct/layerwise 路径，不能假定混合注意力等 connector 支持同一流程。 |
| `ucm.early_prefetch_request_first` | `false` | `false` 使用参考实现的 HRRN 请求轮次选择；`true` 优先让最早的可提交请求继续发起读取，再使用剩余 I/O 窗口处理后续请求。 |
| `ucm.early_prefetch_min_submit_reads` | `16` | Early 读取窗口补充时希望积累的读操作数量，整数 1–64。与实际可用窗口共同决定提交门槛；不是 HTTP 并发、卡数或每次必定提交的读取数。 |
| `ucm.early_prefetch_submit_max_wait_ms` | `2` | 为等待上述读取窗口积累而允许的最长等待，单位毫秒，范围 `(0,10]`。它不是单次磁盘 I/O 或请求的完成超时。 |
| `ucm.enable_event_sync` | `true` | 使用设备事件协调 KV 拷贝与计算的依赖。改变它会改变同步行为，不能当作独立的日志或 profiling 开关。 |
| `ucm.early_owner_max_jobs` | `16` | 每个可用 owner 的在途任务数上限，正整数；不等于 worker 数。只有 owner 能力握手成功的路径才应用该额度。 |
| `ucm.early_owner_max_candidate_blocks` | `0` | 每个 owner 的候选块工作量额度。`0` 按 `max_jobs × 单任务最大块范围` 推导上限；不是无限内存，也不预先为所有候选块分配 L1。正数指定显式块数额度。 |
| `ucm.early_ready_priority` | `true` | 在服务等待队列准入时优先考虑已 READY 的 Early 请求，同时保留有界绕过/等待保护。它改变调度优先级，不会绕过 HBM 容量检查。 |
| `ucm.persist_token_threshold` | `0` | UCM 请求长度阈值，单位 tokens。参考原生 connector 对低于阈值的请求跳过相应持久化/命中流程；默认 0 不因该阈值排除正长度请求。它不是缓存容量。 |
| `ucm.use_lite` | `false` | 选择 UCM 的 lite connector。参考实现中 lite 分支优先于普通 Early/layerwise 分支；切换后不能继续假定 Early 策略和观测合同不变。 |
| `ucm.enable_record_traces` | `false` | 输出 UCM 自身请求长度、block ID 等跟踪日志。不是本包设备 pipeline 采集的总开关，后者由 profile 与 `profiling` 控制。 |

参考 owner 协议需要受支持的 MLA 共享 L1、布局及进程组，PP 等组合可能使该能力不可用。`early_owner_max_jobs=16` 不能独自证明实际启用了 owner；应查看服务启动日志和 worker 观测结果。Early、READY 适配与引擎调度接口相关，填写 JSON 成功不等于新后端/新版本已完成实机兼容验证。

### ucm_connector_config：L1、L2 与读写路径

下表字段相对于 `ucm_connectors[0].ucm_connector_config`；例如文件中写 `"cache_dump_capacity_policy": "wait"`，命令行使用 `--set ucm.ucm_connectors.0.ucm_connector_config.cache_dump_capacity_policy="wait"`。

| 字段 | 示例默认值 | 含义及生效条件 |
|---|---|---|
| `store_pipeline` | `Cache\|Posix` | UCM 存储链，使用主存 Cache 层及 POSIX 存储层。本包当前实验协议要求这个组合，其他存储链需先适配容量、观测和清理逻辑。 |
| `io_direct` | `true` | 使用直接 I/O，相关文件系统及读写地址/长度需满足实现的对齐要求；改变后也会改变 OS 页缓存参与情况。 |
| `posix_io_engine` | `aio` | POSIX 层的 I/O 实现，示例采用 Linux AIO。需具备对应系统依赖及文件系统支持；替代实现的名称以所选 UCM 版本为准。 |
| `posix_write_inflight_limit` | `0` | POSIX AIO 在途写操作的额外限制。参考实现允许 0–4096；`0` 关闭这一额外限额，正数要求 `posix_io_engine=aio`，其他原生队列限制仍然存在。 |
| `share_buffer_enable` | `true` | 启用共享内存形式的 L1 buffer。实际共享范围由模型/并行布局和 native 域身份决定；同一服务可能包含多个独立 L1 域。 |
| `cache_load_exclusive_buffer_number` | `1024` | 为加载保留的物理 L1 节点数，从普通缓存可用节点中扣除。单位是节点，不是 tokens 或 GiB；节点字节数需由实际布局确定。显式 L1 tokens 换算中的 `reserved_nodes` 应与它一致。 |
| `cache_load_backend_only` | `false` | `false` 允许加载复用 L1 命中；`true` 让加载绕过已有 L1 命中走后端。在参考实现中与整块 L1 模式不兼容。 |
| `cache_full_block_load` | `true` | 启用 Early 的整块后端读取路径，要求后端支持 full-block read。这里的整块覆盖一个 token block 的相应 KV 层数据，不是整条请求。 |
| `cache_full_block_l1` | `true` | L1 以整块布局持有 KV，改变物理节点大小和 tokens/字节换算。要求 `cache_full_block_load=true` 且 `cache_load_backend_only=false`。 |
| `cache_full_block_dump` | `true` | 启用整块 dump：组织多层 D2H 数据与后端写回。参考路径要求 Early 整块 L1、正常 DMA 及支持整块写的后端。HBM 源数据释放和 L2 物理写入完成是不同阶段。 |
| `cache_dump_reserved_blocks` | `0` | dump 队列可持有块的额外配额。`0` 取消这个额外配额，但仍需取得真实 L1 槽位；正数限制持有块数。此字段不会额外扩容 L1，也不等于加载预留节点数。 |
| `cache_dump_capacity_policy` | `wait` | L1/持有配额不足时的 dump 策略。`wait` 等待容量释放；`spill` 允许将已持有的数据提前写回以腾出容量。参考实现只接受这两个值，`wait` 要求 `cache_full_block_dump=true`。 |
| `posix_packed_layer_bytes` | `null` | `null` 时本包不向生成的存储配置写入该字段。设置数组时，每个元素为对应层在一个磁盘 KV block 中的存储字节数；参考实现要求正整数、4 KiB 对齐，并与实际层数、L1 slice 和张量布局一致。它会改变存储布局，不是任意压缩比例。 |

容量与路径由主配置统一提供：

| 生成的存储字段 | 修改位置 |
|---|---|
| `cache_buffer_capacity_gb` | `experiment.json` → `capacity.l1`，支持物理容量或 tokens 换算 |
| `posix_capacity_gb` | `experiment.json` → `capacity.l2.gib` |
| `storage_backends` | `experiment.json` → `paths.l2_root`，脚本为每次服务生成独立 L2 子目录 |

`config/ucm.json` 中省略这三个字段，或将其设为 `null`；填写非空值会报错并提示对应公共配置位置，避免重复设置。其他顶层字段和存储对象第一层字段为 `null` 时，不写入实际 UCM 配置；嵌套对象内的值保持原样。

每次服务启动前生成 `本次服务目录/ucm.json`，其中已经填入容量和私有路径。其层级与用户配置相同，可通过 `plan.json` 的 `ucm_configuration` 查看，也会通过 KV connector 的 `UCM_CONFIG_FILE` 交给 UCM。使用原生工具手动启动时，应使用补齐容量与路径的文件；仓库中的模板只定义可复用参数。

默认整块配置是一组相互依赖的开关。改变模型、层数、TP、KV dtype 或 packed 布局后，需重新核对容量；`capacity.l1.calculation.node_bytes` 只参与换算，不会自动配置 packed 层数组。新增参数保持 UCM 原生位置和字段名，具体支持情况由所选源码版本决定。

## aisbench.json 参数参考

[config/aisbench.json](config/aisbench.json) 合并到 `aisbench`。本包使用 AISBench 的 HTTP 性能测量路径；每轮先发送一次无关预热，再串行发送 N 个目标预热请求，最后测量 N 个正式请求。请求数 `n` 只统计正式窗口，不包含预热。

| 字段 | 示例默认值 | 含义及生效条件 |
|---|---|---|
| `aisbench.source` | `auto` | 输入来源，可选 `auto`、`frozen`、`synthetic`、`jsonl`，选择规则见下文。 |
| `aisbench.n` | `40` | 每轮正式请求数，正整数；同时准备 N 个对应的目标预热输入。不是并发数。 |
| `aisbench.input_tokens` | `40000` | 每条正式输入及目标预热输入的 token 长度。本包沿用客户端的精确长度检查；自定义 JSONL 不会被自动截断或补齐。还需满足输入加 `load.output_tokens` 不超过服务上下文上限。 |
| `aisbench.prefix_ratio` | `0.8992` | 目标预热输入与对应正式输入的可复用前缀比例，范围 `[0,1)`。目标前缀按 `floor(input_tokens × prefix_ratio / block_size) × block_size` 向下对齐；默认得到 35968 tokens。不是所有 N 条请求之间共享同一前缀的比例。 |
| `aisbench.seed` | `20260911` | 输入合成及正式请求顺序的随机种子，也传入预热和正式请求的 generation seed。与引擎启动的 `server.seed` 分开配置。 |
| `aisbench.shuffle` | `true` | 是否打乱正式请求的顺序；目标预热保持输入对的原始顺序。冻结负载先打乱 0–49，再保留索引小于 N 的项；其他负载打乱 0–N−1。 |
| `aisbench.dataset` | `null` | `source=jsonl` 时必须填写的正式输入 JSONL 路径。每行提供字符串 `question` 或 `prompt`，至少 N 行；读取前 N 行，再按正式顺序配置排列。路径相对本组件配置目录。 |
| `aisbench.warmup_dataset` | `null` | `source=jsonl` 时本包预热协议需要的配对输入 JSONL，格式与正式输入相同、至少 N 行。第 i 行与正式数据第 i 行配对。其他 source 不使用此路径。 |
| `aisbench.cache_salt` | `natural-formal` | 目标预热与正式请求共同携带的缓存命名空间 salt；两者保持相同才能按该命名空间复用前缀。设为空字符串或 `null` 时本包不发送 cache_salt，也不会自动清缓存。 |
| `aisbench.unrelated_cache_salt` | `natural-unrelated` | 无关预热请求使用的 salt，用于与目标预热/正式请求区分。当 `cache_salt` 为空时，本包也不发送这个 salt。 |

`source` 的选择规则：

- `auto`：N ≤ 50、输入 40000 tokens、block size=64、对齐前缀为 35968 tokens，且 tokenizer 与冻结记录匹配时复用冻结输入；不匹配则按当前 tokenizer 合成输入。不会仅因填写 `dataset` 而自动改用 JSONL。
- `frozen`：强制使用冻结输入；上述参数或 tokenizer 不匹配时报错，不自动切换来源。
- `synthetic`：使用 AISBench 适配仓库中的语料与当前本地 tokenizer 构造精确长度、前缀关系经过检查的输入；需要该仓库配套的语料文件。
- `jsonl`：使用 `dataset` / `warmup_dataset`。本包检查实际长度与输入身份，但不会自动修改文本来实现 `prefix_ratio`，也不会仅凭这个配置值证明自定义数据拥有相应前缀比例。

并发、发送速率、流式响应和输出长度分别使用 `experiment.json` 的 `load.concurrency`、`load.request_rate`、`load.stream`、`load.output_tokens`；模型别名与 tokenizer 使用公共服务/路径配置。

## swe-rebench.json 参数参考

[config/swe-rebench.json](config/swe-rebench.json) 合并到 `swe`，控制已有 SWE 轨迹请求的选择、交错和回放。一条轨迹由同一任务的多轮请求组成；回放保留同轨迹的因果依赖，后续请求等待前序响应。这里不会运行软件开发 agent 重新解题。

| 字段 | 示例默认值 | 含义及生效条件 |
|---|---|---|
| `swe.session_pool` | `24` | 生成请求交错顺序的活跃轨迹池 W，即 W24。影响轨迹交错和缓存复用间隔，不等于 HTTP 并发。本包直接支持 W24/W32；其他值需提供已准备的 `workload_dir`。 |
| `swe.request_limit` | `192` | `subset=small` 的每轮正式请求数，取 workload 顺序的前 N 个请求。目前用于小规模性能与 pipeline profile；不含预热。 |
| `swe.full_request_limit` | `3379` | `subset=full` 的每轮正式请求数，目前用于完整规模性能 profile。它是显式上限配置，不会自动跟随 workload 总数变化，也不与 `request_limit` 相加。 |
| `swe.seed` | `20260911` | 轨迹交错顺序的随机种子，不是模型采样 seed。使用预先生成的 workload 时，其 manifest 中的 seed 必须与此值一致。 |
| `swe.workload_dir` | `null` | 已准备的 workload 目录，包含 `manifest.json`、`order.jsonl`、`profiles.jsonl` 及其引用的 token 数据。非 `null` 时优先使用，路径相对本组件配置文件目录；W/seed 必须与配置一致。 |
| `swe.regenerate_for_tokenizer` | `false` | 在未指定 `workload_dir` 时，使用 `paths.tokenizer` 重新准备历史请求的 token 缓存。换 tokenizer 时应开启；新 token 长度仍须满足上下文限制。显式 workload 不会因此被重新分词。 |
| `swe.trajectory_limit` | `100` | 生成新 workload 时从冻结轨迹列表取前 N 条，随后按 W/seed 交错；不是随机抽取 N 条，也不是正式请求数。参考源列表共有 100 条，生成时要求 N 在 1–100 且不少于 W。显式 workload 不受此字段重新裁剪。 |

负载来源按以下顺序确定：

1. `workload_dir` 非 `null`：直接使用该目录；`regenerate_for_tokenizer` 和 `trajectory_limit` 不触发重新生成。
2. 未指定目录，且 `regenerate_for_tokenizer=true`、seed 不是 `20260911` 或 trajectory_limit 不是 `100`：在本次输入目录生成新 workload。只有开启 tokenizer 重生成时才使用新的 tokenizer 重新分词。
3. 其余情况：使用 SWE 仓库内的 `workloads/W{session_pool}-N3379-seed20260911`。目录名相对 `paths.swe_repository`；所选 W 的材料必须实际存在。

默认配置因此回放 W24 的冻结负载，小规模入口使用前 192 个请求，完整入口使用前 3379 个请求。减少轨迹数后，要同步确认选中 profile 的请求数不超过可用请求数，超出会报错。冻结输入要求原 tokenizer；已准备的 token-cache workload 应由使用者保证与当前服务匹配。

`swe.session_pool` 控制负载交错，`load.concurrency` 限制客户端在途请求数，`server.max_num_seqs` 限制引擎调度序列数，三者分别生效。SWE 协议要求 `load.request_rate=0`、`load.stream=true`；正式请求前的无关预热不计入 N，完整实验的编译校准轮也不计入正式成绩。正式重复轮数由 `execution.repetitions` 控制。

例如，选择小规模 profile 后设置 `swe.request_limit=512`，实际执行前 512 个请求，即使 profile 名称仍含 `n192`。可在执行前只查看计划：

```bash
python3 -B scripts/plan.py --profile swe_w24_n192_performance \
  --set swe.session_pool=32 --set swe.request_limit=512 \
  --set load.concurrency=16
```

## history-sources.example.json 参数参考

[config/history-sources.example.json](config/history-sources.example.json) 仅供 `scripts/inspect_sources.py --history-map` 使用，不在 `experiment.json` 的 `config_files` 中，不会改变服务、客户端或实验参数。普通新实验无需配置这份映射。

文件是“审计标签 → 历史实验目录”的映射。值相对于 `inspect_sources.py --workspace` 指定的目录解析，不是相对 JSON 文件位置；必须是该 workspace 内的非空相对子目录路径，不能是绝对路径、workspace 本身或越出 workspace 的路径。

| 示例键 | 示例目录值 | 用途 |
|---|---|---|
| `aisbench_n40_performance` | `aisbench-performance` | AISBench 性能实验历史材料。 |
| `aisbench_n40_pipeline` | `aisbench-pipeline` | AISBench pipeline 历史材料。 |
| `swe_w24_n192_performance` | `swe-small-performance` | SWE 小规模性能实验历史材料。 |
| `swe_w24_n3379_performance` | `swe-full-performance` | SWE 完整规模性能实验历史材料。 |
| `swe_w24_n192_pipeline` | `swe-pipeline` | SWE pipeline 历史材料。 |

这些目录名是占位示例，使用时填写真实历史材料所在的相对目录。键作为归档标签，仅允许英文字母、数字、下划线和连字符；可以增加、删除或重命名条目，不要求与 profile 同名。脚本只复制选定报告、配置和结果表，记录来源/哈希及仓库状态；它不会执行历史实验控制脚本，也不会把缺失的材料自动补齐。

```bash
python3 -B scripts/inspect_sources.py \
  --workspace /历史材料根目录 --ucm /UCM源码仓库 \
  --history-map config/history-sources.example.json \
  --output runs/new-source-audit
```

`--output` 是将创建的新目录，运行前不应已存在；检查生成的 `manifest.json` 中实际归档的文件清单。加 `--check-remote` 才执行只读远端查询。

## 修改后检查生效值

用 `plan.py` 查看合并配置、启动参数、容量换算与请求数。此命令不启动服务，也不预留设备或运行目录：

```bash
python3 -B scripts/plan.py --config config/experiment.json \
  --profile swe_w24_n3379_performance

# 示例：只查看调整并发和轮数后的计划，不改写配置文件。
python3 -B scripts/plan.py --config config/experiment.json \
  --profile swe_w24_n3379_performance \
  --set load.concurrency=16 --set execution.repetitions=3
```

`--set` 在主配置和组件配置合并后生效，同一个键后写的覆盖前写的。实验及客户端入口另外支持 `--repetitions`，其值优先于配置文件和 `--set execution.repetitions=...`。
