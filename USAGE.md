# 使用说明与配置索引

本包在已有容器内执行，不配置或启动 Docker。默认配置沿用历史实验作为示例，首次使用时先按服务器情况修改路径、平台、设备与容量。验证范围为离线测试；服务构建、推理与设备采集需要在实际运行环境中验证。

## 配置文件位置

| 文件 | 修改内容 |
|---|---|
| [config/experiment.json](config/experiment.json) | paths 中的仓库/模型/tokenizer/Python/输出/L2 路径；platform 的 cuda/ascend、visible_devices；**capacity 中的 HBM/L1/L2 容量**；公共 load 与 execution |
| [config/vllm.json](config/vllm.json) | TP/PP/DP、HF overrides、最大上下文、max_num_seqs、batch tokens、block size、KV dtype、vLLM 参数和环境变量 |
| [config/ucm.json](config/ucm.json) | UCM source_ref；launch 与 store；async/layerwise/Early、HRRN、READY、owner 数、wait/spill、预留节点、packed layout |
| [config/aisbench.json](config/aisbench.json) | n、input_tokens、prefix_ratio、seed、shuffle、正式/预热 dataset、cache salt |
| [config/swe-rebench.json](config/swe-rebench.json) | session_pool=W24/W32、request_limit=192、full_request_limit=3379、seed、workload_dir、换 tokenizer 时 regenerate_for_tokenizer |

主配置 `config_files` 引用四个组件配置，可以改为其他路径；同一 section 不允许同时内联和外部引用。相对数据路径按所属配置文件目录解释，公共 paths 按主配置目录解释。读取主配置的运行入口支持 `--config` 和 `--set dotted.key=JSON值`；覆盖只对该次运行生效，不改写文件。每次运行封存原始配置、哈希、合并后实际值和执行脚本。辅助工具的参数见各脚本的 `--help`。

`load.concurrency` 是客户端并发，`swe.session_pool` 是活跃轨迹池，`server.max_num_seqs` 是引擎序列上限。三者分别配置，不相互替代。profile 的名字只是模板名，W32 或其他请求数以实际配置为准。

首次配置主要修改 `paths`、`platform`、`capacity`、`load` 和 `execution.repetitions`，并同步检查组件文件中的模型、并行数和 UCM 策略。全部字段的默认值、单位与生效条件见 [主配置参数参考](#主配置参数参考)。

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

# 在目标容器中实际构建私有 venv，不安装到原环境、不联网下载。
python3 -B scripts/prepare_environment.py --build
```

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

| 层 | 按 tokens 配置 | 按物理容量配置 |
|---|---|---|
| HBM | hbm.mode=tokens，hbm.tokens；calculation.method=bytes_per_token 并指定 bytes_per_token，或 calibration | hbm.mode=physical，hbm.bytes |
| L1 | l1.mode=tokens，l1.tokens；calculation.method=layout 指定 node_bytes/nodes_per_block/reserved_nodes，或 calibration | l1.mode=physical，l1.gib（整数 GiB） |
| L2 | 不使用 tokens | l2.gib（整数 GiB），另选 enforcement |

HBM tokens 必须按 server.block_size 对齐。换算式为 `tokens / block_size × 实测每block字节`；它指 KV 物理 block 总量，可供请求分配的上界还要扣保留 block。TP/PP/DP 下按各 worker 观察，不盲目乘除卡数。

`hbm.bytes` 配置的是设备上 KV cache 的字节预算。模型权重、激活、通信缓冲和运行时也会占用 HBM，服务器所需总显存应计入这些部分；实际卡的总 HBM 由硬件决定。L1 是 UCM 的主存缓存预算，L2 是配置的存储目录/文件系统容量。

L1 普通 tokens 为 `floor((floor(GiB×2^30/node_bytes)−reserved_nodes)/nodes_per_block)×block_size`。元数据额外占用内存；整数 GiB 未必能精确表达目标。`l1.rounding=at_least` 向上选最小可覆盖容量，`exact` 无法精确表达时失败。

历史默认 HBM 320000 tokens / 5372160000 bytes、L1 19 GiB 仅适用于历史测得布局，不能直接作为 Ascend 或其他模型/TP 的容量证据。

```bash
# 用配置中的 hbm.bytes、l1.gib 启动校准服务，测量 worker 布局后自动停止。
python3 -B scripts/calibrate_capacity.py --profile swe_w24_n192_performance
```

校准会实际启动服务、加载模型和分配缓存，读取布局后停止本次服务；不会自动搜索最大容量。查看新目录中的 capacity-calibration.json、startup-workers.json、capacity-validation.json。将 `capacity.calibration_file` 指向汇总校准文件，将需要使用实测换算的 HBM/L1 `calculation.method` 改成 `calibration`，再设置 `mode=tokens` 和目标值。脚本不会自动改写配置。校准文件绑定模型配置哈希、HF overrides、后端、TP/PP/DP、block size、KV dtype 及 UCM 布局参数，身份不同则拒绝复用。

同时生成 `layouts/worker-000.json`、`worker-001.json` 等单 worker 布局，文件内记录实际 rank/PID，可直接传给 `capacity.py --layout`。例如 `--layout runs/实际校准目录/layouts/worker-000.json`；目录名需替换为脚本打印的路径。汇总文件与单 worker layout 的完整用法见 [README 的测量说明](README.md#如何测量并生成自己的-layout-json)。

L2 的 `enforcement`：

- `ucm_gc_budget`：默认，沿用历史 UCM GC 容量预算；不是物理硬配额。
- `filesystem`：将 paths.l2_root 指向已有受限文件系统，设置 filesystem_limit_gib；验证整个文件系统的容量上限，明确它可能与其他目录共享。普通大磁盘上的子目录不会被宣称为硬配额。
- `loopback`：在运行目录创建 filesystem_limit_gib 大小的 ext4 镜像并挂载到私有 L2 目录，退出时验证 backing file 再卸载清理。需要目标容器已有 mount 权限；脚本不会修改容器权限。元数据占用会降低可用数据容量。真实挂载尚未验证。

## 环境、数据与结果位置

目标镜像、客户端依赖、历史锁文件和迁移要求见 [ENVIRONMENT.md](ENVIRONMENT.md)。客户端可以继续使用各自已安装的环境，无需与服务共用 Python。

`prepare_environment.py` 只从 UCM 指定 commit 导出副本，build.jobs 控制构建并行数。依赖须已在镜像中，或在 build.wheelhouse 配置离线 wheels 并开启 install_dependencies。构建副本补丁记录在独立环境目录：固定提交标识、离线 CMake、禁止清理其他服务的旧共享内存、协调文件/SHM 使用本次命名空间，以及独立累计原生指标缓冲区。后者避免指标消费者禁用/清空造成错误的零 I/O 判断。`profiling.patch_files` 指定的附加补丁也只应用于副本。原仓库不会 checkout、reset 或 commit。

`prepare_inputs.py --profile ...` 只准备和验证输入，不启动模型或发送请求。每次运行自动建立 runs/experiment_...，其中 config-sources/ 和 configuration.json 保存配置，executed-code/ 保存执行脚本，inputs/ 保存文本及精确 token 向量，attempts/ 保存客户端原始输出和 performance.json/request-samples.json。服务目录保存 server.log、startup-workers.json、实际容量和 resource-cleanup.json。

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

## 主配置参数参考

`config/experiment.json` 管理运行环境、公共容量和实验流程，组件的具体行为由 `config_files` 引用的文件管理。下面的默认值对应随包提供的示例配置。JSON 不支持注释；说明集中保存在本节。

### 基础信息与组件引用

| 字段 | 默认值 | 含义 |
|---|---|---|
| `schema_version` | `1` | 本脚本包的配置格式版本；保持为 `1`。不是 vLLM 或 UCM 的版本。 |
| `label` | `early-hrrn-wait` | 实验标签，用于结果标识。名称不会启用 HRRN、wait 或其他策略，策略由 UCM 配置决定。 |
| `config_files.server` | `vllm.json` | vLLM 参数文件，包括 TP/PP/DP、序列上限、block size、模型 overrides 等。 |
| `config_files.ucm` | `ucm.json` | UCM 源码引用、Early/READY、wait/spill、packed 布局等参数。 |
| `config_files.aisbench` | `aisbench.json` | AISBench 请求数、输入长度、前缀比例和数据来源。 |
| `config_files.swe` | `swe-rebench.json` | SWE 轨迹池、N192/N3379 请求数、输入来源等参数。 |

组件文件路径相对于主配置所在目录。组件字段使用合并后的名字覆盖，例如 `--set server.tensor_parallel_size=2`。UCM 提交由 `ucm.source_ref` 决定；默认 `HEAD` 会随源码仓库变化，需要固定实验版本时应填写提交哈希。

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
| `capacity.calibration_file` | `null` | `calibrate_capacity.py` 生成的汇总 `capacity-calibration.json` 路径。设置后检查模型、平台、并行数及布局身份；换模型或平台后不能直接复用。 |
| `capacity.hbm.mode` | `tokens` | `tokens` 从目标 token 数换算字节；`physical` 直接使用 `hbm.bytes`。 |
| `capacity.hbm.tokens` | `320000` | token 模式的 KV 物理 block 总容量，需为 `server.block_size` 的整数倍；请求可用容量还要扣除引擎保留 block。 |
| `capacity.hbm.bytes` | `5372160000` | physical 模式的 KV 字节预算。容量校准脚本也使用此值作为启动预算。 |
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
| `capacity.l1.calculation.reserved_nodes` | `1024` | `layout` 方法中，从普通缓存区扣除的加载预留节点数。应与 `ucm.store.cache_load_exclusive_buffer_number` 一致；修改换算值不会自动修改 UCM 的预留配置。 |
| `capacity.l1.calculation.basis` | 历史 packed 布局说明 | 布局依据的说明文字，不参与计算，也不会启用 UCM packed 布局。 |
| `capacity.l2.gib` | `128` | 写入 UCM `posix_capacity_gb` 的缓存/GC 预算，整数 GiB。 |
| `capacity.l2.enforcement` | `ucm_gc_budget` | 容量约束方式，见下方说明。 |
| `capacity.l2.filesystem_limit_gib` | `null` | `filesystem` 或 `loopback` 模式的文件系统容量限制，必须为不小于 `l2.gib` 的正整数 GiB；`ucm_gc_budget` 模式不使用。 |

`l2.enforcement` 的三种取值：

- `ucm_gc_budget`：使用 UCM 缓存预算，不为目录创建文件系统硬配额。
- `filesystem`：使用预先准备的受限文件系统，脚本检查其总容量不超过 `filesystem_limit_gib`。它不会自动为普通目录配置 quota。
- `loopback`：创建指定大小的 ext4 镜像并挂载到专属 L2 目录；需要容器已有挂载能力和相关工具，真实挂载尚未验证。文件系统元数据会占用部分空间。

HBM 与 L1 可以独立选择模式。默认组合是 **HBM 按 tokens 换算，L1 直接使用 19 GiB**。inactive 字段不会叠加生效，也不是在启动失败时自动使用的备用预算。保留 `hbm.bytes` 和 `l1.gib` 可供容量校准启动使用。

上述换算系数来自历史 CUDA GLM14L TP1 布局。`ucm.json` 的 `posix_packed_layer_bytes` 默认是 `null`，L1 换算字段不会替它恢复历史 packed 布局。Ascend、其他模型、并行数或存储布局应先配置可启动的物理预算，再按 [容量校准步骤](#hbml1l2-容量) 获取实际布局。

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
| `build.cmake_args` | `[]` | 额外 CMake 参数数组，例如 `["-DCMAKE_BUILD_TYPE=Release"]。构建脚本仍会强制关闭联网下载依赖。 |
| `build.wheelhouse` | `null` | 离线 Python 依赖包目录；启用 `install_dependencies` 时必须填写。 |
| `build.install_dependencies` | `false` | `false` 依赖基础 Python 已安装的包；`true` 在新建虚拟环境中从 wheelhouse 安装 UCM 的构建和运行依赖。 |
| `build.image_label` | `quay.io/ascend/vllm-ascend:v0.30.0-a5` | 记录镜像名称的说明字段，不拉取镜像、不启动容器、不安装此版本，也不核验镜像 digest。 |

这组参数由 `prepare_environment.py` 使用；只有加 `--build` 才实际执行构建和安装。基础容器仍需具备匹配平台的 vLLM、PyTorch 和驱动等环境，参见 [环境说明](ENVIRONMENT.md)。

### 修改后检查生效值

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
