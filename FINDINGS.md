# 实验协议与默认值参考

默认参数取自历史 CUDA 实验的数值配置，作为可编辑示例。仓库不包含历史请求、设备标识、源码核查快照或性能成绩，不能据此宣称复现了某次实机结果。

| 模板类型 | 示例负载 | 预热与计时边界 |
|---|---|---|
| AISBench 性能 / pipeline | N40、40000 输入 tokens、1 输出 token、并发32 | 无关预热，再串行预热 N 个目标输入；保留自然 HBM 残留 |
| SWE 小规模性能 / pipeline | W24、N192、并发32、1 输出 token | 无关预热；按固定输入顺序与轨迹依赖回放 |
| SWE 全量性能 | W24、N3379、并发32、1 输出 token | 相同协议，使用完整固定请求清单 |

完整实验默认先运行完整同 N 负载的编译校准，再为每轮正式测量启动新服务、独立 L1 和空 L2；只复用该次实验的编译缓存。预热与测后写回排空不进入请求正式计时。

历史容量示例使用 GLM14L、关闭 MTP、TP/PP/DP=1、block size=64。HBM 5000 blocks 对应 320000 tokens、5372160000 bytes；公共 L1 预算19 GiB，L2预算128 GiB。历史 N40 A/D21 的 L1 示例为18 GiB。模型配置、packed 布局和预留节点不同都会改变容量换算。

默认 UCM 策略启用 Early、READY 优先级、完整 block 的 load/L1/dump，采用 wait 写回容量策略。`source_ref=HEAD` 不固定历史实现；需要固定版本时应显式填写可访问仓库的提交。`label` 和历史组名不决定实际实现。

CUDA 与 Ascend 的设备运行时、原生库、编译缓存和 profiler 需要分别适配。真实模型加载、并行支持和采集关联尚需实机验证；默认历史 CUDA 换算值不能直接作为 Ascend 容量证据。

## 可选的历史来源核查

`scripts/inspect_sources.py` 按显式目录映射读取使用者自己的历史材料；配置样例为 [history-sources.example.json](config/history-sources.example.json)。映射值是相对于 `--workspace` 的目录，不包含任何指定机器上的历史路径。

```bash
python3 -B scripts/inspect_sources.py \
  --workspace /workspace/history --ucm /workspace/ucm \
  --history-map config/history-sources.example.json \
  --output runs/source-audit
```

按需修改映射，或复制为 `config/history-sources.local.json` 后通过 `--history-map` 指定。源码核查输出可能包含本地路径和历史原文，应保留在被 Git 忽略的运行目录中。
