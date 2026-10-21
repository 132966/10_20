import json
from pathlib import Path
p=Path(__file__).parent
r=json.loads((p/'results.json').read_text());q=json.loads((p/'prepared_results.json').read_text());c=json.loads((p/'consumer_results.json').read_text())
s='''# Chunk 共享尾部准备：独立算子实验

此目录与 SGLang 项目无依赖，未修改原实现。GPU：NVIDIA H20-3e。
只优化 sink/tail 的 Index K+scale 搬运，不含建索引、粗筛、TopK 或 attention。

## 实现

- `baseline.py`：原 `stage_tail_shared` 的逐字函数快照，原文件路径及 SHA256 在 `provenance.json`。
- `union.py`：新 Triton `stage_union`，以及推荐入口 `PreparedStage`。
- `consumer.py`：直接读取共享缓存的示例尾部打分 kernel；1 K head、32 Q heads。
- `test_bench.py`：逐字节正确性、6种配置调优、原算子 GROUP=1/2/4/8 调优。
- `test_prepared.py`：预绑定视图入口性能、同一 Graph 清零/增长/缩短长度及更换物理页表。
- `test_consumer.py`：1024 Q 的打分逐项对比、PyTorch 独立参考、准备+打分计时。

## 输入输出

| 输入/输出 | shape / dtype | 含义 |
|---|---|---|
| src | [P,8448], uint8 | 一页8192字节FP8 K，接256字节FP32 scale；单K head |
| nt | [NP], int32 | 一个请求的逻辑页到物理页映射，所有Q共用 |
| lens | [R], int32 | 各Q动态可见长度，必须非负且单调不减 |
| dst | [U,8448], uint8 | 预分配公共候选页容量，回放时不得超容量 |
| meta | [R,3], int32 | 每Q的 [sink数量, tail逻辑起点, 候选数量] |
| desc | [4], int32 | [sink页数, tail逻辑页起点, 末尾逻辑页上界, 有效公共页数] |

stage kernel 没有 Q 输入，无需人为增加 Q-head 维度。

## 方法

沿用原式逐Q计算 sink/tail，不改变任何候选集合。因长度单调，首Q的tail起点最早，末Q的可见终点最远。
公共页是 `[0, ceil(sink_last/64))` 与 `[max(sink_pages, floor(tail_first/64)), ceil(n_last/64))`。
两个页区间不会重复。完整页按int32视图搬运，保留FP8和FP32原始位模式，不进行数值转换。
复制和逐Q元数据生成融合为一次启动。默认1K chunk用256个word/program、4warps；4K chunk用512个word/program。
逐页4096word方案和其他tile大小也经过测试，当前场景未胜过这些配置。

完整页可能包含边界外的字节，它们只是存储，不属于任何Q的候选。
消费端对候选编号 i 要求 `i<count`，先生成逻辑编号：`logical=i if i<sink else tail+i-sink`。
再令 `lp=logical//64`，共享页编号是 `lp if lp<sink_pages else sink_pages+lp-tail_page`。
页内位置是 `logical%64`。示例consumer按此映射直接打分，不重建每Q缓存，不泄漏未来token。
`reconstruct_for_test` 仅用于逐字节比较，不在优化路径计时中。

**不是原 fine_table 的直接替换**：旧接口把每Q尾部连续打包，新布局需要消费端上述映射及mask。
本目录实现了可验证的示例consumer，但没有接入项目DeepGEMM或完整查询。

## 正确性

9组场景×6配置通过；包括短序列、非64对齐sink/window、跨桶、零窗口、1K/4K chunk。
原输出与共享映射重建结果按全部字节比较一致，含scale和末页补零；空闲未写区域保留哨兵。
同一Graph更新长度与页表通过，另验证空长度→增长→缩短→增长，输出一致。
尾部打分输入Q=[1024,32,128]，新旧分数完全一致，首/中/末Q通过PyTorch FP32参考核对。
原文件SHA256复核一致。

## 性能

下表仅计 **尾部准备**，使用 PreparedStage，包含逐Q元数据输出。
Graph列用triton do_bench_cudagraph(rep=50ms)；非Graph列用CUDA Event覆盖50次Python连续提交，5组中位数。
非Graph包含GPU等待CPU提交的间隙，不是纯kernel执行时间。预分配/初始化/JIT编译均在计时之外。
基线逐Q owner、自身原算子，并在GROUP=1/2/4/8中选最快；新旧计时顺序固定，数值有频率/缓存影响。
这里报告的是warm replay microbenchmark，不是端到端模型吞吐率。

| 历史长度 | Q数 | sink/window | 基线非Graph μs | 新非Graph μs | 加速 | 基线Graph μs | 新Graph μs | 加速 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
'''
for x in q['results']:
 h,n,si,w=x['case'];s+=f"| {h} | {n} | {si}/{w} | {x['old_eager_us']:.2f} | {x['prepared_eager_us']:.2f} | {x['eager_speedup']:.2f}× | {x['old_graph_us']:.2f} | {x['prepared_graph_us']:.2f} | {x['graph_speedup']:.2f}× |\n"
x=r['results'][0]
s+=f"\n64K历史、1K chunk、64 sink/2048 window：基线目标缓存 {x['old_reserved_bytes']/2**20:.2f} MiB，公共预留缓存及元数据 {x['new_reserved_bytes']/2**20:.3f} MiB；实际写入从 {x['old_written_bytes']/2**20:.2f} MiB 降到 {x['new_written_bytes']/1024:.2f} KiB。以上不计双方共用原生pool；新接口还省去重复的逐Q原生页表。\n"
s+=f"\n### 准备＋示例尾部打分\n\n非对齐案例：历史65549，1024 Q，sink17/window2049。\n\n| 模式 | 基线 μs | 新实现 μs | 加速 |\n|---|---:|---:|---:|\n| 非Graph | {c['eager_old_us']:.2f} | {c['eager_new_us']:.2f} | {c['eager_speedup']:.2f}× |\n| Graph | {c['graph_old_us']:.2f} | {c['graph_new_us']:.2f} | {c['graph_speedup']:.2f}× |\n\n此项使用通用launch入口；不是项目精筛性能，也不包含粗筛、最终TopK和attention，不能当作完整查询加速比。\n"
s+='''
## 使用与约束

```python
plan = PreparedStage(src, nt, lens, dst, meta, desc, sink=64, window=2048)
plan()  # 或在CUDA Graph中捕获；之后原地更新lens/nt再回放
```

同一请求、单个K head、单调长度是契约。不同请求/不同K head分别处理，不能把不同请求共用一个nt。
上游须保证原生页已写完、其生命周期覆盖复制；后续consumer在同一流或等待完成事件。
共享dst也必须等consumer结束后再复用，不能被下一chunk提前覆盖。
预留U需覆盖所允许的全部回放长度和union范围；此实验用固定window和chunk容量给保守上界。
热路径不做CPU .item()、不同步、不动态分配；调用前验证容量/页表有效性是调用者责任。

运行全部测试并生成本文件：

```bash
bash /home/hadoop-friday-llm/tail_union_lab_20261021/run.sh
```

原始数字：`results.json`、`prepared_results.json`、`consumer_results.json`。
'''
(p/'README.md').write_text(s)
print(p/'README.md')
