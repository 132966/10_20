# Chunk 共享尾部准备：已发布索引边界修正版

独立 Triton 实验，GPU NVIDIA H20-3e；没有修改原 SGLang 项目，没有接入完整查询或 attention。

## 本次修正

旧实验用 `(n-1)//1024*1024` 推算已建索引边界，这是不成立的：chunk执行期间即使长度跨过1K边界，也不代表新桶已经完成并发布。
历史65536、4K chunk、2K window时，旧式使最后一个Q的tail从67584开始，漏掉没有索引的65536～67583。
新增 GPU `indexed_end: [1], int32`，表示本次查询实际可用的连续索引覆盖终点（不包含终点）。它是动态输入，不由Q长度推算，不是编译期常量。

```python
sink = min(n, SINK)
e = min(max(indexed_end, 0), n)
tail = max(sink, min(e, max(sink, n - WINDOW)))
count = sink + n - tail
```

候选集合严格等于：`sink ∪ 最近WINDOW个可见位置 ∪ [indexed_end,n)`。
这只扩大精筛候选，不意味着所有未建索引token都直接进入attention；最终仍可Top2048。

## 文件

- `union.py`：`stage_union` 融合公共页复制、逐Q元数据；`PreparedStage` 预绑定视图和启动参数。
- `baseline.py`：原项目函数的未修改快照，仅用于溯源，**不再用于正确性或性能基线**。
- `baseline_published.py`：修正已发布边界后的逐Q复制基线；与共享方案使用相同候选语义。
- `consumer.py`：直接读取共享缓存的示例精筛打分，1 K head/32 Q heads，没有逐Q重新复制。
- `test_bench.py`：逐字节对比、独立集合覆盖检查、6种搬运配置与4种基线GROUP调优。
- `test_published_end.py`：4K遗漏回归、索引滞后、非对齐边界、同图动态修改indexed_end。
- `test_prepared.py`：准备算子计时，空长度/增长/缩短回放。
- `test_consumer.py`：1K和4K Q打分、PyTorch参考、65537进入最后Q的Top2048回归。
- `provenance.json`：原项目文件路径和SHA256，测试验证原文件未改变。

旧实验结果在Git历史中保留；本地另有 `archive_before_published_end_fix/` 快照。旧加速比不能代表修正后的语义。

## 输入输出与调用

| 参数 | shape / dtype | 含义 |
|---|---|---|
| src | [P,8448], uint8 | 每页8192字节FP8 K＋256字节FP32 scale |
| nt | [NP], int32 | 同一请求的逻辑页→原生物理页 |
| lens | [R], int32 | 各Q动态可见长度，非负且单调不减 |
| indexed_end | [1], int32 | 实际发布且可用的连续索引覆盖终点 |
| dst | [U,8448], uint8 | 公共页缓存预分配容量 |
| meta | [R,3], int32 | 每Q的[sink数量,tail逻辑起点,候选数量] |
| desc | [4], int32 | [sink页数,tail逻辑页起点,结束页上界,有效公共页数] |

```python
indexed_end = torch.tensor([65536], dtype=torch.int32, device='cuda')
plan = PreparedStage(src, nt, lens, indexed_end, dst, meta, desc,
                     sink=64, window=2048)
plan()  # 支持CUDA Graph捕获；原地更新lens、nt、indexed_end后回放
```

每个请求/层/K head的实际发布进度可能不同，调用者必须给当前这一次查询对应的进度，不能用全局最大完成编号。
只有连续前缀已发布且查询流通过事件可见时，才能推进indexed_end；有空洞应停在最早未覆盖位置。
当前实现以一个请求、一个K head、整个chunk同一个发布快照为契约。
**复制到打分完成期间不可并发修改lens、nt、indexed_end或覆盖dst**。推进发布进度不需要重捕图，但要按流/事件保证顺序。

## 4K具体例子

历史65536，R=4096，SINK=64，WINDOW=2048，indexed_end=65536。

| Q | 可见长度 | tail起点 | sink+tail候选数 |
|---|---:|---:|---:|
| 0 | 65537 | 63489 | 2112 |
| 2047 | 67584 | 65536 | 2112 |
| 2048 | 67585 | 65536 | 2113 |
| 4095 | 69632 | 65536 | 4160 |

最后一个Q包含65537。公共页仍为原逻辑页0和992～1087，`desc=[1,992,1088,97]`。
最后Q读取公共页33的页内位置1，即逻辑65537；不会因为它在2K窗口外而被丢弃。

## 布局、容量和性能优化

公共页区间是 `[0,ceil(sink_last/64))` 与 `[max(sink_pages,floor(tail_first/64)),ceil(n_last/64))`。
页只复制一次，以int32视图搬运原始位模式；4K默认512 word/program、4warps，1K默认256 word/program。
元数据生成融合到同一次启动。消费端按候选编号还原logical，再映射共享页，不复制成每Q独立缓冲区。
按整页复制可能搬入候选之外的字节，但consumer必须根据每Q count及logical访问；示例严格过滤未来位置。

容量必须依据**支持的最大可见长度和最小发布终点**计算。若索引可能滞后，不能只按chunk大小＋窗口估计容量。
测试 `Case(..., min_indexed_end=...)` 显式预留动态回放最坏边界对应的共享空间和逐Q基线空间。
PreparedStage不做GPU→CPU读回，调用者须验证实际端点、页表和容量；不是自动扩容接口。
旧精筛页表不能直接复用：需接入此处共享地址映射和mask。本实验只提供示例consumer，没有修改项目DeepGEMM/TopK。

## 正确性

修正逐Q基线与共享结果逐字节一致，包括K、scale、末页补零和未写哨兵。
独立集合oracle逐Q核对sink、window、全部未建索引可见token的并集，不只对照同一公式实现。
9组基础场景×6配置，以及5组发布边界回归×6配置通过；覆盖索引滞后4096、非对齐终点、无已建索引、零窗口。
同一Graph在不改变Q长度时，把indexed_end依次更新为65536、61440、61001、65536，并改变物理页表，均通过。
1K/4K打分与逐Q基线完全相同；首/中/末Q通过PyTorch参考。
另给token65537注入高分，确认它进入4K chunk最后Q的Top2048。TopK回归只验证示例尾部候选，不代表完整V1模型精度。

## 修正后性能：仅尾部准备

两边都使用实际发布终点，表中indexed_end等于历史长度；基线GROUP=1/2/4/8调优。
Graph：triton do_bench_cudagraph(rep=50ms)。非Graph：CUDA Event计时50次Python连续提交，5组中位数，包含GPU等CPU提交的间隙。
初始化、预分配、编译不计时；warm microbenchmark，未锁GPU频率，计时存在波动，不是模型吞吐。

| 历史长度 | Q数 | sink/window | 修正基线非Graph μs | 共享非Graph μs | 加速 | 修正基线Graph μs | 共享Graph μs | 加速 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 65536 | 1024 | 64/2048 | 77.47 | 15.16 | 5.11× | 89.20 | 1.81 | 49.23× |
| 131072 | 1024 | 64/2048 | 78.58 | 15.42 | 5.10× | 89.08 | 1.82 | 49.04× |
| 262144 | 1024 | 64/2048 | 77.54 | 15.42 | 5.03× | 90.66 | 1.77 | 51.26× |
| 65549 | 1024 | 17/2049 | 76.97 | 15.42 | 4.99× | 88.02 | 1.75 | 50.39× |
| 65536 | 4096 | 64/2048 | 447.59 | 15.27 | 29.31× | 479.18 | 1.82 | 263.90× |
| 65536 | 1024 | 64/128 | 28.94 | 15.26 | 1.90× | 29.33 | 1.74 | 16.87× |

4K案例：逐Q基线实际写入1361.25 MiB，共享写入（含元数据）848.27 KiB。逐Q预留2145.00 MiB，共享预留含元数据0.844 MiB。

## 准备＋示例尾部打分

| Q数 | 非Graph基线 μs | 非Graph共享 μs | 加速 | Graph基线 μs | Graph共享 μs | 加速 |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 322.61 | 261.65 | 1.23× | 327.02 | 252.81 | 1.29× |
| 4096 | 2343.45 | 1959.46 | 1.20× | 2343.22 | 1950.98 | 1.20× |

此项不包含历史重排页的精筛、粗筛、最终TopK和attention，不能称为完整查询加速比。

## 运行

```bash
bash /home/hadoop-friday-llm/tail_union_lab_20261021/run.sh
```

环境依赖路径保留为本机配置；其他机器需要自行提供PyTorch、Triton及CUDA环境。
原始结果：results.json、prepared_results.json、consumer_results.json、published_end_results.json。
