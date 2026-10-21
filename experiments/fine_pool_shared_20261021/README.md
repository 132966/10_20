# 同一精筛池共享尾部：原 DeepGEMM 分页接口实验

本实验采用用户确认的方案：**先从原生池复制一次公共sink＋tail到fine池；DeepGEMM精筛只读取这个fine池。没有实现精筛直接读取原生池，也没有修改DeepGEMM的kernel。**

原SGLang项目及此前的tail_union实验保持不变。这是新的独立实验，包含真实DeepGEMM、动态CUDA Graph、边界恢复和输出映射验证，尚未部署到模型运行路径。

最近六次相关问答见 [RECENT_SIX_QA.md](RECENT_SIX_QA.md)。

## 1. 实现文件

| 文件 | 作用 | 实现 |
|---|---|---|
| `tail_union.py` | 复制公共sink＋tail页到fine池的共享区域，生成范围元数据 | Triton；复制自已验证的published-end修正版 |
| `baseline_published.py` | 每Q独立复制的公平基线，同样使用实际indexed_end | Triton |
| `kernels.py::tables` | 生成每Q的单池候选页表、打分存储长度和紧凑候选长度 | Triton |
| `kernels.py::normalize` | 过滤边界、因果/重叠成员，同时把分数恢复成旧式候选顺序 | Triton |
| `kernels.py::map_topk` | TopK候选编号→逻辑位置→原生物理slot | Triton |
| `experiment.py` | 缓冲区、原DeepGEMM调用与两条实验路径 | Python |
| `test_and_bench.py` | 正确性、动态Graph和三段性能测试 | Python/PyTorch |

实际打分调用未修改：

```python
schedule = deep_gemm.get_paged_mqa_logits_metadata(dg_lens, 64, sm_count)
raw = deep_gemm.fp8_paged_mqa_logits(
    q, fine_pool, weights, dg_lens, fine_table, schedule,
    max_seq_len, clean_logits=False,
)
```

每次回放都重新生成动态长度的调度信息，没有把长度和预算变成静态常量。

## 2. 输入和缓冲区：64K历史、4K Q、40个历史候选页

单K head、32个Q head、维度128、page_size=64；indexed_end=65536，sink=64，window=2048。
历史候选页已由粗筛选出，这个实验不计粗筛，也不计建索引。

| 输入/缓冲区 | 示例shape | 含义 |
|---|---|---|
| Q | `[4096,1,32,128]` FP8 | 原DeepGEMM形状，第二维next_n=1；等价于wrapper把[4096,32,128]扩一维 |
| weights | `[4096,32]` FP32 | 查询头权重 |
| native | `[1088,8448]` uint8 | 仅供准备阶段读取，不传给精筛 |
| native页表 | `[1088]` int32 | 单请求逻辑页映射 |
| lens | `[4096]` int32 | 65537～69632 |
| indexed_end | `[1]` int32 | 65536，实际连续发布终点 |
| selected | `[4096,40]` int32 | 已选中的历史重排物理页，位于fine池前64页中 |
| takes | `[4096]` int32 | 动态历史页预算，允许0～40 |
| members | `[64,64]` int32 | 历史页成员的原序列逻辑位置 |
| fine_pool | `[163,64,1,132]` uint8 | 同一池：64个历史页＋99个预留共享页；有效共享页97个 |
| fine_table | `[4096,108]` int32 | 各Q独立页表，全部编号都相对于同一个fine_pool |
| dg_lens | `[4096,1]` int32 | DeepGEMM实际打分跨度，可包含页边界填充 |
| canonical_lens | `[4096]` int32 | 恢复后的有效候选存储跨度 |
| canonical_scores | `[4096,6912]` FP32 | 恢复到原候选排列后的分数，无效项为-inf |

每页实际是8192字节FP8向量＋256字节FP32 scale；[64,1,132]只是字节容器形状，不代表逐token交错存scale。

## 3. 同一个fine池里怎么共享

```text
fine_pool[0:64]    历史重排页
fine_pool[64:163]  当前chunk共享区域
```

有效公共布局是原逻辑页0和992～1087：

```text
fine_pool[64]     ← 原逻辑页0（sink）
fine_pool[65]     ← 原逻辑页992
...
fine_pool[97]     ← 原逻辑页1024（位置65536～65599）
...
fine_pool[160]    ← 原逻辑页1087
fine_pool[161:163] 预留，不使用
```

Q0的tail从63489开始；Q4095的tail从65536开始：

```text
Q0页表：   [40个已选历史页] [64] [65～97]
Q4095页表：[40个已选历史页] [64] [97～160]
```

两者都可以引用fine_pool[97]，不存在Q0和Q4095各复制一份这个页的要求。

## 4. 为什么要恢复分数排列

Q0需要从63489读，但整页从63488开始。DeepGEMM会按页表正常计算整页，允许这个多余位置暂时有分数。

```text
DeepGEMM Q0原始布局：
[历史候选][sink][63488,63489,...,65536]

normalize输出：
[历史候选][sink][63489,...,65536]
```

`normalize`将逐Q尾部重新映射成紧凑候选编号，并把历史页中与sink/tail重叠、不在实际索引覆盖内、或不满足因果约束的成员设为-inf。
它同时完成边界过滤和分数重排，**只处理分数，不再复制K**。

Q0：原始打分跨度4673，恢复后候选跨度4672。
Q4095：原始和恢复后跨度都是6720，即40×64＋64＋4096。

恢复原候选顺序后，后续TopK不需要理解共享页偏移。最终位置映射使用实际indexed_end，不能恢复成旧的按长度推算桶边界公式。

此实验为了隔离正确性和准备收益，TopK使用两边一致、预分配输出的`torch.topk(k=2048)`，再调用Triton映射。**未声称生产版V1融合TopK已接入**。兼容的是DeepGEMM接口和紧凑候选顺序，生产集成还需把实际发布终点接入原TopK映射/过滤等运行路径。

## 5. 正确性验证

- 真实DeepGEMM，对比逐Q复制与共享页：有效分数位置、分数、TopK分值、因果范围、物理slot。
- 位级相同分数时，要求TopK逻辑位置和物理slot完全相同；浮点比较容差另为rtol=1e-5/atol=2e-5。不能把同分边界的任意不同顺序当成错误。
- 首/中/末Q与独立PyTorch点积参考核对，使用FP8量化后实际数据，不假设量化前后相同。
- sink/window和未建索引区间使用独立集合并集oracle，避免两条路径共享同一个错误公式。
- 非对齐sink17/window2049、短尾、预算变化、历史页和固定候选重叠均覆盖。
- 4K chunk最后Q对逻辑65537注入高分，确认其能进入Top2048，没有被2K窗口错误排除。
- 同一张Graph捕获了准备、页表、动态调度、原DeepGEMM、过滤、TopK和映射；回放更新lens、indexed_end、takes、selected以及native页表。
- 页表迁移测试同时搬迁实际native数据，保持逻辑token语义；不是只改页编号。
- 索引滞后到61440、预算0、空长度回放均通过；容量按最小允许indexed_end预留。

已选择历史页在测试中没有重复，members也不重复。实际调用者仍需保证历史候选页不会重复，或者在候选路由中去重；本实现不额外为任意重复页做集合去重。

## 6. 性能

H20-3e，1K/4K Q、32个Q head、1个K head、40个已选历史页。基线同样修正indexed_end，按GROUP=1/2/4/8调优。
Graph数据使用3次交替测量的中位数，每次do_bench_cudagraph(rep=40ms)。非Graph用CUDA Event覆盖20次Python提交，5组中位数，包含提交间隙。未锁频，存在测量波动。
初始化、预分配、JIT编译、参考验证在计时之外。动态调度生成计入精筛阶段。

### 准备＋页表＋调度＋原DeepGEMM精筛＋过滤恢复

| 历史长度 | Q数 | sink/window | 基线非Graph μs | 共享非Graph μs | 加速 | 基线Graph μs | 共享Graph μs | 加速 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 65536 | 1024 | 64/2048 | 262.04 | 208.43 | 1.26× | 262.53 | 185.70 | 1.41× |
| 65536 | 4096 | 64/2048 | 1355.75 | 834.41 | 1.62× | 1325.04 | 804.99 | 1.65× |
| 65549 | 4096 | 17/2049 | 1349.29 | 828.60 | 1.63× | 1342.50 | 806.90 | 1.66× |

### 再加相同测试用TopK与逻辑/物理位置映射

| Q数 | sink/window | 非Graph加速 | Graph加速 |
|---:|---|---:|---:|
| 1024 | 64/2048 | 1.15× | 1.17× |
| 4096 | 64/2048 | 1.22× | 1.22× |
| 4096 | 17/2049 | 1.21× | 1.22× |

4K案例fine池空间：逐Q基线2145.52 MiB，共享方案1.313 MiB。这里仅比较fine池，未把两边共同的原生缓存和分数工作区算进去。

这些是独立算子/精筛链路的测试，不包含粗筛、建索引、attention或完整模型推理，不能当作端到端V1加速比。原始阶段计时见results.json。

## 7. 运行与集成边界

```bash
bash /home/hadoop-friday-llm/fine_pool_shared_20261021/run.sh
```

依赖PyTorch、Triton、DeepGEMM，脚本保留本机环境路径。没有依赖导入整个SGLang，也没有更改它的源码。

```python
from experiment import Data, Runner

d = Data(history=65536, rows=4096, sink=64, window=2048)
shared = Runner(d, shared=True)
shared.full()
# shared.logical / shared.slots 是Top2048的逻辑位置和原生物理slot
```

源K写入必须在准备前完成，连续索引发布终点必须真实且对查询流可见；共享fine区域在查询消费结束前不能被下一chunk覆盖。
地址和容量预分配；lens、预算和indexed_end作为设备输入改变，不能在回放期间无序修改。最大候选跨度必须包含页边界填充，以及允许的最坏索引滞后。

本实验支持同一请求共享一张原生页表。多请求、多层、多KV head应分别管理共享区域、发布进度和生命周期，不能把不相关请求错误共享。
