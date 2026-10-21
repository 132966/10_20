# 精筛实现、本地路径与依赖

本文列出同一精筛池共享尾部实验的本地实现位置。实验没有修改原SGLang项目或DeepGEMM kernel；精筛只读取一个fine池，原生池仅作为准备阶段的复制源。

## 1. 本地实验目录

```text
/home/hadoop-friday-llm/fine_pool_shared_20261021/
```

下表中的相对链接可以在GitHub直接打开；“本地路径”用于本机定位。

| 环节 | 文件和入口 | 本地路径 | 功能 |
|---|---|---|---|
| 输入数据 | [experiment.py](experiment.py)，`Data` | `/home/hadoop-friday-llm/fine_pool_shared_20261021/experiment.py` | 准备Q、权重、原生K池、原生页表、可见长度、实际发布终点、历史候选页及成员映射 |
| 工作区 | [experiment.py](experiment.py)，`Runner.__init__()` | 同上 | 分配单一fine池、各Q页表、长度、分数和TopK输出缓冲区 |
| 准备入口 | [experiment.py](experiment.py)，`Runner.prepare()`，第73行 | 同上 | 执行公共复制或逐Q基线复制，然后生成精筛页表和长度 |
| 公共页复制 | [tail_union.py](tail_union.py)，`stage_union()`，第17行 | `/home/hadoop-friday-llm/fine_pool_shared_20261021/tail_union.py` | 将sink＋tail公共页复制一次到fine池的预留区域，同时生成范围信息 |
| 逐Q基线 | [baseline_published.py](baseline_published.py)，`stage_tail_shared()` | `/home/hadoop-friday-llm/fine_pool_shared_20261021/baseline_published.py` | 每个Q独立复制候选，同样使用真实indexed_end，供公平对比 |
| 页表和长度 | [kernels.py](kernels.py)，`tables()`，第14行 | `/home/hadoop-friday-llm/fine_pool_shared_20261021/kernels.py` | 写fine_table、DeepGEMM打分长度和紧凑候选长度；不同Q可引用同一共享页 |
| 精筛打分入口 | [experiment.py](experiment.py)，`Runner.fine_score()`，第79行 | `/home/hadoop-friday-llm/fine_pool_shared_20261021/experiment.py` | 动态生成调度信息，调用原DeepGEMM分页精筛，再恢复分数布局 |
| 过滤与分数恢复 | [kernels.py](kernels.py)，`normalize()`，第39行 | `/home/hadoop-friday-llm/fine_pool_shared_20261021/kernels.py` | 过滤无效边界、因果/重叠成员，并恢复原来的紧凑候选分数顺序 |
| TopK调用 | [experiment.py](experiment.py)，`Runner.select()`，第85行 | `/home/hadoop-friday-llm/fine_pool_shared_20261021/experiment.py` | 调用两条路径共用的预分配PyTorch TopK；不是生产V1融合TopK实现 |
| 输出位置映射 | [kernels.py](kernels.py)，`map_topk()`，第62行 | `/home/hadoop-friday-llm/fine_pool_shared_20261021/kernels.py` | 候选编号→原序列逻辑位置→原生物理slot |
| 正确性及性能测试 | [test_and_bench.py](test_and_bench.py) | `/home/hadoop-friday-llm/fine_pool_shared_20261021/test_and_bench.py` | 对照基线、PyTorch参考、TopK与映射、动态CUDA Graph和性能测试 |
| 启动脚本 | [run.sh](run.sh) | `/home/hadoop-friday-llm/fine_pool_shared_20261021/run.sh` | 加载本机环境并执行测试 |

行号对应本文编写时的版本，后续代码变化时优先按函数名搜索。

## 2. 真正执行精筛点积的CUDA kernel

Python调用入口是：

```python
deep_gemm.fp8_paged_mqa_logits(...)
```

H20对应SM90实现，本机文件：

```text
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/sglang_sparse/sglangenv/lib/python3.12/site-packages/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh
```

对应的调度实现：

```text
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/sglang_sparse/sglangenv/lib/python3.12/site-packages/deep_gemm/include/deep_gemm/scheduler/sm90_paged_mqa_logits.cuh
```

这些属于本机安装的DeepGEMM依赖，**没有复制到本次GitHub实验目录，也没有修改**。本次自定义的准备、页表、过滤和映射kernel是Triton；真正的点积计算复用DeepGEMM CUDA实现。

可在本机定位：

```bash
rg -n 'def prepare|def fine_score|def select' /home/hadoop-friday-llm/fine_pool_shared_20261021/experiment.py
rg -n '^def tables|^def normalize|^def map_topk' /home/hadoop-friday-llm/fine_pool_shared_20261021/kernels.py
```

## 3. 精筛要用到哪些输入

| 输入 | 本例shape | 在哪准备 |
|---|---|---|
| Q | `[4096,1,32,128]`，FP8 | `Data.q`；next_n=1、32个Q head |
| 查询头权重 | `[4096,32]`，FP32 | `Data.weights` |
| 单一精筛K池 | `[163,64,1,132]`，uint8容器 | `Runner.fine4`，对应同一块`Runner.fine`内存 |
| 每Q候选页表 | `[4096,108]`，int32 | `tables()`写入`Runner.table` |
| 每Q打分长度 | `[4096,1]`，int32 | `tables()`写入`Runner.dglens` |
| DeepGEMM调度元数据 | SM90为`[SM数量+1,2]` | `get_paged_mqa_logits_metadata()`生成 |
| 最大打分宽度 | 本例6912 | 初始化时按预算、尾部容量和边界余量预留 |

示例条件为64K历史、4K Q、40个历史候选页、64 sink、2048 window、indexed_end=65536。

准备阶段另外读取原生K、原生页表、lens和indexed_end；过滤和映射另外使用takes、selected、members。它们不是要求DeepGEMM额外接收第二个K池。

fine池布局：

```text
fine[0:64]    历史重排页
fine[64:163]  共享sink＋tail预留区域，当前97页有效
```

## 4. 整个调用链

```text
Data                         准备测试输入
    ↓
Runner                       分配fine池、页表、分数和输出缓冲区
    ↓
Runner.full()
    → Runner.run_fine()
        → Runner.prepare()
            → PreparedStage / stage_union
                从原生池复制公共页到fine池，只复制一次
            → tables
                各Q写自己的页表，引用同一fine池中的共享页
                生成打分长度和紧凑候选长度
        → Runner.fine_score()
            → get_paged_mqa_logits_metadata
                根据本次GPU长度生成调度信息
            → fp8_paged_mqa_logits
                原DeepGEMM，读取一个fine池，按各Q页表打分
            → normalize
                过滤边界、未来/重叠成员，恢复紧凑候选顺序
    → Runner.select()
        → torch.topk(k=2048)
            两种方案共用的测试TopK
        → map_topk
            还原逻辑位置并映射成原生物理slot
```

推荐阅读顺序：先看 `Runner.prepare()` 和 `Runner.fine_score()`；再看 `tables()`、`normalize()`；需要研究矩阵计算细节时再读DeepGEMM的SM90 CUDA文件。

## 5. 说明与结果

- [实验说明和性能结果](README.md)
- [最近六次技术问答](RECENT_SIX_QA.md)
- [原始测量数据](results.json)
- [来源记录](PROVENANCE.json)

本实验验证了真实DeepGEMM接口和动态Graph，但尚未部署到原SGLang模型运行路径。测试不包含粗筛、建索引或attention，不能将本实验的精筛加速比当作完整模型加速比。
