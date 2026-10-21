# MLA Index 算子清单与建索引说明（共享精筛版本）

本文只保留 MLA Index 输入下的 V1 核心实现，以及新的共享精筛实现。假设后端已提供符合接口的 Index K＋scale、Index Q 和权重，不包含 GQA 输入适配、非分页 DSA 对照、旧实现及备用路径。原总表保留不变。

**集成状态：建索引和粗筛来自现有 V1；共享精筛是已测试的独立实验，尚未替换原 SGLang。下面是模块清单，不表示它们已作为一条完整生产链路接通。**

## 文件位置

| 简称 | 文件（相对仓库根目录） |
|---|---|
| G | [fused_grouping.py](../../sglang/srt/layers/attention/v1_native/fused_grouping.py) |
| V | [kernels.py](../../sglang/srt/layers/attention/v1_deepseek/kernels.py) |
| C | [v1_query.cuh](../../sglang/kernels/jit/csrc/deepseek_v4/v1_query.cuh) |
| S | 当前目录 [tail_union.py](tail_union.py) |
| E | 当前目录 [kernels.py](kernels.py) |

## 自定义算子

| 阶段 | 算子 | 功能 | 实现 | 文件 |
|---|---|---|---|---|
| 建索引初始化 | `initialize` | 按原生页表读 K、反量化、归一化，初始化成员排列 | Triton | V |
| 分组打分 | `_dots` | 成员与锚点点积，以及两个锚点的相似度差 | Triton | G |
| 选择锚点 | `_anchor` | 选择点积最小的成员作为锚点 | Triton | G |
| 分组排序 | `_sort` | 分数降序、同分稳定排序，更新排列供均衡二分 | Triton | G |
| 组装索引 | `assemble` | 生成重排 K、中心、scale 和成员逻辑位置 | Triton | V |
| 页指纹 | `fingerprint` | 计算页数据及成员映射的指纹 | Triton | V |
| 页查重 | `match_pages` | 指纹筛选后逐字节及逐成员验证，寻找可复用页 | Triton | V |
| 页分配 | `allocate` | 复用或分配页，更新引用计数、桶页表和新页标记 | Triton | V |
| 页发布 | `publish` | 写入重排页、成员、指纹及桶对应的中心 | Triton | V |
| 页回收 | `release` | 减引用计数，将零引用页归还空闲栈 | Triton | V |
| 粗筛融合 | `fused.route / v1_route_main` | 每页 8 个中心取 max、页面 TopK、按配置写候选物理页表 | CUDA C++ | C |
| 公共页复制 | `stage_union` | 根据实际已发布索引终点，将 chunk 公共 sink＋tail 页复制一次到精筛池后缀 | Triton | S |
| 精筛页表与长度 | `tables` | 组织历史候选页和共享页引用，生成打分长度及紧凑候选长度 | Triton | E |
| 精筛分数整理 | `normalize` | 过滤边界、因果和历史重复覆盖，恢复紧凑候选分数布局 | Triton | E |
| 结果映射 | `map_topk` | 将最终候选编号还原为逻辑位置，再转换为原生物理 slot | Triton | E |

这里不再列逐 Q 尾部复制、owner 共享、旧页表构造、备用粗筛/精筛及独立旧输出映射等路径。GQA 的 K/V gather 和连续缓存适配也不属于这份 MLA Index 清单。

## 复用的计算组件

| 组件 | 功能 | 来源 |
|---|---|---|
| `fp8_paged_mqa_logits` | 中心打分和候选 Index K 精筛；共享实验实际调用它进行精筛 | DeepGEMM |
| `get_paged_mqa_logits_metadata` | 根据本次长度生成分页打分调度元数据 | DeepGEMM |
| `torch.topk` | 当前共享实验中最终 Top2048；两种对照共用同一实现 | PyTorch |
| `topk_v2.cuh`、CUB、`tl.sort` | 融合路由或分组排序复用的基础操作 | SGLang/CUDA/Triton |
| 原生 MLA attention 后端 | 根据最终索引读取原生 MLA KV，计算 attention | 模型后端 |

**新的共享实验目前使用 `torch.topk + map_topk`，未接入 `fused.topk`。** 因此本表没有把已有的 `fused.topk` 当作共享实验正在执行的步骤；它仍保留在原源码和原总表中。不能把实验中的 TopK 描述为已融合。

## 调用流程

建索引：

```text
上传原生桶页表和元数据
→ initialize
→ 多级 _dots / _anchor / _sort
→ assemble
→ fingerprint → match_pages → allocate → publish
→ 发布请求页表和完成事件
```

查询的目标组织方式：

```text
原生 Index Q、权重、长度、预算和已发布索引边界
→ 中心打分（DeepGEMM）
→ fused.route 选择历史候选页
→ stage_union 将公共 sink＋tail 复制一次到同一精筛池
→ tables 生成各 Q 页表和长度
→ 动态生成 DeepGEMM 调度元数据
→ DeepGEMM 精筛
→ normalize 过滤并恢复候选顺序
→ torch.topk
→ map_topk
→ 原生 MLA attention
```

独立实验实际计时从公共页准备开始，历史候选页由测试输入提供；不包含粗筛、建索引和 MLA attention。上述完整组织方式不是已经测过的端到端链路。

### 动态输入和正确性

- 长度、预算、页表及 `INDEXED_END` 是动态数据；Graph 捕获的容量必须覆盖回放规模。
- `INDEXED_END` 表示实际已发布的连续索引前缀终点，不能仅凭当前 Q 长度推算。
- 未建索引但因果可见的 token 进入共享候选范围，再参与最终 TopK；不意味着整个 chunk 无条件加入最终 attention。
- 原生页先复制到同一个精筛池，各 Q 使用自己的页表引用公共副本。精筛仍只接收一个池基址，无需直接读取两个池。
- 当前实验要求历史候选页和成员唯一，不自动修复任意重复输入。
- stream/event、Indexer、CUDA Graph 的捕获与回放属于运行时管理，不是独立 GPU 算子。

## 已测收益的范围

64K 历史、4096 Q、40 个历史候选页、sink=64、window=2048，CUDA Graph：

| 范围 | 原实现 | 共享实现 | 加速比 |
|---|---:|---:|---:|
| 准备＋页表＋动态调度＋DeepGEMM 精筛＋整理 | 1.325 ms | 0.805 ms | 1.65× |
| 上述流程再加 TopK 和输出映射 | 2.245 ms | 1.840 ms | 1.22× |

这些是共享精筛实验结果，不是整条 V1 查询或 attention 的加速比。详见 [实验说明](README.md) 和 [结果数据](results.json)。

## 建索引建索引为什么这么快，之前借鉴ClusterKV做了哪些改进

借鉴的是“把K分组、用中心做粗筛、把成员重排存储”的思路。当前实现的速度主要来自下面几处设计；构建与其他计算overlap，则进一步减少推理关键路径上的等待。

这里按当前本地代码解释，不把下面每一点都宣称为首次提出，也没有据此给出未经对照实验验证的“相对ClusterKV加速倍数”。

| 设计/改进 | 具体做法 | 减少的开销 |
|---|---|---|
| 固定小桶增量构建 | 每来一个完整1K桶，只处理这1K个K | 不随历史增长而重建全部历史；单桶分组规模固定 |
| 平衡二分分组 | 每组选择两个锚点，按相似度差排序，再平分 | 当前算法不进行反复更新质心的K-means，不计算全部K两两相似度矩阵 |
| 归一化一次 | 初始化时生成归一化K，后续各级复用 | 不重复计算向量范数 |
| 主要移动成员编号 | 分组时排序整数perm，而不是反复重排完整K | 减少128维向量反复搬运 |
| 最后统一重排 | 分组结束后assemble生成重排K、中心和成员映射 | 把真正的K重排集中到最后 |
| GPU上分组 | dots、anchor、sort全部在GPU执行 | 中间结果不用逐组拿回CPU再决策 |
| 工作区预分配 | unit、两个perm、scores、anchor重复使用 | 减少分配和临时tensor开销 |
| 跨流overlap | 当前chunk构建，后续chunk使用 | 在依赖允许时与其他计算并行，减少使用索引前的残余等待 |

固定桶使分组工作量不随全历史增长，但整条构建还包含页指纹、查重、分配和发布等步骤，其开销不应一概视为与池容量无关。

### 1K桶怎么分组

```text
1024个K
    ↓
2组，每组512个
    ↓
4组，每组256个
    ↓
8组，每组128个
    ↓
16组，每组64个
    ↓
32组，每组32个
    ↓
64组，每组16个
    ↓
128组，每组8个
```

总共7级。每组的分裂过程：

```text
取一个起始成员
    ↓
找与它点积最小的成员b
    ↓
找与b点积最小的成员a
    ↓
计算各成员对a、b的相似度差
    ↓
按差分数排序
    ↓
前半和后半成为两个子组
```

这里使用归一化向量；对非零向量，点积对应余弦相似度。全零向量按带下限归一化处理。

最后：

```text
每8个成员 → 1个中心
每8个小组 → 1个64-token重排页
1K桶 → 128个中心、16个重排页
```

### 为什么“只排序ID”有用

```text
unit：归一化后的K，保留原有存储位置
perm：成员编号排列，反复排序它

点积时：通过perm找到unit中的成员
排序时：更新perm，不反复搬128维向量
分组完成：assemble依据最终perm写出重排K
```

`perm[0]` 和 `perm[1]` 交替作为排序输入输出，避免覆盖尚未读完的排列。下一层通过减半组大小，把排序后的前后两半解释成两个子组，不需要额外的“二分复制”kernel。

归一化只服务于分组；最终重排缓存保存的是原有K＋scale，不是把所有真实K永久替换成归一化K。

### 需要说准确的几个地方

1. **不是一个kernel建完索引。**当前1K分组每级3次dots、2次anchor、1次sort。7级共42次，加初始化共43次kernel启动；还不含assemble、页管理和状态登记。
2. **overlap不等于构建免费。**它隐藏等待，但构建仍消耗GPU资源，也可能与前台计算争用资源。需区分构建GPU耗时、使用前残余等待以及整个chunk耗时。
3. **当前页查重发生在分组之后。**页复用能减少后续分配/写入，但不能解释成这条1K路径自动跳过分组。
4. **没有在这里测相对ClusterKV的构建加速倍数。**要定量比较，必须同输入、同分组目标和相同计时范围；查询加速比不能代替构建加速比。
5. **归一化与分组的浮点实现可能改变边界结果。**FP8反量化输入与原始BF16输入不同，归约顺序也可能不同，不能仅凭算法步骤相似宣称逐位等价。

主要收益可以归纳为：固定小桶增量构建、固定层数的平衡分组、只排序成员编号、GPU实现与预分配，再用overlap减少关键路径等待。


## 相关文档

- [本地实现位置](LOCAL_IMPLEMENTATION_PATHS.md)
- [共享缓存六个问题](RECENT_SIX_QA.md)
- [原完整总表（含旧路径）](KERNEL_OVERVIEW_AND_BUILD_SPEED.md)
