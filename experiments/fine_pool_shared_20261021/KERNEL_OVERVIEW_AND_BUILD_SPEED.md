# 算子功能总表与建索引速度说明

本文整理最近两个技术问题的回复：

1. 一共有哪些算子，列一下大表。
2. 建索引为什么这么快，之前借鉴ClusterKV做了哪些改进。

包含旧路径、备用路径、融合路径及新增独立实验；不是一次查询全部执行。新增共享精筛仍是独立实验，未替换原SGLang项目。

## 问题一：一共有哪些算子

文件简称：

| 简称 | 文件 |
|---|---|
| P | 仓库根目录 `qwen_gqa_probe_kernels.py` |
| G | `sglang/srt/layers/attention/v1_native/fused_grouping.py` |
| N | `sglang/srt/layers/attention/v1_native/kernels.py` |
| V | `sglang/srt/layers/attention/v1_deepseek/kernels.py` |
| C | `sglang/kernels/jit/csrc/deepseek_v4/v1_query.cuh` |
| S | 当前实验 [tail_union.py](tail_union.py) |
| E | 当前实验 [kernels.py](kernels.py) |

P/G/N/V/C路径相对于GitHub仓库根目录；S/E位于 `experiments/fine_pool_shared_20261021/`。

| 阶段 | 算子 | 功能 | 实现 | 文件 |
|---|---|---|---|---|
| 输入准备 | `pack_new` | 新增GQA K量化、打包为FP8 K＋scale | Triton | P |
| 输入准备 | `pack_q` | 准备查询Q、打分权重，补齐查询头 | Triton | P |
| DSA对照 | `gather_index` | 从分页缓存收集连续Index K＋scale，供非分页打分 | Triton | P |
| 建索引初始化 | `initialize` | 按原生页表读K、反量化，生成归一化分组数据并初始化排列 | Triton | V |
| 旧分组初始化 | `_init` | 旧4K入口的K读取、归一化及排列初始化 | Triton | G |
| 分组打分 | `_dots` | 成员与锚点的点积，以及分组差分分数 | Triton | G |
| 选择锚点 | `_anchor` | 根据最小点积选择锚点成员 | Triton | G |
| 分组排序 | `_sort` | 按分数稳定降序排序，为逐级均衡拆分生成排列 | Triton | G |
| 分组辅助 | `_publish_exact` | 复制参考分组结果，供对应分组模式使用 | Triton | G |
| 组装索引 | `assemble` | 生成临时重排K、中心、scale及成员逻辑位置映射 | Triton | V |
| 页指纹 | `fingerprint` | 为临时页的数据及成员映射计算指纹 | Triton | V |
| 页查重 | `match_pages` | 先匹配指纹，再核对完整页数据和成员映射，寻找可复用页 | Triton | V |
| 页分配 | `allocate` | 复用或分配物理页，更新引用计数、桶页表和新页标记 | Triton | V |
| 页发布 | `publish` | 发布重排页、成员映射、指纹及桶对应的中心 | Triton | V |
| 页回收 | `release` | 引用计数减一，将零引用页放回空闲页栈 | Triton | V |
| 查询准备 | `query_lengths` | 计算候选长度和选页数量 | Triton | V |
| 查询准备 | `query_metadata` | 准备粗筛/精筛长度、预算相关信息和尾部共享owner | Triton | V |
| 粗筛备用 | `coarse` | Q对中心打分、汇总查询头、每页8个中心取最大值 | Triton | V |
| 页面分数归约 | `reduce_centers` | 已有中心分数每8个取max，得到页面分数 | Triton | V |
| 粗筛融合 | `fused.route / v1_route_main` | 每页8中心取max＋页面TopK＋可选物理页表写入 | CUDA C++ | C |
| 精筛页表 | `fine_table` | 生成各Q的精筛物理页表 | Triton | V |
| 共享页表旧实现 | `fine_table_shared` | 根据owner引用相应尾部缓存 | Triton | V |
| 尾部准备旧实现 | `stage_tail` | 为每个Q复制sink和tail的K＋scale | Triton | V |
| 尾部共享旧实现 | `stage_tail_shared` | 仅owner复制，其余满足共享条件的Q复用副本 | Triton | V |
| 精筛备用 | `fine` | 候选K打分并处理有效范围 | Triton | V |
| 精筛过滤 | `mask_fine` | 屏蔽无效、不可见及与固定候选重叠的成员 | Triton | V |
| 最终选择融合 | `fused.topk / v1_topk_main` | 最终TopK、过滤、成员映射，以及按配置执行物理转换或展开 | CUDA C++ | C |
| 输出映射 | `output_ids` | 将候选编号还原为逻辑位置或原生物理位置 | Triton | V |
| 输出展开 | `expand_logical` | 按对应压缩映射规则展开逻辑位置 | Triton | V |
| K/V收集 | `_gather_slots` | 逻辑位置→物理slot，收集K/V并写入attention所需布局 | Triton | N |
| 旧路径打包 | `_pack_slots` | 收集K、量化打包并记录逻辑成员映射 | Triton | N |
| 旧路径过滤 | `_mask_scores` | 旧查询路径的可见性及重叠过滤 | Triton | N |
| 连续缓存追加 | `_append_flat` | 新增K量化并追加到连续缓存 | Triton | N |
| 新增公共页复制 | `stage_union` | 使用真实indexed_end，将整个chunk的公共sink＋tail页只复制一次到同一精筛池 | Triton | S |
| 新增精筛页表 | `tables` | 生成各Q共享页引用、DeepGEMM打分长度及紧凑候选长度 | Triton | E |
| 新增过滤与恢复融合 | `normalize` | 过滤边界、因果和重叠成员，同时恢复原候选分数顺序 | Triton | E |
| 新增结果映射 | `map_topk` | TopK候选编号→逻辑位置→原生物理slot | Triton | E |

### 复用组件，不算新写的算子

| 组件 | 功能 | 来源 |
|---|---|---|
| `fp8_paged_mqa_logits` | 分页中心/候选K打分，当前实验的实际精筛点积 | DeepGEMM CUDA |
| 非分页MQA logits | 连续K布局下打分 | DeepGEMM CUDA |
| `topk_v2.cuh` | V1融合代码复用的基础TopK | SGLang历史官方实现 |
| CUB、`tl.sort` | 排序等基础操作 | CUDA/Triton |
| FA3/模型attention后端 | 对最终选出的K/V计算attention | 原attention后端 |
| 当前实验 `torch.topk` | 两条路径共用的预分配TopK验证及计时组件 | PyTorch |

当前独立共享精筛链路：

```text
stage_union：公共K只复制一次到fine池
    ↓
tables：各Q页表指向同一fine池中的候选页
    ↓
原DeepGEMM：精筛打分
    ↓
normalize：过滤＋恢复候选分数顺序
    ↓
测试用TopK
    ↓
map_topk：逻辑位置＋原生物理slot
```

原始 `stage_tail_shared` 的owner共享与新增chunk公共范围共享不是同一个机制；新增路径使用实际发布终点，旧源码未因此自动更新。

CPU的 `submit_build()`、Indexer状态管理、stream/event overlap和CUDA Graph捕获/回放属于运行时调度，不算GPU kernel。

本地实现入口见 [LOCAL_IMPLEMENTATION_PATHS.md](LOCAL_IMPLEMENTATION_PATHS.md)。

## 问题二：建索引为什么这么快，之前借鉴ClusterKV做了哪些改进

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
3. **当前页查重发生在分组之后。**页复用能减少后续分配/写入，但不能解释成这条1K路径自动跳过分组。旧4K路径的整桶命中跳过机制是另一个分支。
4. **没有在这里测相对ClusterKV的构建加速倍数。**要定量比较，必须同输入、同分组目标和相同计时范围；查询加速比不能代替构建加速比。
5. **归一化与分组的浮点实现可能改变边界结果。**FP8反量化输入与原始BF16输入不同，归约顺序也可能不同，不能仅凭算法步骤相似宣称逐位等价。

主要收益可以归纳为：固定小桶增量构建、固定层数的平衡分组、只排序成员编号、GPU实现与预分配，再用overlap减少关键路径等待。

## 相关入口

- [本次共享精筛实验及结果](README.md)
- [本地精筛实现路径](LOCAL_IMPLEMENTATION_PATHS.md)
- [最近六次共享缓存技术问答](RECENT_SIX_QA.md)
- 分组实现：仓库 `sglang/srt/layers/attention/v1_native/fused_grouping.py`
- 1K建索引主kernel：仓库 `sglang/srt/layers/attention/v1_deepseek/kernels.py`
