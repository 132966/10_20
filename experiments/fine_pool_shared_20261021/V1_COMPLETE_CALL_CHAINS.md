# V1 完整调用链：请求执行、建索引、查询、overlap 与回收

本文汇总此前整理的较长调用链，按“谁调用谁、每一步做什么、结果交给谁”展开。默认用 **单个索引键头、128 维向量、1K 桶、64 成员页、32 个查询头**解释；具体接口可能带额外的单例维度。

## 0. 先区分三套路径

| 路径 | 本文如何描述 |
|---|---|
| 1K 桶运行时及后台构建 | 用 `submit_build`、`Indexer.prepare`、`Build.run/compute` 等实际名字说明 |
| 共享精筛实验 | 用 `stage_union → tables → 精筛打分 → normalize → TopK → map_topk` 说明；它与完整后端的接入关系需要区分 |
| sink＋tail 紧凑拼接 | 最近讨论的目标布局，目前只改了说明文档，不能当作已经改好的 kernel |

下面的总链串起系统的功能依赖，不表示仓库中每个入口都已经实现相同的整条调用。尤其是旧 `Indexer.query()` 会先调用 `_prepare()` 补齐索引，不能把它直接等同于只消费此前快照的后台 overlap 路径。

## 1. 单请求分块推理的总链

```text
请求进入 SGLang
  → 确定当前分块及各查询的因果可见范围
  → 执行当前层模型计算，得到 Q、索引 K 和 attention 所需 K/V
  → 写入原生缓存，建立源数据就绪依赖
  → 获取当前请求、当前层、当前索引键头的 V1 状态

  → 选择本次 attention 路径
      ├─ 全量路径
      │    → 对全部因果可见 K/V 执行 attention
      │    → 若启用增量维护，仍提交新增完整桶构建
      │
      └─ V1 稀疏路径
           → 获取此前索引快照，满足构建完成依赖
           → 准备查询长度、预算、页表及工作区
           → 中心粗筛
           → 真实成员精筛
           → 最终 TopK
           → 恢复原逻辑位置
           → 按需展开，并经当前请求映射读取原生 K/V
           → 执行 attention
           → 提交新增完整桶构建

  → 继续后续层
  → 下一分块到达同一层时，使用此前已经建立好的索引
  → 请求完成或取消，解除对应状态与资源持有
```

当前分块的新键无需先建好索引才能参加选择：可见且尚未索引的部分由固定后缀覆盖。固定候选进入精筛竞争，不等于它们无条件进入最终 attention。

decode 是否稀疏由后端策略决定。此前讨论的测试配置采用全量 decode，不应把预填充的稀疏路径套到 decode 上。

## 2. 建索引与插索引完整调用链

```text
当前分块、当前层 attention 完成
  → 后端 submit_build(...)
      → 取得本层、本索引键头的 Indexer
      → 准备源数据 ready 事件和初始化完成依赖
      → 切换到独立 build_stream
      → build_stream 等待源数据 ready、初始化等依赖
      → 准备当前请求的原生源页表

      → Indexer.prepare(owner, native_table, length)
          → 获取 CPU 锁，保护请求容器和桶记录等主机状态
          → Indexer._begin()
              → 检查 Indexer 状态
              → 当前 GPU 流等待 Indexer 上一次操作完成事件

          → Indexer._prepare(owner, native_table, length)
              → 校验长度、页表类型、设备和容量
              → 查找或创建该请求的 State
              → 若长度回退，按现有入口释放旧状态再建立新状态
              → complete = length // 1024
              → 与 state.leases 中已有桶数比较，找新增完整桶
              → 未满 1K 的尾部暂不建立完整桶

              → 对每个新增桶：
                  → 领取桶记录 lease
                  → Build.run(native_table, start, lease)
                      → 等待前一次 Build 工作区使用完成
                      → 上传/复制本桶 16 项源页表
                      → 设置 meta = [桶逻辑起点, lease, active]
                      → execute()
                          ├─ eager：Build.compute()
                          └─ 对应子图已捕获：回放构建 CUDA Graph

                      → Build.compute() 的 GPU 序列
                          → initialize
                          → 分组：_dots / _anchor / _sort
                            或相应优化实现中的融合路径
                          → assemble
                          → fingerprint
                          → match_pages
                          → allocate
                          → publish

                      → 记录 Build 工作区使用完成事件
                      → 维护源页表张量的跨流生命周期

                  → state.leases 记录桶的 lease
                  → 将 tables[lease] 接入 state.table
                  → 将中心页引用接入 state.center_table

              → 更新请求已提交长度

          → Indexer._finish()
              → 在当前流记录新的 Indexer.done
          → 释放 CPU 锁

      → 后端记录 pending[gid] 完成事件
      → 后续同层、同索引键头查询建立该事件的消费依赖
```

“建索引”主要是分组、重排和中心生成；“插索引”主要是查重、分配、发布及请求页表更新。它们属于同一条 GPU 流程，不需要把分组结果读回 CPU 再插入。

CPU 锁和 GPU 事件不是一回事：锁保护主机状态，事件约束设备读写顺序。CPU 更新了 `state.length` 或 `state.leases`，不代表 GPU 已经写完对应数据。

### 2.1 Build.compute 内部各步的作用

```text
initialize
  → 根据本桶源页表读取真实 K 和 scale
  → 反量化得到原空间向量
  → 归一化写入 UNIT，供方向分组使用
  → PERM 初始化为 [0,1,…,1023]
  → 根据 active 设置 FLAG

分组
  → 根据单位方向选择锚点、计算差分、排序、均衡二分
  → 更新成员排列，不逐轮搬动整个高维 K

assemble
  → 按最终排列读取真实源 K 和 scale
  → 写出临时重排 K
  → 每 8 个成员生成一个原空间中心
  → 写出每个重排槽位对应的原逻辑位置

fingerprint
  → 对临时页的数据和成员映射计算指纹

match_pages
  → 从存活的重排物理页中筛选指纹相同的页
  → 继续比较完整 K＋scale 字节和成员映射
  → 完全相同则返回已有物理页编号，否则返回 -1

allocate
  → 已匹配页复用
  → 未匹配页从预分配空闲栈领取物理页
  → 更新引用计数
  → 写桶记录页表 tables[lease]
  → 标记哪些页需要复制新内容

publish
  → 向新分配页写 K＋scale、members 和指纹
  → 已复用页不重复写这些内容
  → 写当前桶记录对应的中心数据和 scale

请求页表更新及完成事件
  → 建立请求到这些页的引用
  → 保证后续消费者在相关写入完成后才读取
```

归一化只用于分组；重排池里的真实 K 保留原存储表示。中心是原空间均值，不是单位方向的均值。

### 2.2 一个新增 1K 桶的例子

```text
已有索引范围：[0,65536)
新增源数据后：长度 66560
桶大小：1024

→ 新增完整桶编号：64（从 0 开始）
→ 桶起点：65536
→ 桶成员范围：[65536,66560)
→ 读取源页表第 [1024,1040) 项，共 16 项
→ 形成 128 个叶子组，每组 8 个成员
→ 产生 16 个成员页、128 个中心
→ 中心打包为 2 个中心存储页
→ 更新 state.table[1024:1040]
→ 更新 state.center_table[128:130]
```

一页为 8448 字节：前 8192 字节存 64×128 个 FP8 K，后 256 字节存 64 个 FP32 scale。中心使用相同的页布局。成员映射单独保存。

## 3. 桶内分组的完整链

下面是单轮划分的算法顺序，重复处理组大小 `1024、512、256、128、64、32、16`，最后得到容量为 8 的叶子组。

```text
当前组的有序成员列表 P
  → 取当前组首成员作为起始参考
  → _dots：全部成员与起始参考做点积
  → _anchor：选最小点积成员作为 b
  → _dots：全部成员与 b 做点积
  → _anchor：选最小点积成员作为 a
  → _dots：计算 dot(member,a) - dot(member,b)
  → _sort：按差分降序，保持同分稳定顺序
  → 写到输出排列 O
  → 按容量边界将排序后列表分为两组
  → 交换 P/O 的输入输出角色
  → 处理下一层更小的组
```

`P/O` 是两个成员编号数组，不是两份高维 K。使用两个数组避免输出覆盖尚未读完的输入。

优化分组实验保留上述依赖，但改变启动组织：

```text
大组：
起点 dots
  → anchor＋下一阶段 dots
  → anchor＋差分 dots
  → sort

小组：
一个 program 内完成
起点 dots → anchor → dots → anchor → 差分 dots → sort
```

大组中，每个处理局部成员的 program 可以重复选取相同锚点，再计算自己负责的点积；不同阶段仍需正确排序。小组融合利用单个 program 内的数据复用。优化实验中的融合配置不能自动当成旧 `Build.compute()` 已采用的实现。

## 4. 查询完整功能链

### 4.1 消费先前快照的查询路径

```text
当前层 Q、原生数据准备完成
  → previous_snapshot(...) 或对应快照获取逻辑
      → 找到此前已提交索引的完成事件
      → 查询流建立消费依赖
      → 取得与中心、成员页表一致的可读状态

  → 准备本次查询
      → 查询 Q 与每个查询头的打分权重
      → 每个 Q 的因果可见长度 n
      → 已就绪索引终点 E
      → 当前长度段/层的召回页预算 P
      → 请求历史页表、中心页表和原生源页表
      → 取得容量足够的查询工作区

  → 确定固定候选与可索引历史
      → sink 固定开头
      → 滑动窗口及尚未索引的可见后缀
      → 历史候选排除与固定范围重叠的成员

  → 准备粗筛长度和 GPU 调度信息
  → 分页中心打分
      → 对查询与中心执行索引打分
  → fused.route / 对应共享布局适配 route
      → 每个成员页对应的 8 个中心分数取 max
      → 全局选择预算内的历史页
      → 将请求页序号转换为重排物理页编号
      → 写出候选历史页表

  → 准备固定候选缓存
  → tables：组织历史候选页与固定候选引用
      → 生成精筛页表
      → 生成打分长度和紧凑候选长度

  → 根据精筛长度生成 GPU 调度信息
  → 分页真实成员打分
      → 使用同一精筛池基址和逐 Q 页表
      → 计算候选真实 K 的索引分数

  → normalize
      → 按当前布局整理候选分数
      → 过滤非法历史、因果越界及与固定范围重复覆盖的成员
  → TopK
      → 在有效候选中选最终预算 K
  → map_topk
      → 历史候选经候选页表和 members 恢复原逻辑位置
      → 固定候选按对应布局和逻辑区间恢复原位置
      → 表示为原序列逻辑页及页内偏移
        （按接口需要继续转换为原生物理 slot）

  → 按接口需要展开位置
  → 通过当前请求的映射读取原生 attention K/V
  → 在最终选中的位置上统一执行 attention softmax
  → 本层 attention 完成后提交新增完整桶构建
```

中心分数和精筛索引分数不是 attention softmax 概率。`normalize` 也不是 softmax。

`route` 的输出要区分版本：旧接口可能返回请求页序号，共享实验适配输出的是物理页编号。只有完成相应转换，才能直接交给要求物理页号的 `tables`。

### 4.2 旧 Indexer.query 包装入口的实际链

```text
Indexer.query(...)
  → 校验 Q、权重、原生页表、长度和请求标识
  → 加锁并调用 _begin()
  → 按请求找本批最大可见长度
  → 对每个请求调用 _prepare() 补齐索引
  → 按查询容量档位分批
  → 获取或创建 Query 工作区/子图
  → Query.run(...)
  → 返回位置结果
  → _finish() 记录完成事件
```

这个入口包含查询前补齐构建。它与第 4.1 节“只消费前一阶段快照”的调度方式不同；讨论 overlap 时需要指明使用哪个入口。

### 4.3 共享精筛实验中的实际函数链

```text
Runner.full()
  → Runner.run_fine()
      → Runner.prepare()
          → 共享路径调用 stage()，执行 stage_union 等公共缓存准备
          → tables(...)
      → Runner.fine_score()
          → get_paged_mqa_logits_metadata(...)
          → fp8_paged_mqa_logits(...)
          → normalize(...)
  → Runner.select()
      → torch.topk(...)
      → map_topk(...)
```

该实验以准备好的历史候选页作为输入；粗筛 route 有单独的适配和测试。不能把这段 `Runner.full()` 说成已经执行了完整模型、全部粗筛和 attention。旧 `fused.topk` 也不能在未核对布局的情况下直接替换这里的三步后处理。

## 5. 固定候选准备：已有共享页与紧凑拼接目标

### 5.1 已有共享页方案

```text
逐 Q 可见长度、sink/window、实际就绪终点 E
  → 算出每个 Q 的固定范围
  → 对同一请求取所需源页的公共并集
  → stage_union
      → 将每个公共源页的 K 和 scale 复制一次
      → 放到精筛池公共后缀
      → 生成公共布局描述
  → tables
      → 每个 Q 引用自己需要的公共页
      → 生成实际打分跨度和紧凑长度
  → 精筛打分
  → normalize 去掉布局填充并过滤非法成员
```

共享的是 K 数据副本，不是所有 Q 的分数或因果边界。每个 Q 仍有自己的长度、页选择、分数和 TopK。

### 5.2 最近讨论的 sink＋tail 紧凑拼接目标

```text
有效 sink 和有效 tail
  → 仅复制有效成员及各自 scale
  → 按 [sink | tail] 紧凑写入临时固定候选区域
      → 中间不保留源页填充
      → 超过一页则连续占用多页
      → 只在最后一页末尾留填充
  → tables 按新布局生成引用和长度
  → 精筛打分
  → normalize 保留必要的因果、历史和重复覆盖过滤
  → TopK
  → map_topk 按新布局恢复原逻辑位置
```

例如 128 个历史槽位、2 个 sink、33 个 tail：紧凑固定候选占一页，打分长度和紧凑长度均为 163，物理页容量共 192。多个 Q 若 tail 起点不同，还需要公共范围描述及逐 Q 区间处理，不能一概假定两个长度相同。

**这部分目前是目标布局说明，未在本次文档更新中改 kernel。** 详细例子见 [sink＋tail 紧凑布局说明](TABLES_NORMALIZE_MAP_TOPK_EXPLAINED.md)。

## 6. 跨分块 overlap 的完整时间线

```text
主计算流                                  build_stream

分块 c 到达第 l 层
  → 当前 Q/K/V 准备
  → 取得旧索引并满足完成依赖
  → 查询旧索引＋可见固定后缀
  → 当前层 attention
  → submit_build -----------------------> 等待源数据及前序共享状态依赖
  → 继续第 l+1 层及后续计算                → initialize
                                           → 分组
                                           → assemble
                                           → 指纹、查重、分配、publish
                                           → 更新请求页表
                                           → 记录完成事件

分块 c+1 到达第 l 层
  → 等待/确认该完成事件 <------------------ 完成事件
  → 消费已经建立好的索引
```

**当前分块建立的索引，是供后续分块到达同层时使用。** 重叠窗口来自提交之后的其他工作，不是已经结束的本层 attention。

如果构建完成时刻为 `t_build_done`，查询到达依赖点为 `t_query_arrive`，残余构建等待为：

```text
max(0, t_build_done - t_query_arrive)
```

到达前完成比例为 100%、残余等待为 0，说明统计范围内构建没有在消费点造成残余阻塞。但构建仍占 GPU 算力、带宽和工作区，发布及提交也有开销，不能因此断言完整分块耗时不受影响。

## 7. 请求结束与释放回收链

### 7.1 当前运行时的请求释放入口

```text
Indexer.release_request(owner)
  → 获取 CPU 锁
  → _begin()：当前流等待此前 Indexer 操作完成
  → _release(owner)
      → 从 states 移除该请求 State
      → 遍历 state.leases
          → release kernel 读取该桶 16 项重排页表
              → 每个页 refs 减 1
              → 原引用数为 1：本次变为 0，可以回收
              → 前缀和计算回收页在空闲栈中的位置
              → 更新空闲计数，写回空闲页编号
          → 桶记录 lease 返回可用记录列表
      → 维护请求页表张量的流生命周期
  → _finish() 记录完成事件
  → 释放 CPU 锁
```

中心区域按 lease 组织，记录复用时随对应构建重新写入；不是必须逐页执行 `cudaFree`。归还预分配池的空闲页也不等于进程显存立即归还驱动。

这里的事件保护必须覆盖实际使用者。如果外部直接访问索引池而没有纳入协议，仅等待 `Indexer.done` 并不能自动保护这些额外读者。

### 7.2 若接入前缀缓存持有：扩展生命周期链

```text
请求完成/取消，或前缀索引缓存淘汰
  → 解除对应请求或缓存的持有
  → 必要时先使前缀到桶记录的关联失效
  → 等待相关 GPU 构建和查询使用结束
  → 检查桶记录是否还有其他持有者
      ├─ 有：保持桶及对应资源有效
      └─ 无：释放该桶持有的重排页引用
              → 页面 refs 归零后写回空闲栈
              → 释放/归还中心区域和桶记录资源
```

这是缓存接入协议的功能链，不能仅因存在 `release` kernel 就认为所有前缀缓存回调已实现。

两种保留策略的区别：

- **随活跃请求保留**：最后活跃持有结束后释放，后续原生前缀命中可能仍需重建索引。
- **随原生前缀缓存保留**：请求结束后缓存继续持有桶，后续完整桶命中可以复用；缓存淘汰再解除持有。

## 8. CUDA Graph 加在哪一层

```text
CPU 准备本次控制输入
  → 更新固定地址工作区中的长度、预算、源页表、meta 等
  → 建立必要的流间依赖
  → execute()
      ├─ eager：按顺序提交 kernel
      └─ graph：回放已捕获的兼容子图
  → 记录完成事件
  → 后续消费者依事件读取结果
```

Graph 改变提交方式，不改变建索引、过滤、TopK 和映射的语义。可变化的长度和预算必须由对应入口真正支持，且不得超过捕获容量。某个子图支持 Graph，不等于全部 Python 状态管理或完整模型路径都可以直接捕获。

## 9. 实现和详细说明入口

以下链接均指向本仓库中的文件。

| 内容 | 文件 |
|---|---|
| 后端快照、构建提交 | [qwen_gqa_probe_backend.py](../../qwen_gqa_probe_backend.py) |
| 1K 桶、请求状态、查询及释放运行时 | [v1_deepseek/runtime.py](../../sglang/srt/layers/attention/v1_deepseek/runtime.py) |
| 建索引和页管理 kernel | [v1_deepseek/kernels.py](../../sglang/srt/layers/attention/v1_deepseek/kernels.py) |
| 原始分组实现 | [v1_native/fused_grouping.py](../../sglang/srt/layers/attention/v1_native/fused_grouping.py) |
| 分组等优化实验 | [优化实验 runtime](../v1_kernel_opt_20261022/runtime.py)、[optimized.py](../v1_kernel_opt_20261022/optimized.py) |
| 原查询融合包装与 CUDA 实现 | [query_fused.py](../../sglang/srt/layers/attention/v1_deepseek/query_fused.py)、[v1_query.cuh](../../sglang/kernels/jit/csrc/deepseek_v4/v1_query.cuh) |
| 共享布局 route 适配 | [route_shared.py](route_shared.py)、[route_shared.cu](route_shared.cu) |
| 公共页复制 | [tail_union.py](tail_union.py) |
| tables、normalize、map_topk | [kernels.py](kernels.py) |
| 共享精筛实验调用链 | [experiment.py](experiment.py) |
| 逐步建索引代码导读 | [README_INDEX_BUILD_1K.md](../../sglang/srt/layers/attention/v1_deepseek/README_INDEX_BUILD_1K.md) |
| 算子汇总表 | [MLA_INDEX_KERNELS_SHARED.md](MLA_INDEX_KERNELS_SHARED.md) |
| sink＋tail 紧凑拼接目标布局 | [TABLES_NORMALIZE_MAP_TOPK_EXPLAINED.md](TABLES_NORMALIZE_MAP_TOPK_EXPLAINED.md) |

本文由此前论文附录“运行时调用链与完整输入输出示例”、1K 建索引导读及上述代码入口整理。本次新增说明文档，不修改运行时或 kernel，不新增性能测试。
