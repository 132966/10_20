# V1 完整调用链：请求执行、建索引、查询、overlap 与回收

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

## 3. 桶内分组的完整链

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

## 4. 查询完整功能链

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

## 5. 固定候选准备：共享页与紧凑拼接

### 5.1 共享页方案

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

### 5.2 sink＋tail 紧凑拼接方案

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
