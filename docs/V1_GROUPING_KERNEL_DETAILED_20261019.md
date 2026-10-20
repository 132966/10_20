# V1 分组 kernel 详解：当前 Qwen GQA、1K 桶版本

更新日期：2026-10-19。本文依据当前实际代码整理，不把早期 4K 包装类的行为直接套到当前 1K 路径上。

## 1. 范围与入口

分组的任务是：对一个桶内同一个 KV head 的 K，根据向量方向生成一个排列，使相近方向的 K 尽量被放进相同的小组。它输出成员排列，不直接完成持久索引发布。

当前参数：

|参数|值|
|---|---:|
|桶内 K 数量|1024|
|单个 K 的维度|128|
|一次构建处理的 KV head 数|1|
|最终小组大小|8 个成员|
|一个重排页的成员数|64|
|每桶最终小组数／中心数|128|
|每桶重排页数|16|
|每个重排页对应中心数|8|

每层的每个 KV head 单独构建。Qwen 这里有 8 个 KV heads，每个对应 4 个 Q heads；**建索引不需要 Q，也不需要将 Q 补齐到 32**，补齐发生在查询打分阶段。

实际调用链：

```text
当前评测后端 submit_build()
  → Indexer.prepare() / _prepare()
    → Build.run()
      → Build.compute()
        → v1_deepseek.kernels.initialize()
        → 逐级调用 fused_grouping._dots / _anchor / _sort
        → assemble() 根据排列重排 K、生成中心
        → 后续指纹、查重、分配、发布
```

本文重点解释 initialize 到排列生成，并说明排列如何交给 assemble。

代码入口：

- [runtime.py：Build.compute，第 171 行](sglang/srt/layers/attention/v1_deepseek/runtime.py:171)
- [kernels.py：initialize，第 10 行](sglang/srt/layers/attention/v1_deepseek/kernels.py:10)
- [fused_grouping.py：_dots，第 39 行](sglang/srt/layers/attention/v1_native/fused_grouping.py:39)
- [fused_grouping.py：_anchor，第 56 行](sglang/srt/layers/attention/v1_native/fused_grouping.py:56)
- [fused_grouping.py：_sort，第 65 行](sglang/srt/layers/attention/v1_native/fused_grouping.py:65)
- [kernels.py：assemble，第 33 行](sglang/srt/layers/attention/v1_deepseek/kernels.py:33)
- [grouping.py：PyTorch 参考算法](sglang/srt/layers/attention/v1_native/grouping.py)

## 2. 算法：用两个代表方向做平衡二分

它不是迭代求质心的 K-means，也不计算全部 K 两两之间的相似度矩阵。

对当前一个大小为 n 的组，执行：

1. 取当前排列的第一个成员为起始点 s。
2. 找到与 s 点积最小的成员 b。
3. 找到与 b 点积最小的成员 a。
4. 对每个成员 i，计算 `margin[i] = dot(u[i], u[a]) - dot(u[i], u[b])`。
5. 按 margin 降序稳定排序。
6. 将前一半和后一半视为两个子组，在下一层分别处理。
7. 直到每组只剩 8 个成员。

这里 `u[i]` 是归一化后的 K。对非零向量，点积等于余弦相似度；全零向量经过带下限的归一化仍为零，不将其解释成严格定义的余弦相似度。

margin 越大，表示相对于 b，该成员越接近 a 的方向。由于使用“排序后对半切分”，两组数量固定相等，不是简单地按 margin 正负分组。因此即使一侧方向的成员更多，也不会得到大小悬殊的组。

两个锚点是两次最小点积搜索得到的代表成员，不保证是全组的最远点对。

## 3. 分组工作区：shape 与含义

以下 shape 属于一个 `Build` 实例，即一个 layer × KV head 的构建器。

|缓冲区|shape / dtype|用途|
|---|---|---|
|`source`|`[16]`, int32|当前 1K 桶的 16 个原生物理页编号|
|`meta`|`[3]`, int32|`[桶逻辑起点, 桶记录 lease, active]`|
|`unit`，kernel 参数 U|`[1024,128]`, float32|1024 个归一化 K，按桶内原始位置保存|
|`perm[0]`、`perm[1]`，P/O|各 `[1024]`, int32|成员排列，两个数组交替作为输入和输出|
|`anchor`，A|`[64,2]`, int32|每个当前组的两个锚点成员 ID：第 0 列 b、第 1 列 a|
|`scores`，S|`[1024]`, float32|不同阶段复用：到 s 的点积、到 b 的点积、最终 margin|
|`flag`|标量 int32|小于 0 执行分组；当前 active 构建时初始化为 -1|

`anchor` 有 64 行，因为最后一次实际处理的组大小是 16，此时有 `1024/16=64` 个组；16 再二分成 8 后不再寻找锚点。

`unit` 在分组过程中不搬动。改变的是 `perm`：后续通过 `unit[perm[i]]` 访问向量。这样避免每一级都重排完整的 128 维向量。

上述 U、两个排列、锚点、分数、flag 合计约 524.5 KiB，不包含原生缓存、重排临时页、中心缓冲区和其他索引元数据。

## 4. initialize：从原生分页缓存读取并归一化

当前路径调用的是 `v1_deepseek/kernels.py` 的 `initialize`，不是旧包装类的 `_init`。

当前原生 Index K 缓存的视图通常表示为 `[P,64,1,132]`，但底层一页的实际打包布局是：

```text
前 8192 字节：64 × 128 个 FP8 K 元素
后  256 字节：64 个 float32 scale
总计 8448 字节 / 页
```

不能把这个视图理解为每个 token 的 128 字节 K 后面紧接它自己的 4 字节 scale。

对于桶内位置 i：

```python
physical_page = source[i // 64]
offset = i % 64
x = fp8_K[physical_page, offset].float() * scale[physical_page, offset]
unit[i] = x / max(sqrt(sum(x * x)), 1e-12)
perm[0][i] = i
```

页编号可为 `[8,31,5,...]`，各页不要求在物理上连续。原生缓存已经按 layer/head 编码寻址；`source` 提供当前桶真正使用的 16 个页编号。

启动配置为 `initialize[(64,)]`。每个 Triton program 处理 16 个成员，每个成员 128 维；64 个 program 刚好覆盖 1024 个成员。

实际构建的 `meta.active=1`，此时 flag 被设为 -1，允许后续分组 kernel 执行。Graph 捕获的 dummy 情况 active=0 时会跳过实际分组。当前 HELMET 评测关闭 Graph。

## 5. _dots：同一个 kernel 的三个阶段

接口：

```python
_dots(U, P, A, S, FLAG, N, START, MODE)
```

当前 1K 调用方式：

```python
_dots[(64, 1)](..., N=n, START=0, MODE=mode,
               num_warps=4, enable_fp_fusion=False)
```

- grid 第 0 维：64 个 program，每个处理 16 个排列位置。
- grid 第 1 维：固定为 1，所以内部 `h=0`。
- `group = i // N`：当前排列位置 i 所属的组。
- `ids = P[i]`：这个位置对应哪个桶内原始成员。
- kernel 读取 `U[ids,:]`，对 128 维做乘法与求和。

### MODE=0：寻找与起始成员最不相似的 b

每个组的起始锚点是该组当前排列的第一个成员：

```python
s = P[group * N]
S[i] = dot(U[P[i]], U[s])
```

随后 `_anchor(..., SLOT=0)` 对这个组的 S 求 argmin，把选中成员写入 `A[group,0]`，记作 b。

### MODE=1：寻找与 b 最不相似的 a

```python
b = A[group, 0]
S[i] = dot(U[P[i]], U[b])
```

随后 `_anchor(..., SLOT=1)` 找到最小值对应成员，写入 `A[group,1]`，记作 a。

MODE=1 的点积仍保留在 S 中，下一阶段会用到。

### MODE=2：计算划分分数 margin

```python
a = A[group, 1]
S[i] = dot(U[P[i]], U[a]) - S[i]
```

右侧旧 S 是 MODE=1 的结果，所以最终为：

```text
dot(U[P[i]], U[a]) - dot(U[P[i]], U[b])
```

MODE=2 之后不再调用 `_anchor`，直接进入 `_sort`。

注意：MODE=0 的起始成员不是固定的全局 token 0；经过前面层级排序后，每个子组都使用它自己的当前第一个成员。

## 6. _anchor：选最小点积的成员

接口：

```python
_anchor(P, A, S, FLAG, N, SLOT)
```

一个 Triton program 处理一个当前组，加载该组 N 个分数：

```python
pos = argmin(S[group*N : (group+1)*N], tie_break_left=True)
chosen = P[group*N + pos]
A[group, SLOT] = chosen
```

重要区别：

- argmin 返回的是“当前组内排列位置”。
- 写入 A 的是该位置对应的“成员 ID”。
- 后续通过成员 ID 读取 U，不把组内位置当作原始 token ID。

分数相同时选择当前排列中靠左的位置。这里的“靠左”不是每次都重新按原始逻辑 token ID 最小值选；它取决于当前 P 的顺序。

启动配置为 `[(1024//n,1)]`，每个 program 使用 `num_warps=4`。

## 7. _sort：稳定降序排列，而非物理搬动 K

接口：

```python
_sort(P, O, S, FLAG, N)
```

一个 program 对一个组的 N 个 margin 排序。排序结果写到另一个排列数组 O，P 本身不覆盖。

为了同时处理浮点排序和同分顺序，代码构造 64 位整数键：

```python
score = +0.0 if score == 0 else score  # 统一 +0 / -0
bits = float32_bit_pattern(score)
ordered = bits ^ (0xffffffff if sign_bit_is_set(bits) else 0x80000000)
key = (uint64(ordered) << 32) | uint64(0xffffffff - current_position)
sorted_key = sort(key, descending=True)
```

含义：

1. 高 32 位把 float32 的正负数编码转换为可以按无符号整数排序的顺序。
2. 高位相同时，低 32 位让当前组内位置更小的成员排在前面。
3. 因此相同分数保持原有相对顺序，实现稳定排序。
4. 从排序后的低位恢复旧位置，再读取 P，写入 O。

这不是给浮点分数加 epsilon，不改变非同分成员的真实大小关系。该排序逻辑按正常有限分数解释；不要把它外推为额外实现了 NaN 清洗策略。

当前 1K 路径每次调用后执行：

```python
current = 1 - current
```

下一层以刚写完的 O 为 P。K 向量仍在 U 中，直到后面的 assemble 才按最终排列进行实际重排。

## 8. 逐级分组：为什么最终恰好是 8 个成员

当前循环：

```python
for n in (1024, 512, 256, 128, 64, 32, 16):
    dots(mode=0)
    anchor(slot=0)
    dots(mode=1)
    anchor(slot=1)
    dots(mode=2)
    sort()
    swap_permutation_buffers()
```

|当前组大小 n|当前组数|排序后下一层子组大小|
|---:|---:|---:|
|1024|1|512|
|512|2|256|
|256|4|128|
|128|8|64|
|64|16|32|
|32|32|16|
|16|64|8，结束|

代码没有单独的“二分复制”kernel。上一层排序后的前半、后半自然占据连续的两个区间；下一层把 N 减半，`group=i//N` 就把它们当成两个独立子组。

后面层级只在各自子组内部排序，不会再跨越已经建立的上层分组边界。

一共 7 层，每层 3 次 dots、2 次 anchor、1 次 sort，即 **42 次 Triton kernel 启动**；加上初始化是 43 次。这个计数仅覆盖本文的归一化与分组，不包含后续 assemble、页管理、数据复制或状态登记。

名称中的 fused 不代表一个桶的整个分组只有一个 kernel。当前仍是多个 kernel 在同一 CUDA stream 上顺序执行，kernel 边界提供前后依赖，无需在每一步额外让 CPU synchronize。

## 9. 一个 16 成员的小例子

用二维单位向量说明最后一次 `N=16` 划分；实际向量是 128 维，其余维补零即可。

设初始排列为 `P=[0,1,...,15]`，每个向量的 x 坐标如下，y 取非负值 `sqrt(1-x²)`：

|成员 ID|0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|x|1|-1|0.9|-0.9|0.8|-0.8|0.7|-0.7|0.6|-0.6|0.5|-0.5|0.4|-0.4|0.3|-0.3|

计算过程：

1. s 为成员 0，即 `(1,0)`。
2. 到 s 点积最小的是成员 1，即 `b=(-1,0)`。
3. 到 b 点积最小的是成员 0，即 `a=(1,0)`。
4. 每个成员 `margin = x - (-x) = 2x`。
5. 降序排列得到：

```text
[0,2,4,6,8,10,12,14, 15,13,11,9,7,5,3,1]
```

前 8 个与后 8 个成为两个最终小组。真实 1K 桶则在更高层先逐级划分，最终对 64 个 16 成员子组分别做这一步。

若两个 margin 相同，则保留它们在当前 P 中的先后关系；锚点最小值同分也使用当前组内靠左位置。

## 10. 分组结果怎样成为重排页和中心

最终 `perm[current]` 是 `[1024]` 的排列，每个数字为桶内原始位置 `0..1023`，所有成员都保留，分组不丢弃 token。

后续 `assemble[(16,8)]` 中：

- 第 0 维 p：16 个重排页，每页 64 个成员。
- 第 1 维 g：每页 8 个小组，每组 8 个成员。
- 小组成员从 `perm[p*64 + g*8 : p*64 + (g+1)*8]` 取得。
- 通过 source 页表读取这些成员的原生 FP8 K 和 scale，写入重排临时页。
- 成员逻辑位置写为 `bucket_start + member_id`。
- 对小组的 8 个反量化 K 求均值，再将均值量化为 FP8 中心和独立 scale。

**中心取的是反量化 K 的均值，不是 U 中单位向量的均值。**U 只用于按方向分组。

例如桶起点为 4096，排列开头是 `[9,100,7,...]`，成员逻辑位置就是 `[4105,4196,4103,...]`。这些是逻辑位置，不是长期保存的原生物理 slot。

最终得到 16 个重排页、128 个中心；中心按每页 64 个打包，占 2 个中心页。中心所属的桶记录与重排数据的物理页编号是不同概念，不应说“重排物理页 20 的中心一定写进中心物理页 20”。

随后指纹、页匹配、分配与 publish 才将结果写入持久 V1 空间；这些步骤不属于 `_dots/_anchor/_sort` 本身。

## 11. 1K 如何复用带有 4096 常量的旧 kernel

旧 `fused_grouping.py` 同时包含：

1. `_dots/_anchor/_sort` 等底层 kernel。
2. `FusedGroups` 这个面向 4K、多 head 工作区的包装类。

当前 `Build.compute()` **直接调用底层 kernel，没有调用 `FusedGroups.run()`**。

底层地址表达式中保留了：

```text
P/S/U 的 head 偏移：h * 4096
A 的 head 偏移：h * 256
```

当前 grid 的第 1 维固定为 1，内部 h 恒为 0，所以这些偏移均为 0。第 0 维只有 64 个 program，每个访问 16 个成员，实际仅覆盖 0..1023。因此当前单 head、1K 缓冲区能够复用这些 kernel。

**不能直接把 grid 第 1 维改为 8，就声称支持 `[8,1024,...]` 的连续工作区。**h>0 后旧的 4096/256 stride 会与 1K 工作区布局不匹配；如要把多个 KV heads 合到一次 launch，需要显式适配 stride 和缓冲区形状。

同理，当前路径的 START=0、P 存桶内位置；绝对逻辑位置在 assemble 时加上 meta 中的桶起点。旧 4K 包装类则可以在 P 中存带 start 的位置，再由 kernel 减 START。两种调用约定不能混用。

## 12. reference/fused 配置和数值一致性

`FusedGroups.run()` 会读取 `SGLANG_V1_GROUPING_MODE`，决定走 PyTorch reference 或底层 kernel；当前 1K `Build.compute()` 直接调用底层 kernel，因此**该环境变量不会切换当前这条调用链**。

PyTorch 参考算法与当前 kernel 的概念步骤相同：归一化、选 b/a、计算 margin、稳定排序、平衡二分。但不自动保证逐位结果一致：

- 当前 source 是 FP8 K 与 scale，归一化建立在反量化值上；旧参考若输入原始 BF16 K，输入本身就不同。
- FP32 点积的归约顺序可能不同，接近的分数可能改变排序或锚点选择。
- 当前 dots 设置 `enable_fp_fusion=False`，只能约束部分浮点融合行为，不能据此宣称与所有 PyTorch 路径位级等价。
- 稳定排序与靠左同分规则只在实际算出的分数相同条件下规定顺序。

这里分组的 tie-break 与前面评测中讨论的查询 Top2048 不确定性不是同一处实现：本文 `_sort` 已明确构造当前位置作为同分排序键，不应把查询 TopK 的原子写出问题套在这个分组排序上。

## 13. 与异步执行、CUDA Graph 的关系

当前 HELMET 后端按 layer × KV head 获取各自 Indexer，把构建提交到一个独立的 `build_stream`。各 head 构建在该 stream 上顺序排队；多个 head 不是各自拥有独立并发 stream。

同一个桶内部，各分组 kernel 依次执行。它们可以与主流上后续模型计算重叠，但不能无视自身的数据依赖并行执行不同 MODE。

当前评测配置 `graphs=False`，所以这里走 eager kernel 启动。底层 `Execution` 另有 Graph 捕获与回放机制；本文不把“代码支持捕获”写成“这次评测实际使用了 Graph”。

分组没有 Q 输入，也没有查询长度维度；固定 1K、固定 128 维有利于固定工作区。不同桶输入通过 source、meta 和缓存内容更新，逻辑上每个桶独立分组，不需要把相邻两个 1K 桶一起重新聚类。

## 14. 阅读建议与本次文档边界

建议依次阅读：`Build.compute()` → `initialize()` → `_dots()` → `_anchor()` → `_sort()` → `assemble()`。

本文通过静态代码核对描述当前行为；未修改 kernel，也没有新做性能或数值等价性实验。当前路径有 7 级分组、43 次归一化/分组启动，是代码推导的启动次数，不是 profiler 采样结果。
