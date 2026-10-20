# 六项关键实现：小型真实 GPU 测试

全部直接调用原始 GPU kernel，不修改生产实现，不启动模型服务。数据人工构造，便于明确验证；不代表真实模型精度或性能基准。

|实现|语言|测试文件|规模与断言|
|---|---|---|---|
|fused.route|CUDA C++|[example_v1_query_single_head.py](example_v1_query_single_head.py)|1024 Q、单头；8 中心取 max、选 40 页、写入物理页表，与 PyTorch 参考逐项核对|
|fused.topk|CUDA C++|同上|2688 候选选 Top2048；验证过滤、集合、逻辑位置、物理 slot；测试 direct_fine 开关两条路径|
|_sort|Triton|[test_fused_grouping_small.py](../v1_native/test_fused_grouping_small.py)|16 个成员；负数、正负零、同分稳定顺序、输入不被覆盖、flag 停用|
|match_pages → allocate → publish → release|Triton|[test_page_management_and_tail_small.py](test_page_management_and_tail_small.py)|16 页；首次分配、再次复用、两次释放、数据与引用计数；额外检查指纹相同但内容不同不得复用|
|stage_tail_shared|Triton|同上|4 个 Q、两对共享者；复制 K 字节与 scale，验证共享者不重复复制；检查精筛页表指向 owner 的缓冲区|
|_gather_slots|Triton|[test_gather_slots_small.py](../v1_native/test_gather_slots_small.py)|2 个 Q、每行 3 个位置的最小例子；另测真实 Top2048 工作区复用，逐项核对 K/V|

## 1. route 与 topk

详细数据见 [单头示例说明](README_QUERY_SINGLE_HEAD_EXAMPLE.md)。遵照要求使用 1024 个 Q。原接口 topk 要求 B>512，不为缩小例子而改接口。一个 Q head、一个 KV head；所有 Q 使用相同输入与长度，便于核对，不模拟完整 prefill 因果序列。

真实实现：`sglang/kernels/jit/csrc/deepseek_v4/v1_query.cuh`，Python 加载入口为本目录 `query_fused.py`。

## 2. _sort：16 个分数

输入分数：`[2,-1,2,0,-0,3,-3,1,1,2,-1,0,4,4,-4,0]`。

成员编号故意打乱，确保稳定排序按当前排列位置保留同分顺序，而不是按成员编号排序。Python 参考使用 `(-score, current_position)` 排序，与真实 Triton 输出逐项核对。

实现：`../v1_native/fused_grouping.py`。

## 3. 页管理：A、B 共用 16 个页

这里 allocate/release 固定操作一个 1K 桶的 16 页，不能任意缩为一页。

```text
空池：16 个空闲页
A 插入：分配 16 页，引用均为 1，空闲 0
B 插入相同内容和成员：命中 A 的页，引用均为 2，空闲仍为 0
释放 A：引用变为 1，B 的页保留
释放 B：引用变为 0，16 页全部归还
```

数据发布核对重排页字节、成员映射和中心页。另将某页数据或成员改动，但故意保留旧指纹，检查 match_pages 的精确比较能拒绝复用。此处 payload 是简单测试字节，测试的是存储和复用机制，不做 FP8 注意力数值计算。

实现：本目录 `kernels.py`。

## 4. 共享尾部：4 个 Q 只复制两份

```text
lens      = [1100,1100,1101,1101]
tail_rows = [   0,   0,   2,   2]
```

Q0/Q1 共用 Q0 的临时区域，Q2/Q3 共用 Q2 的临时区域。两对都用相同原生页表；SINK=64、WINDOW=64，尾部起点按当前实现得到 1024。

- Q0 复制逻辑位置 0～63 和 1024～1099，共 140 个位置。
- Q2 复制逻辑位置 0～63 和 1024～1100，共 141 个位置。
- 每份使用 3 个页，最后一页剩余位置填零。
- Q1/Q3 自己的临时区域保留哨兵值，证明没有重复写入。
- fine_table_shared 生成 `[1,2,3]、[1,2,3]、[7,8,9]、[7,8,9]`，证明共享查询实际指向相同物理工作区。

`stage_tail_shared` 消费已准备好的 tail_rows，不独自判断两个请求是否可共享；上层必须正确生成 owner 信息。本例显式构造等价的两对输入。

实现：本目录 `kernels.py`。

## 运行

使用已有 torch、Triton、SGLang CUDA 环境，将本仓库根目录加入 PYTHONPATH。在本目录：

```bash
python example_v1_query_single_head.py
python ../v1_native/test_fused_grouping_small.py -v
python test_page_management_and_tail_small.py -v
```

排序单独运行：

```bash
python ../v1_native/test_fused_grouping_small.py SmallKernelTests.test_sort_stable_and_signed_zero -v
```

## 已执行结果

- 融合查询：此前本会话运行通过，1024 个 Q，route 与 topk 的两条映射路径全部通过。
- 分组：此前本会话 6 项真实 GPU 测试全部通过，其中包含独立 _sort 测试。
- 本次新增页管理与共享尾部：2 项真实 GPU 测试全部通过，耗时约 6.44 秒（包含测试开销，不是 kernel 性能指标）。
- 没有修改原 kernel，也没有重跑模型评测。

## 6. K/V 收集融合：_gather_slots（Triton）

实现：[v1_native/kernels.py](../v1_native/kernels.py)，入口 `_gather_slots` 与 `GatherWorkspace.gather()`。
测试：[test_gather_slots_small.py](../v1_native/test_gather_slots_small.py)。

融合操作：逻辑位置 → 请求的物理 slot 映射 → 同时读取 K/V → 直接写 attention 所需布局。未包含 attention 本身，也不改变源数据物理分散的事实。

```text
请求映射 REQ = [5,1,7,0,6,2]
选中位置 IDS = [[0,2,-1], [4,1,4]]
读取物理 slot = [[5,7,填零], [6,1,6]]
输出 K/V shape = [2,3,1,128]
```

最小例子直接调用真实 kernel，分别测试两个 head；K/V 数值不同，源池与 IDS 使用非连续步长，核对 K/V、重复位置、负索引填零和源池不被修改。固定维度 128，核对全部维度。负索引填零只是 gather 行为，**不等于 attention 自动屏蔽这些位置**；正索引必须由调用者保证合法。

第二项调用真实 `GatherWorkspace.gather()`，使用 `[2,2048]` 的索引和 `[2,2048,1,128]` 输出；两次调用改变索引与 head，核对结果并检查输出地址复用。这里只测 gather，不要求测试索引互不重复。

```bash
python ../v1_native/test_gather_slots_small.py -v
```

这是正确性测试，不是与 PyTorch 的速度对比。文件名保留 README_FIVE_KERNEL_EXAMPLES.md，避免旧链接失效，内容已扩展为六项。

第六项实测：2 项真实 GPU 测试全部通过（test_small_kernel、test_real_workspace_reuse），运行耗时 4.36 秒，包含测试开销，不作为性能数据。
