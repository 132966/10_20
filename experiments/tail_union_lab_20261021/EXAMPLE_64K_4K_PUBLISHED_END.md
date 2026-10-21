# 64K 历史、4K chunk：共享尾部准备算子的完整例子

本文对应修正后的 [union.py](union.py)：使用实际的 GPU `indexed_end` 输入，不能再根据 Q 的长度推测索引已经建到哪里。

实验参数：历史长度 65536，当前 chunk 有 4096 个 Q，sink=64，window=2048。单个 K head，K 向量 128 维；后续示例打分有 32 个 Q head。**复制 kernel 本身不读取 Q 向量。**

## 1. 当前哪些 token 已经有索引

本次查询的索引已经完成并发布到逻辑位置 65535，即：

```python
indexed_end = [65536]  # [0,65536) 已有可用索引，终点不包含在内
lens = [65537, 65538, ..., 69632]
```

当前 chunk 的位置是 `[65536,69632)`，没有索引。本例中 `indexed_end` 在整个 chunk 查询期间固定为 65536。

| Q 编号 | 当前 token 的逻辑位置 | 可见长度 n | 可见位置 |
|---:|---:|---:|---|
| 0 | 65536 | 65537 | 0～65536 |
| 1 | 65537 | 65538 | 0～65537 |
| 2047 | 67583 | 67584 | 0～67583 |
| 2048 | 67584 | 67585 | 0～67584 |
| 4095 | 69631 | 69632 | 0～69631 |

**可见长度增长，不代表索引同步完成。**即使已经执行到 Q4095，也不能认为当前 chunk 前面的桶已经发布。

## 2. 输入、输出与具体 shape

每个原生物理页存放 64 个 K，实际按下面的布局打包：

```text
前 8192 字节：64 × 128 个 FP8 K 元素
后  256 字节：64 个 FP32 scale
合计 8448 字节
```

不是每个 token 的 128 字节 K 后面紧跟 4 字节 scale；本实现把整页的 scale 放在整页 K 后面。

| 参数 | 本例 shape / dtype | 含义 |
|---|---|---|
| `src` | `[1088,8448]`, uint8 | 原生物理页池；测试实际创建1088页 |
| `nt` | `[1088]`, int32 | 同一请求的逻辑页→原生物理页映射 |
| `lens` | `[4096]`, int32 | `[65537,…,69632]` |
| `indexed_end` | `[1]`, int32 | `[65536]`，实际已发布索引的连续覆盖终点 |
| `dst` | `[99,8448]`, uint8 | 公共候选缓存预留容量，本次实际使用97页 |
| `meta` | `[4096,3]`, int32 | 每Q的 `[sink数量, tail逻辑起点, 候选数量]`，输出 |
| `desc` | `[4]`, int32 | 公共缓存页布局，输出 |

`1088 = 69632 / 64`。原生物理页数恰好等于请求逻辑页数只是这份独立测试的设置，生产池可以更大。
`nt` 在测试里用随机排列，验证源物理页不连续时仍然正确。

`dst` 的99页是**当前修正版测试**的预留量，不再是旧实验的115页。它是容量，不是有效数据数量；没有要求必须恰好分配99页，至少要覆盖允许的全部回放范围。

## 3. 先计算每个 Q 的 sink 和 tail

辅助函数：

```python
@tr.jit
def bounds(n, end, S: tl.constexpr, W: tl.constexpr):
    s = tl.minimum(n, S)
    t = tl.maximum(
        s,
        tl.minimum(
            tl.minimum(tl.maximum(end, 0), n),
            tl.maximum(s, n - W),
        ),
    )
    return s, t, s + n - t
```

也可以按含义拆开：

```python
sink = min(n, 64)
visible_indexed_end = min(max(indexed_end, 0), n)
window_start = max(sink, n - 2048)
tail = max(sink, min(visible_indexed_end, window_start))
count = sink + n - tail
```

保留的集合是：

```text
sink ∪ 最近2048个可见位置 ∪ 全部未建索引的可见位置
```

这三个集合可能重叠，合并后的候选没有重复。已有索引且不属于sink/window的历史位置，走历史重排页的粗筛路径，不由这里复制全部数据。

### Q0

```text
n = 65537
sink = 64
indexed_end = 65536
window_start = 65537 - 2048 = 63489

tail = min(65536,63489) = 63489
count = 64 + 65537 - 63489 = 2112
```

需要位置 `0～63` 与 `63489～65536`。

### Q2047：刚好到窗口与索引边界相遇的位置

```text
n = 67584
window_start = 67584 - 2048 = 65536
tail = 65536
count = 64 + 2048 = 2112
```

### Q2048：窗口开始向后滑，但未建索引范围不能丢掉

```text
n = 67585
window_start = 65537
indexed_end = 65536

tail = min(65536,65537) = 65536
count = 64 + 2049 = 2113
```

位置65536虽然已经离开最近2K窗口，但没有索引，因此继续保留。

### Q4095：最后一个 Q

```text
n = 69632
window_start = 67584
indexed_end = 65536

tail = min(65536,67584) = 65536
count = 64 + 4096 = 4160
```

需要位置 `0～63` 与 `65536～69631`。因此65537一定参与候选打分。

最终部分 `meta` 是：

```python
meta[0]    = [64, 63489, 2112]
meta[1]    = [64, 63490, 2112]
meta[2047] = [64, 65536, 2112]
meta[2048] = [64, 65536, 2113]
meta[4095] = [64, 65536, 4160]
```

`count` 仅包含这里的sink＋tail，不包含粗筛选中的其他历史重排页，也不是最终attention固定使用的token数。后续仍然可以从全部精筛候选中选Top2048。

## 4. 整个 chunk 的公共范围

kernel 的这几行读取动态输入：

```python
pid = tl.program_id(0)
first = tl.load(LENS)
last = tl.load(LENS + R - 1)
end = tl.load(INDEXED_END)
_, tail, _ = bounds(first, end, S, W)
```

本例：

```text
first = 65537
last = 69632
end = 65536
tail = 63489  # 第一个Q需要的最早tail位置
```

同一个发布终点下，随着n增长，tail起点单调不减。所有Q的尾部并集覆盖 `63489～69631`。

换算成完整页：

```python
sp = (tl.minimum(last, S) + 63) // 64
tp = tl.maximum(sp, tail // 64)
ep = (last + 63) // 64
total = sp + tl.maximum(ep - tp, 0)
```

得到：

```text
sp = 1       # sink页数量
tp = 992     # tail起始逻辑页，63489 // 64
ep = 1088    # 结束逻辑页上界，不包含1088
total = 1 + 1088 - 992 = 97
```

所以：

```python
desc = [1, 992, 1088, 97]
```

公共缓存内容：

| 公共页 | 原序列逻辑页 | 原序列位置 |
|---:|---:|---|
| `dst[0]` | 0 | 0～63 |
| `dst[1]` | 992 | 63488～63551 |
| `dst[2]` | 993 | 63552～63615 |
| `dst[33]` | 1024 | 65536～65599 |
| `dst[96]` | 1087 | 69568～69631 |
| `dst[97:99]` | 无 | 预留，本次不用 |

位置63488因整页复制而一起搬入，但所有Q的有效候选都不会包含它。**存进公共缓存，不等于自动成为某个Q的候选。**

本例公共范围与修正前一样，区别主要在后半个chunk的每Q读取范围。以前数据虽然复制进来了，但错误的meta使后面的Q不读取它们。

## 5. 为什么预留99页

当前 `test_bench.py` 的容量根据最大长度和最小允许的发布终点计算。本例只给 `min_indexed_end=65536`：

```python
first_tail = 63489
cap = min(
    1088,
    ceil(64 / 64) + ceil(69632 / 64) - first_tail // 64 + 2,
)
# min(1088,1+1088-992+2) = 99
```

末尾的2页是测试分配的保守余量。若回放时允许索引滞后到61440或更早，必须按那个更小的发布终点重新计算并预留足够容量，不能继续假设99页总是够用。

## 6. `PreparedStage` 如何启动 kernel

本例推荐入口：

```python
plan = PreparedStage(
    src, nt, lens, indexed_end, dst, meta, desc,
    sink=64, window=2048,
)
plan()
```

`R=4096`，默认选择：

```text
B = 512 个int32单位/program
num_warps = 4
PAGE = False
```

这里4个warp对应128个CUDA线程；不是512个线程，也不是512个token。

`PreparedStage` 提前创建：

```python
src.view(torch.int32)  # [1088,2112]
dst.view(torch.int32)  # [99,2112]
```

每页8448字节，相当于2112个4字节单位。只是共享内存的视图，不分配新缓存、不做数值转换。

启动program数量：

```text
复制网格 = ceil(99 × 2112 / 512) = 409
元数据所需 = ceil(4096 / 128) = 32
最终 grid = (max(409,32),) = (409,)
```

等价的底层调用是：

```python
stage_union[(409,)](
    src.view(torch.int32), nt, lens, indexed_end,
    dst.view(torch.int32), meta, desc,
    4096, 64, 2048, 512, False,
    num_warps=4,
)
```

形状和tile配置确定后网格固定，lens和indexed_end的值仍从GPU读取，因此支持Graph动态回放。

## 7. 每个 program 搬哪些字节

实际走的是 `PAGE=False` 分支：

```python
x = tl.arange(0, B)
target = pid * B + x
page = target // 2112
off = target % 2112
valid = target < total * 2112
```

`x、target、page、off、valid` 的shape都是 `[512]`。

| 变量 | 含义 |
|---|---|
| target | 展平后目标缓存的int32元素编号 |
| page | 公共缓存中的页编号 |
| off | 该页内的int32偏移 |
| valid | 是否属于本次有效97页 |

每个program最多搬512×4=2048字节。

### program 0

```text
pid = 0
target = 0～511
page = 0
off = 0～511
```

复制公共页0的前2048字节。

### program 4：同时跨过向量/scale边界和页边界

```text
target = 2048～2559

前64个元素：目标页0的off=2048～2111，是64个scale的原始位模式
后448个元素：目标页1的off=0～447，是下一页FP8 K的开头
```

因此无需分别启动向量复制和scale复制，地址计算会给每个元素找到正确的源页。

### program 400：只剩最后一页的scale

```text
有效元素数量 = 97 × 2112 = 204864
pid=400时 target=204800～205311
只有204800～204863有效，共64个int32单位
```

这64个单位正是公共页96的scale。其余位置被mask屏蔽。

总共409个program中，0～400参与有效复制；401～408没有有效复制。不会写入 `dst[97:99]` 的预留区。

`PAGE=True` 是调优时测试的另一个分支：一个program处理一页，使用4096元素tile并mask到2112。当前4K配置没有采用它。

## 8. 从公共页找到原生物理页，再搬运

```python
lp = tl.where(page < sp, page, tp + page - sp)
physical = tl.load(NT + lp, page < total, other=0).to(tl.int64)
data = tl.load(SRC + physical * 2112 + off, valid, other=0)
tl.store(DST + target, data, valid)
```

先得到原序列逻辑页：

```text
公共页0  → 逻辑页0
公共页1  → 逻辑页992
公共页33 → 逻辑页1024
公共页96 → 逻辑页1087
```

再查 `nt` 找原生物理页。以下只是说明地址计算的假设值，不是随机测试页表的固定内容：

```python
nt[0] = 8
nt[992] = 500
nt[1024] = 31
```

那么：

```text
dst[0]  ← src[8]
dst[1]  ← src[500]
dst[33] ← src[31]
```

例如公共页1的第10个int32单位：

```text
page=1, off=10
lp=992
physical=500
源偏移 = 500×2112+10 = 1056010
目标偏移 = 1×2112+10 = 2122
```

等价于 `dst_int32[2122] = src_int32[1056010]`。每次复制4字节，FP8 K和FP32 scale的位模式都原样保留。

## 9. 同一次 kernel 内生成4096行 meta

```python
if pid < tr.cdiv(R,128):
    r = pid*128 + tl.arange(0,128)
    n = tl.load(LENS+r,r<R,other=0)
    s,t,c = bounds(n,end,S,W)
    tl.store(META+r*3,s,r<R)
    tl.store(META+r*3+1,t,r<R)
    tl.store(META+r*3+2,c,r<R)
```

本例前32个program额外做这部分工作：

```text
program0：Q0～Q127
program1：Q128～Q255
……
program31：Q3968～Q4095
```

`r、n、s、t、c` 的shape都是 `[128]`。一个program可以同时负责512个搬运元素和128行元数据，它们并不要求一一对应。

对Q4095：

```text
META[4095*3]   = 64
META[4095*3+1] = 65536
META[4095*3+2] = 4160
```

按二维视图看就是 `meta[4095]=[64,65536,4160]`。

## 10. 写 desc

只有program0执行：

```python
if pid == 0:
    tl.store(DESC, sp)
    tl.store(DESC+1, tp)
    tl.store(DESC+2, ep)
    tl.store(DESC+3, total)
```

本例结果为 `[1,992,1088,97]`。
这些都是独立输出；本kernel其他program不依赖读取刚写出的desc，不需要在program间做屏障。后续consumer须在该kernel完成之后执行。

## 11. 后续每个Q如何读取：以65537为重点

每个Q先限制候选编号：

```python
valid = i < count
logical = i if i < sink else tail + i - sink
```

再根据desc得到共享页：

```python
lp = logical // 64
shared_page = lp if lp < sp else sp + lp - tp
slot_in_page = logical % 64
```

| Q | 候选i | logical | 共享页 | 页内位置 |
|---|---:|---:|---:|---:|
| Q0 | 64 | 63489 | 1 | 1 |
| Q0 | 2111，最后有效候选 | 65536 | 33 | 0 |
| Q4095 | 64 | 65536 | 33 | 0 |
| Q4095 | 65 | 65537 | 33 | 1 |
| Q4095 | 4159，最后有效候选 | 69631 | 96 | 63 |

Q0不会读65537，因为它的有效候选到i=2111为止；Q4095会读取65537，因为它的tail从65536开始，i=65有效。

代码中的 `consumer.py` 直接按这个映射读K和scale并打分，不展开为4096份独立K缓存。
`reconstruct_for_test` 只是为逐字节验证重建旧布局，未加入优化路径性能计时。

## 12. 可直接运行的构造和调用示例

在配置好PyTorch、Triton和CUDA的环境中，从本目录运行下面代码。使用有限FP8数据和正scale，源页表随机打乱。

```python
import torch
from union import PreparedStage

src = torch.empty((1088,8448),dtype=torch.uint8,device='cuda')
src[:,:8192] = torch.randn((1088,8192),device='cuda').to(
    torch.float8_e4m3fn
).view(torch.uint8)
src.view(torch.float32)[:,2048:] = torch.rand((1088,64),device='cuda') + 0.01

nt = torch.randperm(1088,dtype=torch.int32,device='cuda')
lens = torch.arange(65537,69633,dtype=torch.int32,device='cuda')
indexed_end = torch.tensor([65536],dtype=torch.int32,device='cuda')
dst = torch.empty((99,8448),dtype=torch.uint8,device='cuda')
meta = torch.empty((4096,3),dtype=torch.int32,device='cuda')
desc = torch.empty((4,),dtype=torch.int32,device='cuda')

plan = PreparedStage(src,nt,lens,indexed_end,dst,meta,desc,sink=64,window=2048)
plan()

assert meta[0].tolist() == [64,63489,2112]
assert meta[2048].tolist() == [64,65536,2113]
assert meta[-1].tolist() == [64,65536,4160]
assert desc.tolist() == [1,992,1088,97]

logical_pages = torch.cat([
    torch.arange(1,device='cuda'),
    torch.arange(992,1088,device='cuda'),
])
torch.testing.assert_close(dst[:97],src[nt[logical_pages].long()],rtol=0,atol=0)
print('PASS: 4K chunk public pages and per-Q bounds')
```

`.tolist()` 只用于演示核对，会同步GPU；正式热路径不需要这些操作。

若要验证65537进入最后Q的Top2048以及动态发布终点，请运行 `test_consumer.py` 和 `test_published_end.py`。

## 13. 修正前后与本次性能

修正前把最后Q的tail设为67584，候选数2112；修正后必须是65536，候选数4160。

逐Q基线也已同步修正，使用 `baseline_published.py`，不是拿漏算的旧基线来比较。

逐Q基线：

```text
前2048个Q：每Q33页
后2048个Q：每64个Q增加一页，分别34～65页
有效写入页数 = 2048×33 + 64×sum(34,…,65) = 168960页
字节数 = 168960×8448 = 1427374080字节 = 1361.25 MiB
```

共享方案：

```text
K＋scale：97×8448 = 819456字节
meta：4096×3×4 = 49152字节
desc：16字节
合计：868624字节 = 848.27 KiB
```

测得的**尾部准备算子**时间，来源 `prepared_results.json`：

| 模式 | 修正逐Q基线 | 公共缓存方案 | 加速 |
|---|---:|---:|---:|
| 非Graph | 447.59 μs | 15.27 μs | 29.31× |
| CUDA Graph | 479.18 μs | 1.82 μs | 263.90× |

非Graph使用CUDA Event覆盖Python连续提交，包含GPU等待CPU提交的间隙；Graph测warm replay。初始化、预分配和编译不计时。未锁频，数据存在测量波动。

加入示例尾部打分后，4K案例约为1.20×。**这里没有测完整V1查询或attention，不能把263.90×解释成它们的加速比。**

## 14. 动态回放的边界

- `indexed_end` 是设备tensor内容，改变它不要求重新编译/捕获，但输出空间必须足够。
- 只有索引已发布且对查询流可见时才能推进它；不是发起异步build就能推进。
- 必须表示连续覆盖终点。有未完成空洞时，不能跳过空洞使用后面某个桶的最大位置。
- 源数据、页表、长度、发布终点要保持一致；复制和消费期间不能被其他流无序修改。
- 当前测试只改独立实验，不会自动修正原项目的查询调度、候选去重或历史重排页过滤。项目接入时这些环节也必须使用一致的发布快照。
