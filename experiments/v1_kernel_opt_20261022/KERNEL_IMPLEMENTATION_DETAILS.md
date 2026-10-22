# 15个 V1 算子的具体实现与优化说明

本文按实际源码解释：输入是什么、GPU怎样处理、输出是什么、减少了什么开销，以及哪些优化已测试。建索引以本次 `v1_kernel_opt_20261022` 为准；共享精筛以此前 `fine_pool_shared_20261021` 为准。核对日期：2026-10-22。

文中的“优化点”既包含原来已有的设计，也包含这次新增改动，逐项区分；没有实测的部分不宣称加速倍数。

## 1. 统一变量、数据布局与调用链

建索引示例：1个 Index K head，1K桶，每个K有128维，每页64个K。桶起点4096，桶记录 lease=7，因此 `META=[4096,7,1]`。要使用lease=7，调用方需预分配至少8条桶记录；适配器默认records=2仅用于小测试。

|名称|形状/含义|
|---|---|
|原生池|字节视图 `[P_n,8448]`；每页8192字节FP8 K＋256字节scale|
|FP8视图|与原生池共用内存；有效K区域为每页 `[64,128]`|
|FP32视图|同一内存解释成 `[P_n,2112]`，每页下标2048～2111保存64个scale|
|INT32视图|同一内存解释成 `[P_n,2112]`，用于按4字节搬运或比较，不做数值转换|
|TABLE|`[16] int32`，本桶的原生物理页编号|
|META|`[3] int32`，桶逻辑起点、桶记录lease、active|
|P/O|成员排列；本桶有效 `[1024] int32`|
|U|归一化K；本桶有效 `[1024,128] float32`|
|S|分组分数；本桶有效 `[1024] float32`|
|A|每个当前组两个锚点ID；最大有效 `[64,2] int32`|
|CAP|重排历史池容量，例如4096页；不是当前已用页数|

**实际适配器保留旧工作区容量**：U分配 `[4096,128]`，P/O/S分配 `[4096]`，A分配 `[256,2]`，但1K路径只访问所需前缀。不能把工作区容量4096理解为这次构建了4K桶。

页布局是：

```text
一页8448字节
├─ [0,8192)    64×128个FP8 K字节
└─ [8192,8448) 64个FP32 scale
```

`[64,1,132]` 是接口所用的页形状，不能据此理解为物理上每个K后面紧跟一个scale；实际是K区域和scale区域分别连续。

建索引：

```text
initialize
  → 7轮分组：N=1024、512、256、128、64、32、16
  → assemble
  → fingerprint → match_pages → allocate → publish
  → 请求不用此桶时release
```

共享查询目标链路：

```text
中心打分（DeepGEMM）→ route
原生池 → stage_union
          ↓
       tables → DeepGEMM精筛 → normalize → torch.topk → map_topk
```

中心和候选矩阵打分属于复用的DeepGEMM组件，不计入这里15个自定义功能。当前共享实验最终TopK仍是 `torch.topk`，不是旧 `fused.topk`。原性能实验使用预先提供的候选页；route连接测试从已提供的中心分数开始，不包括完整模型的中心打分。

## 2. 15项总览

|序号|功能名|当前实现|实现语言|本次变化|
|---|---|---|---|---|
|1|initialize|reference/kernels.py|Triton|沿用|
|2|_dots|原_dots＋anchored_dots＋split_fused|Triton|大组部分融合、小组整轮融合|
|3|_anchor|融合进上述两个kernel|Triton|默认优化路径不再单独启动_anchor|
|4|_sort|CUB首轮；中间轮tl.sort；小轮融合|CUDA C++/Triton|按组大小选择实现|
|5|assemble|reference/kernels.py|Triton|沿用|
|6|fingerprint|fingerprint_words|Triton|32位加载，保持原hash|
|7|match_pages|match_words|Triton|32位精确比较|
|8|allocate|reference/kernels.py|Triton|沿用|
|9|publish|publish_words|Triton|32位复制|
|10|release|reference/kernels.py|Triton|沿用|
|11|route|route_shared.cu|CUDA C++/CUB|沿用此前物理页输出适配|
|12|stage_union|tail_union.py|Triton|沿用此前公共页复制|
|13|tables|共享实验kernels.py|Triton|沿用|
|14|normalize|共享实验kernels.py|Triton|沿用|
|15|map_topk|共享实验kernels.py|Triton|沿用|

这里的15是功能项，不等于GPU启动15次。融合后多个功能可以在一次启动内完成，分组又要重复7轮。

## 3. initialize：读取、反量化、归一化、建立初始排列

源码：[reference/kernels.py](reference/kernels.py)，函数 `initialize`；调用见 [runtime.py](runtime.py) 的 `Build.initialize()`。

### 输入

|变量|Shape|示例/含义|
|---|---|---|
|K|`[P_n,8448]` FP8视图|原生页的K字节|
|KS|`[P_n,2112]` FP32视图|读取同一页的scale|
|TABLE|`[16]`|如 `[8,31,5,…]`|
|META|`[3]`|`[4096,7,1]`|

### 实现与例子

启动 `initialize[(64,)](...)`，共64个program，每个处理16个K，合计1024个。program不是单个线程，内部共同处理一个 `[16,128]` tile。

```text
桶内成员i
 → TABLE[i//64]找到原生物理页
 → 页内i%64读取FP8 K和scale
 → FP32 K = FP8值 × scale
 → norm = sqrt(sum(K²))
 → U[i] = K / max(norm,1e-12)
```

例如FP8向量前两维 `[3,4]`、scale=0.5，反量化为 `[1.5,2]`，归一化后 `[0.6,0.8]`。同时写 `P[i]=i`。program 0把active转换为FLAG：active非零时-1，否则0；后续分组kernel在FLAG<0时才执行。

### 优化如何产生

这是沿用的融合：寻址、反量化、范数归约、除法和初始排列一次完成，没有先写出完整反量化K，再由第二个kernel归一化。归一化只做一次，7轮共享U。原始K没有被修改。

### 输出

|变量|有效Shape|内容|
|---|---|---|
|UNIT/U|`[1024,128]`|归一化向量|
|PERM/P|`[1024]`|`[0,1,…,1023]`|
|FLAG|`[1]`|active时-1，否则0|

active=0时U/P保持原内容，调用方必须让后续链路也尊重active。初始化不是索引查重命中检测。

## 4. _dots：分组点积，以及本次融合

源码：[reference/fused_grouping.py](reference/fused_grouping.py) 的 `_dots`；新实现见 [optimized.py](optimized.py) 的 `anchored_dots`、`split_fused`。

### 输入

|变量|有效Shape|含义|
|---|---|---|
|U|`[1024,128]`|归一化K|
|P|`[1024]`|当前排列，按连续N项组成一组|
|A|`[组数,2]`|两列为b、a成员ID|
|S或PREV|`[1024]`|上一阶段分数|
|FLAG|`[1]`|是否执行|
|N、MODE|编译参数|组大小和点积阶段|

### 原实现过程

以第一轮N=1024为例：

```text
seed=P[0]
MODE0：s0[i] = dot(U[P[i]], U[seed])
选b = s0最小值对应的成员
MODE1：sb[i] = dot(U[P[i]], U[b])
选a = sb最小值对应的成员
MODE2：score[i] = dot(U[P[i]], U[a]) - sb[i]
```

差分数大表示更偏向a；排序后的前后两半分别进入下一轮。不是对全部K计算1024×1024两两相似度。

原_dots每次启动64个program，每个算16个成员×128维。原函数带head轴，适配器调用grid `(64,1)`；START=0，P保存桶内编号。

### 新实现一：N≥128，部分融合

```text
_dots MODE0 → s0
anchored_dots MODE1：argmin(s0)找b，同时算sb → s1
anchored_dots MODE2：argmin(s1)找a，同时算sa-sb → scores
排序
```

仍然每个program处理16个成员。每个tile自行读取所属组全部N个分数，做argmin，再计算本tile点积；同组内会重复argmin，这是用额外小归约换取减少两次独立kernel启动。组内第一个tile负责写A，避免多个program重复写同一锚点。

**PREV和OUT分离很关键**：不能一边让某些program覆盖组分数、一边让其他program还在读取它做argmin。实现使用s0/s1/scores，kernel顺序保证前一阶段完成后才读。

### 新实现二：N≤64，整轮融合

`split_fused[(1024//N,)]`：每组一个program，读取 `[N,128]` 成员，完成seed点积、选b、b点积、选a、a点积、差分数和稳定排序。逻辑上x在同一program内复用，避免跨kernel反复从全局内存读取；实际寄存器/共享内存分配由编译器决定，不能承诺所有数据永远都只在寄存器。

大组没有采用整轮融合：N=1024实测约1ms，资源压力过大，比原一轮约19.6μs更慢。小组使用实测配置：N64为4 warps、N32为16、N16为8。

### 输出

|变量|有效Shape|内容|
|---|---|---|
|s0/s1/scores|`[1024]`|对应阶段分数|
|A|`[1024/N,2]`|大组融合时顺便写锚点|
|O|`[1024]`|小组整轮融合时同时输出新排列|

本次减少的是启动和中间阶段开销，不是用Tensor Core加速成功；Tensor Core候选未启用，原因见末尾。

## 5. _anchor：选最小分数成员

源码：原 `_anchor` 在 [reference/fused_grouping.py](reference/fused_grouping.py)；优化路径融合在 [optimized.py](optimized.py)。

### 输入

|变量|有效Shape|含义|
|---|---|---|
|P|`[1024]`|当前组内成员排列|
|S|`[1024]`|各排列位置的点积分数|
|FLAG|`[1]`|执行开关|
|N、SLOT|标量配置|组大小；SLOT=0写b，1写a|

### 实现与例子

原grid `(1024//N,1)`，一个program处理一个组的N项，`argmin(..., tie_break_left=True)`得到位置pos，再经P得到成员编号。

例如某组 `P=[9,2,7,4]`、`S=[0.8,-0.4,-0.4,0.2]`：最小值同分选左边，pos=1，chosen=P[1]=2。输出的是成员ID=2，不是位置1，也不是分数-0.4。

### 优化如何产生

原实现已把argmin和按P取值合成一次kernel，避免PyTorch多个操作及中间pos/chosen张量。**本次默认路径进一步不单独启动它**，直接把选择结果用于后继点积。小组整轮融合还避免把每个中间阶段分数写出再读取；为了与参考结果对照，最终A和scores仍写回。

### 输出

|变量|Shape|内容|
|---|---|---|
|A[:,SLOT]|`[1024/N]`|各组的b或a成员编号|

## 6. _sort：稳定排序，只移动成员ID

源码：[reference/fused_grouping.py](reference/fused_grouping.py)、[sort_cub.cu](sort_cub.cu)、[optimized.py](optimized.py) 的 `split_fused`。

### 输入

|变量|有效Shape|含义|
|---|---|---|
|P|`[1024] int32`|当前排列|
|S|`[1024] float32`|差分数|
|FLAG|`[1] int32`|执行开关|
|N|标量配置|组大小|

### 原实现

每组一个program。先把+0/-0统一，再把FP32位模式映射为可按无符号整数排序的键：负数按位取反，非负数翻转符号位。高32位放分数顺序，低32位放 `0xffffffff-i`。64位键降序后，同分按原位置i递增。

例如P=`[9,2,7,4]`，S=`[0.2,0.8,0.8,-0.1]`，排序位置为 `[1,2,0,3]`，输出O=`[2,7,9,4]`。

只排序4字节ID，不在每轮搬128维K。P/O交替作为输入输出；N减半后，排好的一段前后两半自然成为两组，不需要额外split-copy。

### 本次具体改变

- N=1024：CUDA block含128线程，每线程8项，使用CUB `BlockRadixSort<unsigned,...,int>`。32位有序分数为key，成员ID为value，依靠CUB稳定排序保留同分原顺序，不再把位置塞进64位key。
- N=512/256/128：仍使用原Triton稳定排序。
- N=64/32/16：排序已在 `split_fused` 内完成。

32位CUB只在首轮有收益：Graph 10.769→7.220μs；N512却从3.638变为5.008μs，所以不启用。64位CUB试验也没有保留为默认实现。

### 输出

|变量|有效Shape|内容|
|---|---|---|
|O|`[1024] int32`|各组分数降序、同分保序的新排列|

P与O不能重叠。随机、同分、正负零、无穷和FLAG停用已测试；不把这些测试扩展成任意NaN语义保证。

## 7. assemble：最后一次重排K，同时生成中心

源码：[reference/kernels.py](reference/kernels.py)，本次沿用。

### 输入

|变量|Shape|含义|
|---|---|---|
|K/KS|原生池视图|原始FP8 K及scale|
|TABLE|`[16]`|原生页表|
|META|`[3]`|桶起点、lease、active|
|IDS|有效 `[1024]`|7轮后的最终排列|

### 实现与例子

grid `(16,8)`：16个输出页，每页8个program，每个program处理8个成员×128维。若某8成员排列为 `[3,100,7,8,9,14,2,5]`：

```text
根据IDS和TABLE找到原生K
 → 原样复制8个FP8 K及其scale到TMP相邻位置
 → MAP写入桶起点+IDS
 → 反量化这8个原始K，逐维求均值
 → 为中心求scale并量化成FP8
```

中心scale是 `2^ceil(log2(max(max_abs(mean),1e-4)/448))`，中心值为mean/scale再转FP8。中心来自原始反量化K，**不是U中的归一化K**。

### 优化如何产生

沿用设计：7轮仅改成员排列，真正的K只在最后重排一次；利用本次已读取的8个K顺便求中心、写映射，不另开中心计算和成员映射kernel。不是一次复制就构建全部索引，也不是这次新修改。

### 输出

|变量|Shape|内容|
|---|---|---|
|TMP|`[16,8448]`|重排K及scale|
|TC|`[2,8448]`|128个中心及scale，共2个中心页|
|MAP|`[16,64] int32`|原逻辑位置，例如4096+3=4099|

TMPS/TCS只是TMP/TC的FP32视图，不额外存一套scale。源码旧注释提到每记录8个中心页，这是4K背景；此1K路径实际每记录2个中心页，128个中心。

## 8. fingerprint：用32位加载生成完全相同的指纹

源码：原 `fingerprint` 在 [reference/kernels.py](reference/kernels.py)，新 `fingerprint_words` 在 [optimized.py](optimized.py)。

### 输入

|变量|Shape|含义|
|---|---|---|
|TMP32|`[16,2112] int32`|TMP的同内存视图|
|MAP|`[16,64] int32`|成员逻辑位置|
|META|`[3]`|active控制|

### 原逻辑与新实现

原来每页一个program，创建16384个逻辑元素，前8448个读取字节，其余补零；成员映射也用同样大的向量仅取前64项。两个hash都是字节及成员的整数加权和。

新实现每页仍一个program，但只需4096个逻辑元素，前2112项各加载一个32位word，另外独立64项读取MAP。

一个word内四个字节为b0、b1、b2、b3，其原字节位置是4i～4i+3：

```text
sum_b = b0+b1+b2+b3
weighted = sum_b*(4i+1) + b1 + 2*b2 + 3*b3
         = b0*(4i+1)+b1*(4i+2)+b2*(4i+3)+b3*(4i+4)
```

例如i=0、字节 `[1,2,3,4]`，weighted=30，与逐字节 `1×1+2×2+3×3+4×4` 完全相同。累加得到XW与XS后：

```text
h0 = XW + sum(MAP)*104729
h1 = (XW+6*XS)*1000003 + sum(MAP[j]*(j+1))*65537
```

### 优化如何产生

读取有效字节总数没减少，仍须完整覆盖K和scale；减少的是逻辑lane、掩码和整数归约规模，避免把仅64项的成员归约扩成16384项。不是换成更弱的新hash，测试与原两个int64值逐位一致。指纹仍可能碰撞，不能代替内容比较。

### 输出

|变量|Shape|内容|
|---|---|---|
|HASH|`[16,2] int64`|每个临时页的两个指纹|

Graph实测3.194→2.243μs，1.42×。

## 9. match_pages：先筛指纹，再精确比较内容

源码：新 `match_words` 在 [optimized.py](optimized.py)。

### 输入

|变量|Shape|含义|
|---|---|---|
|TMP32/MAP/HASH|`[16,2112] / [16,64] / [16,2]`|本桶临时页|
|FINE32|`[CAP,2112] int32`|已发布重排页|
|MEMBERS|`[CAP,64] int32`|池中成员逻辑位置|
|HASHES|`[CAP,2] int64`|池中指纹|
|REFS|`[CAP] int32`|引用计数，>0表示在用|
|META|`[3]`|active|
|CAP/SCAN|配置|例如4096/4096；SCAN为覆盖CAP的2次幂|

### 实现与例子

16个program，每个负责一个临时页。先扫描CAP项引用计数和指纹：

```text
j<CAP 且 refs[j]>0 且两个hash相等
 → good候选掩码
 → 找good中的最小物理页号
 → 精确比较该页的K/scale和64个成员ID
 → 一致则返回；不一致则剔除该候选，继续找下一个
```

例如指纹命中页7和20，页7内容不同、页20内容相同，最终MATCH[p]=20；没有完全一致的页则-1。同hash不保证只有一个候选。

新实现比较2112个int32而不是8448个uint8，位模式相同才算一致；成员用独立64项比较。即使K完全相同，只要逻辑成员不同，也不会复用。

### 优化如何产生

减少比较和归约的元素数量，不减少必须验证的有效数据量；指纹扫描仍与CAP相关，没有改成哈希表，也没有解决所有大容量扩展问题。新临时页之间不在此函数互相去重，只查已经在池中且refs>0的页。

### 输出

|变量|Shape|内容|
|---|---|---|
|MATCH|`[16] int32`|如 `[20,-1,7,…]`|

4096页Graph实测：命中4.281→3.593μs；碰撞5.987→4.825μs。空池/未命中收益较小。

## 10. allocate：批量领页、复用和加引用

源码：[reference/kernels.py](reference/kernels.py)，沿用。

### 输入

|变量|Shape|含义|
|---|---|---|
|META|`[3]`|lease决定写哪条桶页表|
|MATCH|`[16]`|复用页或-1|
|FREE|`[CAP]`|空闲物理页栈|
|COUNT|`[1]`|空闲栈有效长度|
|REFS|`[CAP]`|引用计数|
|TABLES|`[records,16]`|每桶记录的物理页表|

### 实现与例子

一个program处理16项：`fresh=MATCH<0`，对fresh做cumsum得到新页排名rank，sum得到needed。从栈顶按 `FREE[count-1-rank]` 批量取页，复用项仍用MATCH。

只看前4项示意：MATCH=`[20,-1,7,-1]`，fresh=`[0,1,0,1]`，rank=`[-1,0,0,1]`。若空闲栈顶依次是31、30，新页输出为 `[20,31,7,30]`。没有fresh的项不使用rank取栈。

然后对每个dst原子加引用，COUNT减needed，写TABLES[lease]和NEW。

### 优化如何产生与约束

GPU内一次完成16页的前缀和、领页、计数和登记，CPU不逐页分配。**这是已有实现，本次未修改。** COUNT在allocate里是普通读写，不支持无序并发分配；必须由流/事件或上层串行化保护共享分配器。relaxed原子仅表示引用计数更新的内存序，不意味着自动解决发布同步。

### 输出/被更新状态

|变量|Shape|内容|
|---|---|---|
|TABLES[lease]|`[16]`|最终物理页编号|
|NEW|`[16] bool`|是否需要写新页|
|REFS|`[CAP]`|相应项+1|
|COUNT|`[1]`|减去新领页数|

容量必须足够。代码存在device_assert，但不能把它当成所有运行配置下都生效的完整溢出恢复机制。

## 11. publish：只写新页，中心按桶记录发布

源码：新 `publish_words` 在 [optimized.py](optimized.py)。

### 输入

|变量|Shape|含义|
|---|---|---|
|TMP32|`[16,2112]`|临时重排K/scale|
|TC32|`[2,2112]`|临时中心/scale|
|MAP/HASH|`[16,64] / [16,2]`|成员、指纹|
|TABLES/NEW|`[records,16] / [16]`|目标物理页与是否新页|
|META|`[3]`|lease、active|

### 实现与例子

16个program，每个临时页一个。如果NEW[p]为真，把TMP对应2112个word复制到TABLES[lease,p]，另写64个成员和两个hash；如果是假，复用页不用重新写。

每个program还发布其8个中心，不论重排页是否复用。中心属于lease：

```text
src_center = p*8+g
 dst_center = lease*128+src_center
```

例如lease=7、p=2、g=0，目标中心编号为912。中心页号=912//64=14，页内中心=16。

旧中心K复制处理1024个字节；新实现处理256个word（每中心32个word）。另8个word复制8个FP32 scale的原始位模式，不重新量化、不做数值类型转换。

### 优化如何产生

减少复制指令涉及的逻辑元素和掩码规模。复用页跳过数据写入是原本已有的优化；本次新增的是32位搬运。新页的有效写入字节数仍相同，所以不能宣称HBM流量必然降到1/4。

### 输出/被更新状态

|变量|Shape|内容|
|---|---|---|
|FINE|`[CAP,8448]`|新页K和scale|
|MEMBERS/HASHES|`[CAP,64] / [CAP,2]`|新页成员与指纹|
|CENTERS|`[records*2,8448]`|当前lease的128个中心及scale|

Graph全新页3.484→2.299μs，1.52×；全部复用几乎没有差异。publish返回只是GPU序列中的一步，跨流消费者仍需等待完成事件，不能仅看TABLES已经写入就提前读页内容。

## 12. release：减引用并批量回收

源码：[reference/kernels.py](reference/kernels.py)，沿用。

### 输入

|变量|Shape|含义|
|---|---|---|
|TABLE|`[16]`|要释放的桶记录页表|
|REFS|`[CAP]`|各物理页引用数|
|FREE/COUNT|`[CAP] / [1]`|空闲栈及有效长度|

### 实现与例子

一个program处理16页。`old=atomic_add(REFS[page],-1)`，只有old=1意味着本次减到0。对dead做cumsum得到回收排名，sum得到回收数量，原子给COUNT加n，得到原栈尾base，再把页写入 `FREE[base+rank]`。

例如本次页 `[20,31,7,30]`，旧引用 `[2,1,1,3]`，dead=`[0,1,1,0]`；base=10，则FREE[10]=31、FREE[11]=7。其他页仍有引用，不回收。

### 优化如何产生与约束

沿用GPU批量原子计数＋前缀和压紧，避免CPU逐项判断。这里只归还编号，不擦除页数据；后续分配新页会覆盖内容。COUNT增长和FREE写入不是一个原子事务，不能与无序并发allocate混用；所有查询完成后才能释放其引用。

### 输出/被更新状态

|变量|Shape|内容|
|---|---|---|
|REFS|`[CAP]`|所释放引用减1|
|FREE|`[CAP]`|尾部新增零引用页编号|
|COUNT|`[1]`|增加回收页数|

release不自动清空TABLE，调用方必须避免同一引用被重复释放。

## 13. route：8中心max＋页面选择＋物理页输出

源码：[../fine_pool_shared_20261021/route_shared.cu](../fine_pool_shared_20261021/route_shared.cu)，沿用此前适配。

### 输入

以1024个Q、最多4096个历史页、最多选40页为例：

|变量|Shape|含义|
|---|---|---|
|scores|`[1024,4096*8] float32`|每页8个中心的打分，已经完成Q-head聚合|
|published|`[1024] int32`|各Q允许使用的已发布页数|
|budgets|`[1024] int32`|动态页面预算，例如40|
|rt|`[1024,4096] int32`|请求页序号→重排物理页号|

### 实现

一个CUDA block处理一个Q，128线程。ITEMS为覆盖页容量所需的每线程项数，4096页时为32。每个有效页用两次float4加载8个中心分数，取max，直接保存在block内部排序输入中。

CUB `BlockRadixSort` 把全部容量项排序，输出前take项；**这是全block排序后截取TopK，不是只对K项工作的选择算法**。take为预算、已发布数量、输出宽度三者最小值。

最后直接用选中的请求页序号查rt，把物理页号写到selected。比如页面选择结果是请求页序号 `[2,0]`、rt前3项为 `[20,35,9]`，输出selected=`[9,20]`。

take=0或take=active有快速路径，不排序；每次回放都会把输出剩余位置清为安全页号0。

### 优化如何产生

8中心max、页排序、物理页映射在一个kernel内，避免写出页面分数矩阵后再单独TopK及映射。此前适配已把旧的“请求页序号输出”改为物理页输出，并使用GPU published，不凭长度猜已建页数。本次没有再改它。

### 输出

|变量|Shape|内容|
|---|---|---|
|selected|`[1024,40] int32`|候选物理页编号，后续tables直接使用|
|takes|`[1024] int32`|实际选页数|

当前支持最多4096页，即256K个Index K位置，C4下约1M原始token。保持静态共享内存。页面同分顺序不要套用分组_sort的“原位置稳定”说明；route的线程装载顺序不同，测试应按其实际规则比较。

## 14. stage_union：整个chunk公共范围只复制一次

源码：[../fine_pool_shared_20261021/tail_union.py](../fine_pool_shared_20261021/tail_union.py)。

### 输入

|变量|Shape|示例|
|---|---|---|
|SRC|`[P_n,8448] uint8`，kernel用int32视图|原生Index K池|
|NT|`[P_n] int32`|单请求逻辑页→原生物理页|
|LENS|`[4096] int32`|`[65537,…,69632]`，因果可见长度|
|INDEXED_END|`[1] int32`|65536，实际已发布连续索引终点（不含该位置）|
|S/W|配置|64/2048|
|DST|`[容量,8448]`|预分配的精筛池后缀视图|

### 具体范围计算

对每个Q长度n：

```text
s = min(n,S)
t = max(s, min(clamp(end,0,n), max(s,n-W)))
c = s+n-t
候选固定范围 = [0,s) ∪ [t,n)
```

first=65537时，s=64、t=63489、c=2112。last=69632时，t被实际end限制为65536，c=4160；**当前chunk未建索引的可见位置也包含在tail里**，不能只复制最后2048个位置。

公共页覆盖用首Q的tail起点和末Q的长度：sp=1、tp=992、ep=1088，因此共97页。逻辑页0和992～1087各复制一次，既有历史窗口，也包含当前chunk位置。

默认将池视图按int32平铺，4096 Q时BLOCK=512：每program处理512个word=2048字节，program可能跨页，不是一Q一program。由target//2112找到公共页、target%2112找到页内word，再通过NT定位原生页。按最大容量固定grid，实际有效范围用mask控制。

部分program同时生成每Q的s/t/c，program0写公共布局。目的DST是整个精筛池的后缀，因此消费者依然只有一个K池基址。

### 优化如何产生

旧方法为每个Q单独复制sink/tail，大量重复搬相同历史页；新方法复制整个chunk所需页的并集一次，各Q用独立页表引用公共副本。它确实减少重复HBM读写及缓存容量，和仅把uint8换int32的优化不同。

这是此前已有的共享优化，本次未改。要求单请求共享NT、LENS非负且非递减、end为实际已发布前缀；跨请求或不同页表不能直接合并。源KV必须已经写好并保持有效，复制不能读尚未生成的当前chunk K。

### 输出

|变量|Shape|内容|
|---|---|---|
|DST|有效 `[97,8448]`|公共页字节副本，实际预分配可更大|
|META|`[4096,3]`|每Q的 `[sink,tail,count]`|
|DESC|`[4]`|`[1,992,1088,97]`，sink页数、tail首逻辑页、结束页、总页数|

公共缓存在HBM中。META保留给说明和测试；当前tables/normalize会自行按LENS与END计算边界，并没有读取META。

## 15. tables：给每个Q组织同一精筛池的页表

源码：[../fine_pool_shared_20261021/kernels.py](../fine_pool_shared_20261021/kernels.py)，函数 `tables`。

### 输入

|变量|Shape|含义|
|---|---|---|
|LENS|`[R]`|各Q可见长度|
|END|`[1]`|已发布索引终点|
|TAKES|`[R]`|历史候选页数|
|SELECTED|`[R,MAXTAKE]`|重排物理页号|
|DESC|`[4]`|公共缓存布局|
|CAP|配置|历史重排池容量，公共后缀从这个页号开始|
|FP/B|配置|输出页表宽度及覆盖它的2次幂|

### 具体实现

每Q一个program，一次处理B个页表槽。前take项直接复制SELECTED；后面把该Q需要的sink页和tail页转换成公共后缀页号。不同Q可以引用相同公共物理页，也可从公共tail的不同位置开始。无效预留槽写0，避免后端预取越界。

**有头部页内padding的例子**：一个Q，n=161、end=128、S=2、W=32，take=2。计算s=2、t=128、c=35。假设DESC=`[1,2,3,2]`、CAP=10、SELECTED=`[3,7]`。

```text
FT有效项 = [3,7,10,11]
前两页：128个历史候选
第10页：公共sink所在完整页，含逻辑0～63
第11页：tail所在页，从逻辑128开始
```

精筛按页布局计算：历史128项＋sink页64项＋最后tail页可见33项，所以DGLENS=225。

真正的紧凑候选只有：历史128项＋sink2项＋tail33项，所以CLENS=163。两者差62，就是sink页中不需要的逻辑2～63；后面normalize跳过它们。DGLENS不是必定等于完整页数×64，末页可以只取部分长度。

### 优化如何产生

只写小页表，不为每个Q复制K；保持DeepGEMM原有单池ABI。查询长度变化仍从GPU读取，固定的是容量与grid。CLENS用于描述紧凑候选数量；当前experiment的torch.topk输入仍是固定WIDTH并用-inf填充，并不是直接用CLENS裁剪张量。

### 输出

|变量|Shape|内容|
|---|---|---|
|FT|`[R,FP] int32`|历史候选＋公共页引用|
|DGLENS|分配 `[R,1]`，线性访问R项|DeepGEMM实际布局的打分长度，最小1|
|CLENS|`[R]`|紧凑候选长度，允许0|

## 16. normalize：去掉布局padding并过滤历史重复覆盖

源码：共享实验 `kernels.py`，函数 `normalize`。

### 输入

|变量|Shape|含义|
|---|---|---|
|RAW|`[R,实际stride] float32`|DeepGEMM的页布局分数|
|LENS/END/TAKES|`[R]/[1]/[R]`|长度、实际索引终点、历史候选数量|
|SELECTED|`[R,MAXTAKE]`|历史候选物理页|
|MEMBERS|`[CAP,64]`|历史页的原逻辑位置|
|WIDTH|配置|固定输出分数宽度|

### 实现与例子

grid `(R,ceil(WIDTH/256))`，一个program处理某Q的256个输出位置。输出固定采用紧凑顺序：历史候选页→sink→精确tail。

沿用上节例子，hist=128：

```text
输出0～127   ← RAW 0～127，附加历史边界过滤
输出128～129 ← RAW 128～129，sink逻辑0、1
输出130～162 ← RAW 192～224，tail逻辑128～160
其余输出    ← -inf
```

历史项通过SELECTED/MEMBERS取得逻辑位置，只保留 `member>=s && member<t && member<n && member<end`。比如历史页里出现逻辑0，因它已经属于sink，历史副本置-inf；逻辑130已在tail中，历史副本也置-inf。

### 优化如何产生

一次完成共享页布局寻址、padding跳过、边界/因果过滤以及紧凑输出，避免多次mask/gather/拼接。共享复制节省了重复数据，但必须有这步恢复每Q自己的合法范围，不能因为公共缓存包含末Q数据就让前Q看到未来。

它仍读取RAW并写一个新的分数矩阵，不是零HBM开销。本次没有再融合进DeepGEMM或最终TopK，也没有实测它单独比旧融合TopK更快。

### 输出

|变量|Shape|内容|
|---|---|---|
|OUT|`[R,WIDTH] float32`|紧凑候选顺序，非法位置为-inf|

这不是通用去重：它消除历史与固定sink/tail的重叠，仍要求输入历史页/成员自身唯一。当前函数只整理分数，不执行softmax。

## 17. map_topk：候选下标→逻辑位置→原生slot

源码：共享实验 `kernels.py`，函数 `map_topk`。

### 输入

|变量|Shape|含义|
|---|---|---|
|IDS|`[R,TOP] int64`|torch.topk输出的紧凑候选下标|
|VALUES|`[R,TOP] float32`|对应分数|
|LENS/END/TAKES|`[R]/[1]/[R]`|当前查询元数据|
|SELECTED/MEMBERS|`[R,MAXTAKE]/[CAP,64]`|历史候选的成员映射|
|NT|`[原生逻辑页容量] int32`|单请求共享原生页表|

### 实现与例子

一个program处理一个Q的TOP项（通常2048）。若rank<take×64，它在历史候选里：先选对应物理页，再读MEMBERS。否则按紧凑布局还原sink/tail逻辑位置。

沿用take=2、s=2、t=128：

- rank=65：来自第二个历史候选页的第1个成员，即MEMBERS[SELECTED[1],1]。
- rank=128：固定sink逻辑位置0。
- rank=130：tail第一项，逻辑位置128。

然后 `slot=NT[logical//64]*64+logical%64`。例如logical=130、NT[2]=9，则slot=9×64+2=578。

### 优化如何产生

把候选解码、成员映射、原生页表寻址和非法结果处理合在一个Triton kernel，不用多个PyTorch索引表达式及中间张量。没有再复制K；最终attention根据slot读取原生KV。

### 输出

|变量|Shape|内容|
|---|---|---|
|LOGICAL|`[R,TOP] int32`|原序列逻辑位置|
|SLOTS|`[R,TOP] int32`|原生物理slot|

VALUES为-inf或位置非法时输出-1。当前共享实验此函数**不包含C4的×4展开，也不包含TopK本身**；若模型需要展开，应由另一个接入步骤处理，不把旧fused.topk的展开能力写成本函数已有功能。

## 18. 哪些收益已经实测，哪些没有

本次H20，单K head，1K桶，4096页池，完整结果见 [RESULTS.md](RESULTS.md)。

|范围|模式|原实现 μs|优化后 μs|加速比|
|---|---|---:|---:|---:|
|initialize＋7轮分组|eager|559.568|264.122|2.12×|
|initialize＋7轮分组|Graph|89.330|55.012|1.62×|
|完整构建＋release|eager|655.075|353.898|1.85×|
|完整构建＋release|Graph|105.591|68.797|1.53×|

42次分组启动降到19次：四个大组轮次各4次，三个小组轮次各1次。还有initialize、assemble、页指纹、查重、分配和发布，不是整个建索引只有19次。

计时排除编译、预热、捕获和CPU输入上传；eager包含Python提交间隙。release用于重复试验复原池，不能当成生产纯构建时间。不能把这些倍数写成整条查询、attention或模型推理的倍数。

Tensor Core确实试过：TF32x3生成mma指令，但只有1～2个锚点，需要填充矩阵列宽，整轮N1024从19.571变23.931μs；数值微差还可能改变分组排列，因此未采用。cuBLAS的动态打包＋bmm＋整理也没有收益。**没有直接测试CUTLASS替换**。查询点积仍使用DeepGEMM；非矩阵计算不能因为使用Tensor Core就自然更快。

保留优化的正确性覆盖：随机、全零、重复向量；稳定排序、FLAG；指纹一致；碰撞后精确比较；发布字节；复用和两次释放；Graph更新源页表和桶起点。不是数学证明，也尚未做完整模型精度回归。

## 19. 源码、测试与阅读顺序

|内容|实现|测试|
|---|---|---|
|完整建索引调度|[runtime.py](runtime.py)|[test_build.py](test_build.py)|
|原initialize/assemble/页管理|[reference/kernels.py](reference/kernels.py)|test_build.py、bench_pages.py|
|原dots/anchor/sort|[reference/fused_grouping.py](reference/fused_grouping.py)|tune_group.py|
|新分组与页操作|[optimized.py](optimized.py)|[tune_group.py](tune_group.py)、[bench_pages.py](bench_pages.py)|
|首轮CUB排序|[sort_cub.cu](sort_cub.cu)、[sort_cub.py](sort_cub.py)|[test_sort_cub.py](test_sort_cub.py)|
|Tensor Core候选|optimized.py的dots_tc|[test_tensorcore_round.py](test_tensorcore_round.py)|
|route|[route_shared.cu](../fine_pool_shared_20261021/route_shared.cu)|[test_route_shared.py](../fine_pool_shared_20261021/test_route_shared.py)|
|stage_union|[tail_union.py](../fine_pool_shared_20261021/tail_union.py)|[test_and_bench.py](../fine_pool_shared_20261021/test_and_bench.py)|
|tables/normalize/map_topk|[kernels.py](../fine_pool_shared_20261021/kernels.py)|同上及route连接测试|
|共享精筛调用|[experiment.py](../fine_pool_shared_20261021/experiment.py)|同上|

建议先读runtime里的串接，再读anchored_dots/split_fused和首轮CUB；页管理按fingerprint→match→allocate→publish→release读；查询侧按stage_union→tables→normalize→map_topk读。

GPU Graph支持依赖预分配容量、捕获前编译和正确的流依赖。它不会自动把这些kernel融合，也不自动保证共享分配器并发安全；跨流overlap是调用层职责。文档中的源码链接指向各项功能的具体实现。
