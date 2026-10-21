# MLA Index 算子：输入、流程、例子与输出

本文件对应 [MLA_INDEX_KERNELS_SHARED.md](MLA_INDEX_KERNELS_SHARED.md) 中的15项，每项统一按照「输入表 → 处理流程 → 具体例子 → 输出表」说明，不逐行解释代码。

## 共同约定

- 单请求、1个 Index K head；涉及实际 Q 的测试使用32个 Q head。分组、页管理、route 本身不需要人为增加 Q head 维度。
- 建索引：1K桶，K维度128；每8成员一个中心；每64成员一个重排页。因此1024成员→128中心→16重排页。
- 原生/重排页每页8448字节：前8192字节是64×128个FP8值，后256字节是64个FP32 scale。不是每个token的128字节K后立即跟4字节scale。
- 同一块页内存的字节/FP8视图是 `[页数,8448]`；FP32视图是 `[页数,2112]`。FP8数值指针用于读取K数值，uint8用于字节指纹/复制；不能混用语义。
- 表格列有效范围；分组源码按每head 4096成员、256组的工作区步长寻址。单head的1K路径只使用前1024成员，A第一轮只使用第一行。有效shape不等于所有实际分配容量。
- 池容量P不是当前请求页数；lease是V1桶记录编号，不是原生物理页编号。
- 本文描述核心算子和独立共享精筛实验。原SGLang未修改。新版route适配在本目录，实际中心打分仍需调用方提供；测试不是完整模型推理。

## 文件入口

|简称|实现|语言|
|---|---|---|
|V|[原建索引/页管理 kernels.py](../../sglang/srt/layers/attention/v1_deepseek/kernels.py)|Triton|
|G|[fused_grouping.py](../../sglang/srt/layers/attention/v1_native/fused_grouping.py)|Triton|
|C|[原 v1_query.cuh](../../sglang/kernels/jit/csrc/deepseek_v4/v1_query.cuh)|CUDA C++|
|R|[route_shared.cu](route_shared.cu)、[route_shared.py](route_shared.py)|CUDA C++＋Python封装|
|S|[tail_union.py](tail_union.py)|Triton|
|E|[kernels.py](kernels.py)|Triton|

输入表中的“输入/输出”表示原地更新；输出缓冲区即使没有列在输入表，也都是调用方提前分配后传入的指针。


## 1. initialize

实现：V。准备归一化分组数据，不修改原生K。

### 输入表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|K|[P_n,8448]，FP8视图|原生K池中的FP8数值|
|KS|[P_n,2112]，FP32视图|同一内存的scale读取视图|
|TABLE|[16]，int32|本桶源页，例如[8,31,5,…]|
|META|[3]，int32|[4096,7,1]；本算子只用active|

### 处理流程

```text
TABLE [16] → 定位原生页
→ FP8 K [1024,128]＋scale [1024]
→ 反量化K [1024,128]
→ 各行除以自身范数 → UNIT [1024,128]
同时：生成PERM [1024]、设置FLAG [1]
```

### 具体例子

桶内成员0从原生页8偏移0读取。若K=[3,4,0,…]、scale=0.5，反量化为[1.5,2,0,…]，范数2.5，输出UNIT[0]=[0.6,0.8,0,…]。PERM[0]=0。

### 输出表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|UNIT|[1024,128]有效范围，FP32|后续分组使用的单位向量|
|PERM|[1024]有效范围，int32|[0,1,…,1023]|
|FLAG|[1]，int32|active非0写-1；否则写0|

active=0时不更新UNIT/PERM。FLAG是执行开关，不是完成事件。一个program处理16成员，1K使用64个program。


## 2. _dots

实现：G。根据MODE计算锚点点积，或两个锚点的相似度差。

### 输入表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|U|[1024,128]有效范围，FP32|initialize输出的UNIT|
|P|[1024]有效范围，int32|当前成员排列|
|A|第一轮[1,2]有效范围，int32|每组[b,a]；MODE=0不读取|
|S|[1024]有效范围，FP32，输入/输出|MODE=2读取原有的对b点积|
|FLAG|[1]，int32|小于0执行|
|N / START / MODE|整数|第一轮N=1024；本路径START=0；MODE取0/1/2|

### 处理流程

```text
P → 在U中找到成员向量
MODE=0：与当前组P首成员点积 → S → _anchor选b
MODE=1：与b点积 → S → _anchor选a
MODE=2：与a点积 − S中已有的与b点积 → S
```

### 具体例子

设a=[1,0,…]，b=[-1,0,…]，成员x=[0.6,0.8,0,…]。MODE=1写S=-0.6；MODE=2写S=0.6−(-0.6)=1.2。

### 输出表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|S|[1024]有效范围，FP32|MODE=0/1供_anchor；MODE=2供_sort|

S[i]对应P[i]，不是永远对应成员i。单head、1K共64个program，每个16成员。U/P/A不修改。


## 3. _anchor

实现：G。每组选分数最小的成员编号，写入指定锚点列。

### 输入表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|P|[1024]有效范围，int32|成员排列|
|S|[1024]有效范围，FP32|各排列位置的点积分数|
|FLAG|[1]，int32|小于0执行|
|N / SLOT|整数|N=1024；SLOT=0写b，SLOT=1写a|

### 处理流程

```text
S按组查找最小分数位置pos
→ P[pos]得到成员编号
→ 写A[group,SLOT]
```

### 具体例子

P前四项=[7,2,9,5]，S前四项=[0.8,-0.3,-0.9,0.2]，其余分数更大。pos=2，P[2]=9，所以SLOT=0时A[0,0]=9，而不是2。

### 输出表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|A|第一轮[1,2]有效范围，int32，部分更新|本次只写b列或a列，供后续_dots读取|

同分选当前排列最靠左的位置。第一轮1个program；第二轮N=512时2个program，各组独立选锚点。


## 4. _sort

实现：G。按组内分数稳定降序排列成员ID。

### 输入表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|P|[1024]有效范围，int32|输入成员排列|
|S|[1024]有效范围，FP32|对a点积减对b点积|
|FLAG|[1]，int32|小于0执行|
|N|整数|第一轮1024，之后512、256等|

### 处理流程

```text
P＋S → 每组按分数降序、同分保持原位置顺序
→ O新排列
→ 下一轮将组大小减半，前后各作为一个子组
```

### 具体例子

缩小为8成员：P=[7,2,9,5,1,8,3,6]；S=[0.2,0.9,-0.5,0.9,-0.8,0.3,0,-0.2]。输出O=[2,5,8,7,3,6,9,1]。同分成员2仍在5前。下一轮两组分别[2,5,8,7]和[3,6,9,1]。

### 输出表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|O|[1024]有效范围，int32|排序后的成员排列，作为下一轮P|

只排序ID，不搬128维K；P和S不变。按人数平分，不按正负号分组。第一轮1个program处理1024成员。


## 5. assemble

实现：V。按最终排列生成临时重排页、中心和逻辑映射。

### 输入表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|K / KS|[P_n,8448] FP8 / [P_n,2112] FP32|原生K和scale的同内存视图|
|TABLE|[16]，int32|桶的原生页表|
|META|[3]，int32|[4096,7,1]，读取桶起点与active|
|IDS|[1024]有效范围，int32|最终成员排列|

### 处理流程

```text
IDS → TABLE → 读取原生K和scale
├ 按排列复制 → TMP＋TMPS
├ 桶起点＋IDS → MAP [16,64]
└ 每8个K反量化求平均 → 128个中心 → FP8量化 → CENTERS＋CS
```

### 具体例子

IDS前8项=[7,2,9,5,1,8,3,6]，对应原生页TABLE[0]中的这些偏移。MAP[0,:8]=[4103,4098,4105,4101,4097,4104,4099,4102]。若这8个反量化向量分别为[1,0,…]到[8,0,…]，中心为[4.5,0,…]，随后量化存储。

### 输出表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|TMP / TMPS|[16,8448] FP8 / [16,2112] FP32视图|同一临时页内存的重排K及原scale|
|CENTERS / CS|[2,8448] FP8 / [2,2112] FP32视图|128个临时中心及各自scale|
|MAP|[16,64]，int32|原序列逻辑位置|

中心来自原生K反量化均值，不是UNIT均值。中心scale采用二的幂，按max(abs(mean))/448向上取整（带下限）。grid=(16,8)，每program处理8成员并生成1中心。这里尚未发布到最终池。


## 6. fingerprint

实现：V。对每页字节和成员映射生成两个加权摘要。

### 输入表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|TMP|[16,8448]，uint8字节视图|临时重排K＋scale完整字节|
|MAP|[16,64]，int32|成员逻辑位置|
|META|[3]，int32|只读取active|

### 处理流程

```text
每页8448字节＋64成员位置
→ 两种加权求和
→ HASH [16,2]
```

### 具体例子

简化手算页：字节前两项[1,2]、成员前两项[10,20]，其余全0。a=1×1+2×2+(10+20)×104729=3141875。b=(1×7+2×8)×1000003+(10×1+20×2)×65537=26276919。HASH[0]=[3141875,26276919]。

### 输出表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|HASH|[16,2]，int64|每页两个指纹，供match_pages预筛选|

16个program各处理一页。这里只按字节整数处理，不做FP8解量化；摘要可能碰撞，必须再核对完整内容。


## 7. match_pages

实现：V。先匹配指纹，再逐字节和逐成员比较，寻找可复用页。

### 输入表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|TMP / MAP / HASH|[16,8448] uint8 / [16,64] int32 / [16,2] int64|当前16个临时页|
|META|[3]，int32|active开关|
|FINE|[32,8448]，uint8|本例重排池容量32页|
|MEMBERS|[32,64]，int32|池中页的逻辑成员映射|
|HASHES|[32,2]，int64|池中页的指纹|
|REFS|[32]，int32|引用计数，>0表示可参与查重|
|CAP / SCAN|整数|CAP=32实际池容量；SCAN=32扫描向量长度，通常取覆盖CAP的2次幂|

### 处理流程

```text
临时页指纹 → 扫描池中refs>0且指纹相同的页
→ 按物理页编号从小到大逐个核对完整K/scale和MAP
→ 完全相同则输出物理页编号
→ 全部不匹配则输出-1
```

### 具体例子

临时页0与池页7、20指纹相同，但页7内容不同、页20内容完全相同：MATCH[0]=20。临时页1没有完全相同页：MATCH[1]=-1。

### 输出表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|MATCH|[16]，int32|如[20,-1,7,…]；已有页编号或未命中-1|

16个program，每个处理一个临时页并扫描CAP个槽。good可能同时有多个true。未用池槽refs=0不参与。


## 8. allocate

实现：V。为16个临时页确定最终物理页，更新页管理状态。

### 输入表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|META|[3]，int32|本例lease=7、active=1|
|MATCH|[16]，int32|非负表示复用；-1需要新页|
|FREE|[CAP]，int32，输入/输出状态|空闲物理页栈，有效前COUNT项|
|COUNT|[1]，int32，输入/输出|空闲页数量|
|REFS|[CAP]，int32，输入/输出|各页引用计数|
|TABLES|[L,16]，int32，输入/输出|每个桶记录对应16个物理页，L可取128|

### 处理流程

```text
MATCH → 标出fresh → 统计需要多少新页
→ 从FREE栈顶领取新页，命中的直接复用
→ 每个目标页引用计数+1
→ 写TABLES[lease]及NEW，减少COUNT
```

### 具体例子

假设只有临时页1未命中，MATCH开头=[20,-1,7,…]。COUNT=3，FREE有效项=[4,11,9]。新页领取9：TABLES[7,:3]=[20,9,7]，NEW[:3]=[0,1,0]，COUNT变2，相关REFS各加1。

### 输出表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|TABLES[7]|[16]，int32|本桶最终物理页表|
|NEW|[16]，标记缓冲区（常用int32）|1表示新分配、需要发布页数据；0表示复用|
|REFS / COUNT|[CAP] / [1]，int32|引用计数增加、空闲数量减少|

一个program处理16页。FREE内容没有清空，COUNT缩小后栈顶旧值不再有效。COUNT读取/写回不是并发安全的批量领取；调用方必须串行化共享分配器操作，不能仅因REFS使用原子操作就任意并发。


## 9. publish

实现：V。把临时结果写入分配好的池；中心按桶记录独立存储。

### 输入表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|TMP|[16,8448]，uint8|临时重排页|
|TC / TCS|[2,8448] uint8 / [2,2112] FP32视图|临时128中心的同一块存储|
|MAP / HASH|[16,64] int32 / [16,2] int64|临时成员和指纹|
|META|[3]，int32|桶起点、lease=7、active|
|TABLES|[L,16]，int32|allocate确定的目标物理页|
|NEW|[16]，标记|是否需要复制该重排页|

### 处理流程

```text
NEW[p]=1 → 按TABLES[lease,p]复制TMP、MAP、HASH到池
NEW[p]=0 → 已有页复用，不重复写页数据
无论是否复用 → 把每页对应8中心写入lease专属中心区域
```

### 具体例子

TABLES[7,0]=20且NEW[0]=1：TMP[0]写入FINE[20]。该页8中心不写“中心页20”，而写全局中心编号7×128+[0..7]=896..903，即中心物理页14的前8项。整个lease7使用中心页14、15。

### 输出表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|FINE|[P_r,8448]，uint8，部分更新|新增重排页的K＋scale|
|MEMBERS / HASHES|[P_r,64] int32 / [P_r,2] int64，部分更新|新增页成员映射、指纹|
|CENTERS / CS|[2L,8448] uint8 / [2L,2112] FP32视图|按1K桶记录容量L计算的中心池，写本lease的128中心|

16个program各处理一个重排页。TC/TCS和CENTERS/CS分别是同内存视图：一次复制FP8中心字节，另一次复制scale。源码旧注释提到8中心页，与当前每lease128中心的地址计算不符；本说明按实际1K代码的2页解释。完成事件和请求页表发布仍由运行时保证。


## 10. release

实现：V。释放一个桶对16个重排页的引用，把不再使用的页放回空闲栈。

### 输入表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|TABLE|[16]，int32|要释放的桶物理页表|
|REFS|[CAP]，int32，输入/输出|物理页引用计数|
|FREE|[CAP]，int32，输入/输出|空闲页栈|
|COUNT|[1]，int32，输入/输出|当前空闲页数量|

### 处理流程

```text
TABLE → 各页引用计数减1
→ 旧计数为1的页现在变0
→ 统计这些页数量并领取FREE写入区间
→ 将这些页编号追加到空闲栈
```

### 具体例子

TABLE前两项=[20,9]，REFS[20]=2、REFS[9]=1，其余页减后仍>0。COUNT原为3。释放后REFS[20]=1不回收，REFS[9]=0回收：FREE[3]=9，COUNT=4。

### 输出表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|REFS|[CAP]，int32，部分更新|本桶涉及页的引用计数减1|
|FREE|[CAP]，int32，部分更新|栈尾新增零引用页编号|
|COUNT|[1]，int32|空闲数增加|

一个program处理16页；TABLE不清空，FINE内容也不擦除。不能重复释放同一份引用。release本身不释放CPU桶记录或管理中心lease，运行时另行处理。


## 11. fused.route：物理页输出适配

实现：R（源逻辑来自C）。沿用每页8中心max和CUB页面排序，输出与tables一致的物理页编号。

### 输入表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|scores|[R,P×8]，FP32|中心打分结果，32个Q头已经在前面的打分阶段汇总；例如R=1024、P=64|
|published|[R]，int32|每Q实际发布且允许参与路由的请求页表前缀项数；不能从可见长度猜|
|budgets|[R]，int32|每Q请求选页预算，例如40|
|rt|[R,P]，int32|请求页序号→重排池物理页；同一请求可按行重复|

### 处理流程

```text
scores每8个中心取max → 每页一个分数
→ 只在published有效前缀里选页
→ 选取数量=min(非负预算,有效页数,输出容量M)
→ 用rt把选中页序号转换为物理页
→ selected [R,M]＋takes [R]
→ 直接作为tables的SELECTED＋TAKES
```

### 具体例子

某行rt=[20,35,9]，页面分数=[0.2,0.9,0.7]，published=3、budget=2。选中页序号[1,2]，新输出物理页[35,9]，takes=2。若published改成1，即使后两页分数更高，也只能输出[20,0]、takes=1。

### 输出表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|selected|[R,M]，int32|前takes项为重排物理页；其余写0，必须结合takes使用|
|takes|[R]，int32|截断到有效范围后的实际选页数量|

适配不写旧owner尾部页表；统一交给tables组织共享页。原版selected是页序号，原版可选ft分支才做rt映射，不能把原selected直接传新tables。新接口的published和rt必须与END属于同一个已发布快照；有效候选前缀由调用方保证。每Q一个CUDA block。take等于全部有效页数时直接输出请求页表顺序，不保证按分数排序；TopK集合语义不变。当前封装容量支持1..4096页（256K个Index K位置，对应C4下约1M原始token），保留静态共享内存，不包含此前尝试的16384页动态共享内存扩展，中心分数要求有限值或-inf掩码，不接受NaN。


## 12. stage_union

实现：S。整个chunk的公共sink＋tail页只复制一次。

### 输入表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|SRC|[P_n,8448] uint8存储，kernel按int32视图读取|原生K＋scale完整页面|
|NT|[L_n]，int32|单请求逻辑页→原生物理页|
|LENS|[4096]，int32|[65537,…,69632]，非负且单调不减|
|INDEXED_END|[1]，int32|[65536]，已发布连续索引前缀终点，不含|
|R / S / W|整数|4096个Q；sink64；window2048|
|B / PAGE|编译参数|例如512 / False，控制复制粒度|

### 处理流程

```text
LENS＋INDEXED_END → 各Q精确sink/tail范围
→ 整个chunk公共范围按64位置向外覆盖为完整页
→ NT寻址原生页 → 复制K和scale字节到DST一次
→ 同时写META和DESC
```

### 具体例子

Q0需要sink[0,64)、tail[63489,65537)；末Q需要sink[0,64)、tail[65536,69632)。公共逻辑页为0和992..1087，共97页。DST[0]对应逻辑页0，DST[1]对应992，DST[33]对应1024。META首行=[64,63489,2112]，末行=[64,65536,4160]。

### 输出表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|DST|本例预留[99,8448] uint8，实际前97页有效|公共页副本，是fine池后缀的视图；不是独立的第二个精筛池基址|
|META|[4096,3]，int32|每Q的[sink数量,tail起点,固定候选数量]|
|DESC|[4]，int32|[1,992,1088,97]：sink页数、tail起始逻辑页、结束逻辑页不含、总有效页数|

原样复制，不做反量化。尾部必须包含未发布索引的可见token；不代表这些token全都通过最终TopK。META是描述输出，当前tables/normalize会从LENS/END重算范围，没有直接消费META。各Q还是独立打分，复用的是HBM数据副本。


## 13. tables

实现：E。组合历史物理页和共享物理页，输出精筛页表及长度。

### 输入表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|LENS / END|[R] / [1]，int32|因果可见长度与实际发布终点|
|TAKES / SELECTED|[R] / [R,M]，int32|实际历史页数及物理页编号；可直接接新route输出|
|DESC|[4]，int32|stage_union公共布局|
|CAP / MAXTAKE / FP|整数|历史池预留容量、M、每Q精筛页表容量|
|S / W|整数|sink和window|
|SHARED / B / SCRATCH|编译参数|True；页表处理宽度；共享分支不使用SCRATCH|

### 处理流程

```text
SELECTED有效前TAKES项 → FT前缀
LENS＋END → 每Q sink/tail范围
DESC＋CAP → 对应公共缓存物理页 → FT后缀
计算按页打分跨度DGLENS，以及紧凑候选长度CLENS
```

### 具体例子

小例子：CAP=10，公共页10存sink0..63、页11存tail128..191、页12存192..255。两Q的SELECTED=[[3,7],[5,7]]，TAKES=[2,2]，LENS=[193,194]，END=128，S=W=64。输出FT=[[3,7,10,11,12],[5,7,10,11,12]]。两行DGLENS/CLENS分别257、258，因为末页分别只需要1、2项。

非对齐例子：tail需要129..192共64项，但页首是128。打分覆盖128..192共65项，因此历史128＋sink64时DGLENS=257，CLENS=256；normalize去掉多带的128。

### 输出表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|FT|[R,FP]，int32|同一个fine池中的物理页表；未用项填安全页0|
|DGLENS|实验实际分配[R,1]，int32；逻辑上每Q一个数|DeepGEMM打分跨度，最少填1以满足后端；不是最终有效候选数|
|CLENS|[R]，int32|TAKES×64＋精确sink/tail数，不含页对齐空隙|

CAP不是已用页数或选页预算。真实4K例子FP=108，Q0的DGLENS=4673、CLENS=4672，末Q均6720。CLENS不扣除历史部分将被置-inf的成员；当前Runner计算它但不传给normalize，后者自行重算边界。


## 14. normalize

实现：E。整理分数并屏蔽不合法候选，不是softmax或向量归一化。

### 输入表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|RAW|[R,STRIDE]，FP32|DeepGEMM原始输出，可有行尾预留空间|
|LENS / END|[R] / [1]，int32|每Q可见长度、实际发布终点|
|TAKES / SELECTED|[R] / [R,M]，int32|历史候选数、历史物理页|
|MEMBERS|[P_r,64]，int32|历史页成员逻辑位置|
|MAXTAKE / WIDTH / STRIDE|整数|SELECTED行容量、OUT行容量、RAW实际行跨度|
|S / W / B / SHARED|编译参数|sink、window、处理块宽、共享路径True|

### 处理流程

```text
历史候选：由MEMBERS取逻辑位置
→ 仅保留sink之后、tail之前、可见且已发布的成员
公共候选：跳过整页复制带来的前后边界空隙
→ 输出[历史固定槽位][精确sink][精确tail]
→ 非法和行尾预留位置写-inf
```

### 具体例子

教学缩小为每页4项：历史成员[0,5,9,11]，sink[0,2)、tail[9,11)、可见长度11、END=9。历史部分仅位置5合法，输出[-inf,score(5),-inf,-inf]；固定部分接[score(0),score(1),score(9),score(10)]。历史位置0、9虽被过滤，仍各自通过固定部分参与一次。实际实现页大小固定64，缩小例子只解释布局。

### 输出表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|OUT|[R,WIDTH]，FP32|紧凑固定候选布局的分数，非法项-inf，供最终TopK|

不会把所有无效历史槽位压缩删除；只恢复固定候选布局并写掩码。不搬K、不改MEMBERS、不做通用重复ID去重。


## 15. map_topk

实现：E。将TopK候选下标还原为逻辑位置，再映射到原生物理slot。

### 输入表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|IDS / VALUES|[R,2048]，int64 / FP32|实验torch.topk输出的候选下标和分数|
|LENS / END|[R] / [1]，int32|可见长度、已发布终点|
|TAKES / SELECTED|[R] / [R,M]，int32|历史页数及物理页|
|MEMBERS|[P_r,64]，int32|重排成员逻辑位置|
|NT|[L_n]，int32|单请求原生页表|
|MAXTAKE / TOP / S / W / B|编译参数|页预算容量、2048、sink、window、处理块宽|

### 处理流程

```text
IDS候选下标
├ 历史部分 → SELECTED物理页 → MEMBERS逻辑位置
├ sink部分 → sink内偏移
└ tail部分 → tail起点＋偏移
→ LOGICAL
→ NT[logical//64]×64＋logical%64
→ SLOTS
```

### 具体例子

TAKES=2、sink64、tail起点1000。候选0..127是历史，128..191是sink，192起是tail。IDS=[65,130,194]，SELECTED第二页=7且MEMBERS[7,1]=900，得到LOGICAL=[900,2,1002]。若NT[14]=31、NT[0]=8、NT[15]=5，SLOTS=[1988,514,362]。

### 输出表

|变量|Shape / dtype|含义与例子|
|---|---|---|
|LOGICAL|[R,2048]，int32|原序列逻辑位置，非法为-1|
|SLOTS|[R,2048]，int32|原生物理slot，非法为-1|

不搬K/V、不计算attention，不包含C4的×4展开。VALUES=-inf的项输出-1。最终attention后端必须使用相同逻辑序列/原生页表语义。


## 整体关系与本次适配边界

```text
建索引：initialize → 多级_dots/_anchor/_sort → assemble
      → fingerprint → match_pages → allocate → publish
回收：release

查询：中心打分（外部DeepGEMM）
   → 新route（8中心max＋选页＋物理编号转换）
   → stage_union＋tables
   → 动态调度元数据＋DeepGEMM精筛
   → normalize → torch.topk → map_topk
   → 原生attention（本实验未接入）
```

共享缓存与route是不同职责：route负责选历史页，stage_union负责公共固定候选，tables拼接两者。

旧 `fused.topk` 没有淘汰，但尚未在这里接入；此处实际执行 `normalize → torch.topk → map_topk`，不能称为三步已经融合。本文也不把C4展开描述为已完成。

原route有效页数按lens推算，输出selected是请求页序号；新适配接收实际published前缀页数，selected直接输出物理页，另输出实际takes。调用方必须保证published、rt、END与已发布数据一致。Graph可以更新这些设备数组，但不会自动修复错误的发布状态。

## 测试与复现

测试脚本：[test_route_shared.py](test_route_shared.py)。结果：[route_test_results.json](route_test_results.json)，日志：[route_test.log](route_test.log)。

测试包括：非恒等页表、预算为0或超过有效页数、发布页数为0、容量非2次幂、同分、padding、1024个Q，以及同一Graph回放改变中心分数、预算、发布页数、已发布索引终点和请求页表。容量分支覆盖1、17、64、129、1024、3904、4096页。还将新route输出接到共享精筛链路，与逐Q复制基线对比分数、TopK逻辑位置和物理slot。Q shape为[1024,1,32,128]，1个K head、32个Q head。

中心打分输入为测试构造，并未运行完整模型粗筛或SGLang推理；不据此报告新的端到端加速比。原共享缓存性能数字也未重新计入route耗时。

环境复现：

```bash
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/sglang_sparse/activate_cuda_compat.sh
export PYTHONPATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/sglang_sparse/sglangenv/lib/python3.12/site-packages
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=2
/usr/local/bin/python3.12 test_route_shared.py
```

先完成CUDA扩展编译和warmup，再捕获Graph。源码运行位置必须含route_shared.cu、route_binding.cpp、route_shared.py以及原共享实验文件。
