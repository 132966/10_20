from common import ROOT
import json,hashlib

def load(n):return json.loads((ROOT/n).read_text())
def rows_table(header,rows):return '|'+ '|'.join(header)+'|\n|'+'|'.join(['---']*len(header))+'|\n'+''.join('|'+ '|'.join(map(str,row))+'|\n' for row in rows)
build=load('build_results.json');pages=load('page_results.json');tuned=load('tuned_group_results.json');tc=load('tensorcore_round_results.json');sort=load('sort_results.json')
text='''# V1 算子优化实测（独立实验，2026-10-22）

原 SGLang 和此前共享精筛实验未修改。这里复制原建索引/分组代码作为参考，提供可切换的1K桶构建适配器。输入仍为单个Index K head、128维、原生page_size=64；没有人为增加Q维度。查询打分涉及32 Q head，但本次没有替换查询DeepGEMM。

## 结论与范围

已验证的收益来自分组融合、按32位处理页面；Tensor Core和cuBLAS候选经过实测，没有作为默认替换。CUB稳定排序仅用于N=1024的第一轮，其余大组保留Triton排序；小组保持整轮融合。本次完整构建对照包含initialize、7轮分组、assemble、fingerprint、match_pages、allocate、publish以及末尾release；release是为了重复测量每次都新分配的生命周期，不能把该数字当作线上纯构建或整段推理耗时。

两个完整构建版本在同一张H20-3e上交替测量：eager包括Python提交间隙，Graph测重复捕获操作的GPU回放，编译/预热/捕获耗时排除。单算子/候选通常3轮中位数，完整构建5轮；未锁频。页面与候选实验分别在另一张空闲H20上运行，跨表绝对值不用于直接推导总时间。

## 完整构建结果

单位微秒；加速比为原实现/优化实现。

'''
rows=[]
for r in build['records']:
 for scope in ['initialize_and_grouping','build_and_release']:
  for mode in ['eager_us','graph_us']:
   a=r[scope]['original'][mode];b=r[scope]['optimized'][mode]
   rows.append([r['cap'],scope,mode,f'{a:.3f}',f'{b:.3f}',f'{a/b:.2f}×'])
text+=rows_table(['池容量页数','范围','模式','原实现','优化实现','加速比'],rows)
text+='''
## 15项算子的处理决定

|算子|适合的技术|本次处理|
|---|---|---|
|initialize|归约、读取和归一化|沿用；不属于有足够列宽的矩阵乘|
|_dots|向量点积、可尝试Tensor Core|大组把_anchor融合进下一次dots；小组整轮融合；Tensor Core单独实测并拒绝默认启用|
|_anchor|argmin归约|融合进后继点积或整轮分组，保持最左同分语义|
|_sort|排序，不适合cuBLAS/CUTLASS GEMM|小组融合；大组评估CUB稳定基数排序，按实测决定保留|
|assemble|gather、每8个K求均值、FP8打包|沿用；中心必须来自原始反量化K，不改变数值语义|
|fingerprint|整数计算和归约|fingerprint_words：一次加载4字节，分离64项成员归约，保留原两个int64摘要逐位一致|
|match_pages|扫描、字节一致性检查|match_words：比较2112个32位字而非8448个字节，成员数组单独64项检查；保留碰撞复核|
|allocate|空闲栈、引用计数|沿用；依赖分配器串行化，不能拿GEMM替代|
|publish|内存复制|publish_words：页与中心按32位复制，scale位模式不变|
|release|原子计数、前缀和|沿用，保持引用计数和空闲页回收语义|
|route|页面归约与TopK|沿用之前的物理页输出版本；不是矩阵乘，本次没有改它|
|stage_union|内存复制|沿用已用int32搬运的共享实现，没有再复制到第三个池|
|tables|整数地址与长度计算|沿用，保持动态输入与单精筛池接口|
|normalize|分数重排和掩码|沿用，不将过滤当成矩阵乘；未重新接入旧fused.topk|
|map_topk|索引映射|沿用，不做数值计算，不适合Tensor Core|

查询中心/候选打分已经使用DeepGEMM Tensor Core路径。本次没有实测替换成CUTLASS：它不是可直接代替排序、页管理、寻址操作的库，也未证明可超过当前查询后端。不能把“没有采用”解释为测过所有CUTLASS配置均无效。

## 分组融合实现

- 大组N=1024/512/256/128：原来3次dots＋2次anchor＋1次sort，改为第1次dots＋两次anchored_dots＋sort，共4次kernel。
- anchored_dots中，每个16成员tile读取所属组的分数，得到最左argmin，再做下一次点积；组内重复小规模argmin换取少两次启动。
- PREV和OUT必须是不同工作区，避免不同program之间读写竞争。两个额外FP32工作区各4096项，共32KiB。
- 小组N=64/32/16：split_fused把3次点积、两次锚点选择和稳定排序合成一个program/组。按实测使用4/16/8个warp。
- 不启用N>=128的完整融合：大矩阵长时间驻留会增加寄存器压力，N=1024的尝试约1ms，明显比原路径慢。
- 原始7轮共42个kernel；融合后为19个，另加初始化、组装和页管理。不改变桶大小、每页64成员或每中心8成员。

逐轮候选时间见tuned_group_results.json；表中为Graph微秒。

'''
rows=[]
for r in tuned:
 n=r['N'];pick='partial' if n>=128 else {64:'full_w4',32:'full_w16',16:'full_w8'}[n]
 a=r['timing']['original']['graph_us'];b=r['timing'][pick]['graph_us'];rows.append([n,pick,f'{a:.3f}',f'{b:.3f}',f'{a/b:.2f}×'])
text+=rows_table(['N','融合策略（未计后续CUB替换）','原实现','融合','加速比'],rows)
text+='\n## 页面操作结果\n\n同为4096页容量，Graph微秒；eager完整原始记录见page_results.json。单独kernel的eager耗时可能被提交开销淹没，不保证每一项都更快。\n\n'
rows=[]
for r in pages['records']:
 if r['cap']!=4096:continue
 a=r['timing']['original']['graph_us'];b=r['timing']['words']['graph_us'];rows.append([r['op'],f'{a:.3f}',f'{b:.3f}',f'{a/b:.2f}×'])
text+=rows_table(['操作','原实现','32位版本','加速比'],rows)
text+='''
## Tensor Core / cuBLAS候选为什么没有启用

`dots_tc` 使用FP32输入、TF32x3点积；编译出的PTX包含mma.sync，确认真实使用Tensor Core。每组只有1～2个锚点，点积的列宽很小，需要填充到16列，计算利用率低。

单独“重算两个点积”对照中Tensor Core约3.0微秒、原两个dots约3.5微秒，但原算法第二次点积已为选锚点计算过，不能据此宣称整轮分组加速。下面是包含三阶段dots、anchor和sort的真正一轮结果；更慢且存在数值顺序变化。

'''
rows=[]
for r in tc:
 a=r['timing']['original']['graph_us'];b=r['timing']['tensorcore']['graph_us'];rows.append([r['N'],f'{a:.3f}',f'{b:.3f}',r['mismatch_trials_out_of_10']['random'],r['mismatch_trials_out_of_10']['near_ties']])
text+=rows_table(['N','原一轮Graph μs','Tensor Core一轮Graph μs','随机样例排列变化/10','近同分样例变化/10'],rows)
text+='''
点积分数最大差异约6e-7～8e-7；这不等于已证明模型精度下降，但已经不满足本次严格保持原分组结果的目标。

cuBLAS候选使用pack_bmm收集动态成员和锚点，再torch.bmm，最后整理两个点积之差；总Graph耗时约7.6～8.4微秒，原两个dots约3.5微秒。Profiler确认实际调用cublas内核，见bmm_backend.json。打包成本不能提前忽略，因为每轮P和锚点都变化。默认float32最高精度、allow_tf32=False；不把这个结果推广为所有cuBLAS模式都慢。

## CUB排序候选

64位键直接照搬到CUB时，基数遍数较多，Graph下没有收益，归档在sort_64bit_results.json。进一步使用32位有序分数作为key、成员编号作为value，利用CUB稳定排序保持同分原顺序。原始P不可原地覆盖，FLAG停用语义保留。最终只在N=1024时采用CUB（Graph约10.77→7.22微秒）；N=512及以下CUB的Graph耗时更高，不采用。这样同一套配置同时支持eager和Graph，不以牺牲Graph性能换取CPU包装开销的下降。

'''
rows=[]
for r in sort:
 a=r['timing']['triton']['graph_us'];b=r['timing']['cub']['graph_us'];rows.append([r['N'],f'{a:.3f}',f'{b:.3f}',f'{a/b:.2f}×'])
text+=rows_table(['N','原Triton Graph μs','CUB候选 Graph μs','加速比'],rows)
text+='''
## 正确性验证

- 分组：随机排列、随机单位向量、全零和重复向量；锚点、差分分数、最终排列逐位核对。
- 排序：随机分数、同分、正负零、正负无穷；输入不覆盖、FLAG停用及Graph更新分数后回放。
- 指纹：随机字节及成员位置，两个int64指纹与原实现完全一致。
- 查重：空池、命中、相同指纹但不同字节，容量32/257/4096。
- 发布：全新页、部分复用、全部复用；页字节、中心字节、成员和指纹一致。
- 完整构建：随机FP8源页表，1K桶多次构建；第二lease复用页面、两次释放；同一Graph修改源页表和桶起点；inactive跳过。逐位核对重排K、中心、映射、指纹、页表、引用计数、空闲栈。
- 非数学形式证明，也没有做模型推理精度评测；没有修改原生SGLang或上线使用。

## 文件与使用

|文件|作用|
|---|---|
|optimized.py|分组融合、32位页操作、Tensor Core/cuBLAS试验kernel|
|runtime.py|保留原输入布局的预分配1K单head构建适配器，optimized开关切换|
|sort_cub.cu / sort_cub.py|CUB稳定排序候选及编译包装|
|reference/|原实现快照，来源和SHA见PROVENANCE.json|
|test_build.py|完整构建、复用、释放和动态Graph对照|
|bench_pages.py|页面正确性及基准|
|tune_group.py|分组候选正确性及调优|
|test_tensorcore_round.py|Tensor Core整轮性能及近同分检查|
|test_sort_cub.py|CUB稳定排序测试|

```python
from runtime import Build
# native: uint8[P_n,8448]；table: int32[16]；meta: int32[3]
plan = Build(native, table, meta, capacity=4096, records=2, optimized=True)
plan.run()  # 提交GPU构建，预分配状态写入plan.fine/centers/tables等
```

Graph捕获前先完成编译和预热；回放可以更新原生页表、源数据、桶起点、lease和active，但不得超过预分配容量。引用计数/空闲栈仍遵循原串行化要求；这里没有实现跨流调度或自动发布事件。`cycle()`只用于基准，构建后立即release，不应作为生产保留索引的入口。

环境：H20 SM90；使用原CUDA兼容环境和Torch/Triton依赖。所有更改在本目录，原文件校验和保持不变。
'''
(ROOT/'RESULTS.md').write_text(text)
# Check original sources remain byte-identical.
for entry in load('PROVENANCE.json').values():assert hashlib.sha256(__import__('pathlib').Path(entry['source']).read_bytes()).hexdigest()==entry['sha256']
print('Report written; original source checksums unchanged')
