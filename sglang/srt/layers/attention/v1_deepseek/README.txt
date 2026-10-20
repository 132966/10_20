DeepSeek FP8/page64 V1 索引实现（单卡 SM90）

本目录与旧 Qwen v1_native 独立。只重排 Index K，不复制 MLA attention KV。
eager 与 CUDA Graph 使用相同算子；graphs=False 可关闭内部捕获。

接口
1. Indexer(native_cache, Config(...))：每层绑定独立固定容量缓存。
   native_cache 为 uint8 [P_n,64,1,132] 或 [P_n,8448]。
   注意实际每页布局：8192 字节 FP8 K，随后 64 个 FP32 scale；
   不是逐 token 交错的 128+4 字节。
2. prepare(owner, native_table, length)：建立尚未完成的完整 4K 桶。
   每次建桶只读取该桶的 64 个页编号；物理页可以分散在整个原生池。
   图内输入：source[64]、meta[3]={逻辑桶起点,桶记录号,active}，
   固定地址的原生缓存、V1 Arena 和 GPU 分配状态。
   产生 64 个重排页，每页 64 个成员、8 个中心；成员保存逻辑位置。
   新页从 GPU 空闲栈分配；共享页以指纹筛选后逐字节/成员精确比对，引用归零回收。
   中心缓存为 [ceil(P_r/8),64,1,132]，持久精筛缓存为 [P_r,64,1,132]。
   每个桶记录固定对应 8 个连续中心页；共享重排页和桶中心记录分别寻址。
   中心页表由运行时根据桶记录维护，不要求调用者额外提供。
3. query(q, weights, native_tables, lengths, owners, lengths_cpu,
         return_logical=False)：自动 prepare，然后查询。
   q: FP8 [T,H,128]；weights: FP32 [T,H]，H 可为 32 或 64，已合入 Q scale。
   native_tables: int32 [T,L_n]，每行对应查询所属请求的当前原生页表。
   lengths: GPU int32 [T] 或 [T,1]；lengths_cpu 是相同长度的调度器 CPU 信息。
   owners: 每行的 (请求槽位, generation)，不是 token slot。
   输出 int32 [T,topk] 当前原生物理 slot；无有效候选时填 -1。
   return_logical=True 返回 (物理 slot, 逻辑位置)。
   CPU/GPU 长度一致由调用者保证；正常查询不为检查它而读取 GPU 回 CPU。

查询流程
按可见长度计算完整桶数；Q 对中心打分，每页取 8 个中心中的最大分数；
选候选重排页；对候选 Index K 精筛，同时加入原生 sink 和未建桶尾部/窗口；
过滤重复及不可见成员；原生 TopK-v2；通过当前原生页表转换位置。
默认 weighted ReLU 分数与 DSA 数学规则一致；score_mode='signed' 是显式可选项。
中心从原生 FP8 K 反量化构建，不宣称与旧 BF16 Qwen 索引相同。

Graph
每层按需捕获 1 张建桶图、每个 Q 档位 1 张查询图。
同层后续 4K 桶复用建桶图；K 长度增长不重新捕获查询图。
不同层当前各自绑定缓存地址，不是所有层共享同一个 CUDAGraph 对象。
跨流用 CUDA event 排序，返回结果有独立存储，避免下一次回放覆盖。
默认 ReLU 粗筛和精筛均使用原生分页 DeepGEMM，
TopK 使用 SGLang 原生 v2 的实现。粗筛/精筛 DeepGEMM 调度表在查询图内由实时长度计算；
小 Q 路径更新 TopK 计划，大 Q 的融合 register/streaming 路径不需要 cluster 计划。
精筛页表直接引用选中的重排页；原生 sink/尾部/窗口写入预留暂存页。
默认 query_optimizations=true：
- 容量不超过 4096 个重排页时，使用迁入的 CUB 路由，融合每页 8 个中心归约和候选页选择；超出容量保留原路径。
- Q 档位大于 512 时，融合原生 TopK-v2 与逻辑/当前原生物理位置映射；小 Q 保留原生 cluster/plan 路径。
- 每次回放在 GPU 比较当前长度及 sink/尾部涉及的全部物理页编号；与第 0 行相同的行共享一次暂存复制，否则使用自己的暂存区域。
  不按旧 slot 身份缓存结果，不假定页表或有效长度不变；共享只在本次查询工作区内生效。
- ReLU 且候选比例为 1 时，直接执行原生全量分页 Index 查询，跳过中心打分和候选重排。
- 合并长度和缓存共享元数据生成；请求长度的 CPU 检查使用普通整数运算。
设置 query_optimizations=false 可使用未融合查询路径作对照；不改变建索引算法、P 预算或打分规则。
排序实现对完全相同的中心分数可能采用不同的同分顺序，不能宣称同分候选的 ID 逐位相同。
暂存页附加在精筛缓存分配尾部，不属于可共享的持久重排页，不参与引用计数。
预留上限为 68 * 最大 Q 档位个页（65 个数据页加 tile 边界余量）。signed 模式仍使用独立 Triton 粗筛/精筛算子。

C4 模型接入（显式可选）
设置 SGLANG_V1_DEEPSEEK_CONFIG 为 JSON，字段为 Config 参数，另需
总层预算 total_budget_bytes。例：
{"max_context":65536,"max_requests":1,"q_buckets":[1,4,16,64],
 "graphs":true,"topk":512,"total_budget_bytes":17179869184}
此例只适合原始上下文不超过 262144 的配置；max_context 按 C4 压缩位置计数。
总预算必须覆盖所有本地 C4 层的保守预留；启动时检查预算和空闲显存。
这不是原生 KV pool 的自动缩容机制，须通过 mem-fraction-static 留出显存，
还要给其他运行期临时张量保留余量。预算上限不意味着立即分配全部空间。

启动需 --disable-cuda-graph --disable-prefill-cuda-graph；它们关闭外层模型图，
本模块自己的建桶/查询子图仍由 graphs=true 开启。
接入口在 dsv4/indexer.py：原生 Q/K 准备后调用 C4Adapter，结果写回原生
c4_sparse_page_indices；其后的 MLA attention 仍为原生实现。
ForwardBatch 传递 CPU 请求槽位，adapter 从 CPU 阶段/长度计算每行归属。
C4 模型接入从配置读取 Index Q 头数（原生默认 64，独立示例默认 32）。
C4 查询长度沿用当前原生 c4_topk_lengths_raw=seq_len//4，可为 0。
请求池 listener 处理正常结束/abort 和 clear；不同 generation 不复用旧请求状态。
每次访问原生缓存仍调用 accessor，保留原生恢复等待；原生地址变更要求 flush。

限制
模型接入口只支持单卡、FP8、非推测解码、无 HiSparse。
尚未实现嵌入外层完整模型 CUDA Graph、FP4、TP/PP、投机树验证。
传统 DSA 可直接使用独立 Indexer 算子接口；目前自动模型 hook 只接 C4。
共享前缀必须不可变；历史数据原地更改须先 release_request/clear。
合成测试和接口测试不等于 DeepSeek V4 整模型服务、输出质量或 HiCache 组合验证。

验证入口（项目根目录）
bash run_v1_deepseek_test.sh run_v1_deepseek_validation.py
覆盖 eager/Graph、原版 DeepGEMM/FP32、Top512/2048、FP8 位级保留、
紧凑中心、共享引用、generation、原生页迁移、回退、flush、C4 接口和 >512K。
性能脚本包括 CPU 调度/输入复制/输出复制，不能把共享繁忙 GPU 的结果当作空卡性能。

查询优化验证：test_v1_deepseek_query_optimized.py、test_v1_deepseek_tail_replay.py、
test_v1_deepseek_large_offsets.py；完整组合入口 run_v1_query_optimized_suite.py。
性能使用旧实验实际 selected_pages、真实 K/4096 个真实 Q；纯算子与含接口开销结果分别保存。

2026-10-16 L3 / Q1024 查询改进
- 复用旧 CUB router 的 float4 中心加载，保留排序规则和选页预算。
- 大 Q 的 TopK 内完成 sink/tail 边界过滤；每页属于同一4K桶，内部桶
  无须读取64个成员，边界桶仍逐项检查。不修改分组和候选数量。
- 大 Q 共享尾部以32行分组调度，减少空执行块；混合长度/页映射仍
  逐行判断实际所有者。小 Q 保留原调度网格。
- 原地 DSA 调度为实验选项，SM90 实测每次约增加0.1ms，因此默认关闭；
  默认继续使用原 DeepGEMM 调度，长度变化仍在每次回放生效。
- query_mask_fusion、query_tail_compaction、query_inplace_metadata（默认关闭）可独立配置；
  query_optimizations=False 保留未融合查询路径。
测试入口：run_v1_l3_q1024_final.py；结果：V1_L3_Q1024_RESULTS_20261016.txt。
该结果仅测试 L3，不能据此外推全层平均性能。

2026-10-16 查询输出优化补充
- expand_output=False 为默认，返回当前原生缓存的物理 TopK，不改变 C4 接口。
- 显式 expand_output=True 时返回 int32 [T,4*topk+64]：逻辑压缩位置 p
  展开为 4p..4p+3，无效位置和末尾64项为 -1；不是物理页号展开。
  return_logical=True 同时返回未展开的逻辑 TopK。
- Q>512 且精筛容量不超过32768时，在融合 TopK 尾部直接写出展开结果；
  其余路径用单次 Triton 展开，工作区预分配，eager/Graph 共用同一逻辑。
- query_route_table_fusion 支持选页时直接生成精筛页表；L3实测无稳定收益，
  默认关闭，仅保留可选对照。原地调度 query_inplace_metadata 仍默认关闭。
- 同工作区配对实测展开查询：64K 0.4979→0.4725ms；976K 2.8910→2.8677ms。
  不能用不同引擎/地址之间的差值单独推断某个融合算子的收益。
验证：test_v1_output_route_fusion.py（混合长度、增长、回退、空行、Top512/2048、
同图页表变化），test_v1_deepseek_tail_replay.py。
性能：bench_v1_l3_q1024_fusions_final.py；配对消融：bench_v1_expansion_same_workspace.py。
结果：V1_L3_Q1024_FUSION_RESULTS_20261016.txt。仅L3、Q1024、旧预算、查询部分。

选页/精筛页表融合经复测无稳定收益，按用户要求恢复默认关闭。
query_route_table_fusion=False：选页后独立生成精筛物理页表；其他优化保持不变。

2026-10-16 按旧版方式裁剪展开输出映射
query_output_mapping_optimization=True（默认）且expand_output=True时，稀疏查询
只还原成员逻辑位置并展开，不查原生页表生成物理输出；大Q融合TopK同时省去
物理输出和raw中间索引写入，小Q的Triton映射跳过物理转换，raw仍供TopK使用。
return_logical=True仍可返回逻辑位置，非法项及末尾64项保持-1。
expand_output=False的原生attention输出不变，仍使用当前原生页表生成物理位置。
全量直查路径继续使用原生TopK；本优化针对稀疏输出映射，不改变选页和分数。
展开模式的私有work.output/work.raw不可作为结果读取；使用公开query返回值。
需复现旧物理/raw输出对照时设置query_output_mapping_optimization=False。
测试test_v1_output_mapping.py，基准bench_v1_output_mapping.py，结果V1_OUTPUT_MAPPING_RESULTS_20261016.txt。

2026-10-16 边界过滤判断减少访存
query_bucket_rank_mask=True（默认）：融合TopK内由selected分组页编号计算
bucket=(rank//64)*4096；非边界桶跳过物理页表和首个成员读取。
仅边界桶才读取当前重排页表并逐成员过滤，保留原sink/window及尾部语义。
前提是State.table按4K桶连续登记每桶64页；rank是逻辑分组页编号，不是物理页号。
query_bucket_rank_mask=False保留成员位置推导桶的对照路径。
测试test_v1_rank_mask.py；性能bench_v1_rank_mask.py，结果V1_RANK_MASK_RESULTS_20261016.txt。
仅大Q融合TopK的mask阶段受影响，小Q和query_mask_fusion=False保持原实现。

2026-10-16 精筛页表直接映射 / int32候选编号
- query_int32_selected=True（默认）：CUB选页写int32分组编号；后续Triton与融合
  TopK直接消费，减少selected缓冲区一半空间。torch.topk/signed/超出CUB范围
  自动保留int64，不做额外转换。候选预算和排序规则不变。
- query_direct_fine_map=False（默认）：可选True时，输出成员映射直接查本次
  fine_table中的物理页，省去selected→reordered_table二次寻址；小Q和大Q均支持。
  单独开启收益不稳定，暂不默认开启。该选项与选页/页表生成融合互相独立。
- 原生长度、调度、页表仍在回放时更新；没有固定历史长度或缓存物理页身份。
验证test_v1_fine_map_i32.py、test_v1_deepseek_query_optimized.py、test_v1_direct_map_migration.py。
性能bench_v1_fine_map_i32.py，结果V1_FINE_MAP_I32_RESULTS_20261016.txt。
