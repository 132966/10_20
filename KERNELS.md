# 当前主要自定义算子功能表

独立、融合与备用路径并列；不是一次查询全部执行。旧版扩展源文件另附在目录中。

|实现|阶段|功能|语言|文件|
|---|---|---|---|---|
|`pack_new`|输入准备|新增 GQA K 打包成 FP8 K＋scale|Triton|[qwen_gqa_probe_kernels.py](qwen_gqa_probe_kernels.py)|
|`pack_q`|输入准备|打包 Q、权重并补齐查询头|Triton|[qwen_gqa_probe_kernels.py](qwen_gqa_probe_kernels.py)|
|`gather_index`|DSA 对照|收集连续 Index K＋scale|Triton|[qwen_gqa_probe_kernels.py](qwen_gqa_probe_kernels.py)|
|`initialize`|建索引|按页表读取、反量化、归一化 K 并初始化排列|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`assemble`|建索引|生成临时重排 K、中心及成员映射|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`fingerprint`|页管理|计算数据及成员指纹|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`match_pages`|页管理|指纹及完整内容匹配|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`allocate`|页管理|分配/复用页、引用计数和桶页表|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`publish`|页管理|发布重排页、中心、映射和指纹|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`release`|页管理|引用计数递减、零引用页回收|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`query_lengths`|查询准备|计算有效长度|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`query_metadata`|查询准备|准备长度、预算相关及尾部共享元数据|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`coarse`|粗筛备用|中心打分和页面归约|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`reduce_centers`|粗筛|8 中心分数取 max|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`fine_table`|精筛准备|生成物理页表|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`fine_table_shared`|精筛准备|生成共享尾部页表|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`stage_tail`|精筛准备|复制固定候选及尾部 K＋scale|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`stage_tail_shared`|精筛准备|按 owner 复用尾部副本|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`fine`|精筛备用|候选 K 打分及过滤|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`mask_fine`|过滤|屏蔽无效、不可见和边界重复候选|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`output_ids`|输出|逻辑/物理位置映射|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`expand_logical`|输出|按配置展开逻辑位置|Triton|[kernels.py](sglang/srt/layers/attention/v1_deepseek/kernels.py)|
|`_init`|分组|旧4K入口归一化和排列初始化|Triton|[fused_grouping.py](sglang/srt/layers/attention/v1_native/fused_grouping.py)|
|`_dots`|分组|锚点点积与差分打分|Triton|[fused_grouping.py](sglang/srt/layers/attention/v1_native/fused_grouping.py)|
|`_anchor`|分组|选择最小点积锚点|Triton|[fused_grouping.py](sglang/srt/layers/attention/v1_native/fused_grouping.py)|
|`_sort`|分组|稳定降序排列、均衡拆分|Triton|[fused_grouping.py](sglang/srt/layers/attention/v1_native/fused_grouping.py)|
|`_publish_exact`|分组|复制参考分组结果|Triton|[fused_grouping.py](sglang/srt/layers/attention/v1_native/fused_grouping.py)|
|`_gather_slots`|收集/旧路径辅助|逻辑映射＋K/V gather＋输出布局融合|Triton|[kernels.py](sglang/srt/layers/attention/v1_native/kernels.py)|
|`_pack_slots`|收集/旧路径辅助|量化打包并记录成员映射|Triton|[kernels.py](sglang/srt/layers/attention/v1_native/kernels.py)|
|`_mask_scores`|收集/旧路径辅助|旧路径分数过滤|Triton|[kernels.py](sglang/srt/layers/attention/v1_native/kernels.py)|
|`_append_flat`|收集/旧路径辅助|新增 K 量化追加到连续缓存|Triton|[kernels.py](sglang/srt/layers/attention/v1_native/kernels.py)|
|`v1_route_main / fused.route`|查询融合|页面 max＋TopK＋可选页表写入融合|CUDA C++|[v1_query.cuh](sglang/kernels/jit/csrc/deepseek_v4/v1_query.cuh)|
|`v1_topk_main / fused.topk`|查询融合|TopK＋过滤＋成员映射/物理位置转换/展开融合|CUDA C++|[v1_query.cuh](sglang/kernels/jit/csrc/deepseek_v4/v1_query.cuh)|

核心 TopK、CUB、tl.sort、DeepGEMM 和 FA3 为复用组件，不作为本项目原创算法。
