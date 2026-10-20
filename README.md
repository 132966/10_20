# V1 自定义 GPU 算子与小型测试

这是当前 1K 桶 V1 实现的源代码摘录和配套测试，保留 SGLang 相对目录。包含当前讨论的六项重点功能以及相关旧路径源文件，不是可直接 pip install 的完整 SGLang 服务。

- [全部主要算子功能表](KERNELS.md)
- [六项重点实现、测试和例子](sglang/srt/layers/attention/v1_deepseek/README_FIVE_KERNEL_EXAMPLES.md)
- [建索引完整调用链](sglang/srt/layers/attention/v1_deepseek/README_INDEX_BUILD_1K.md)
- [单头融合查询例子](sglang/srt/layers/attention/v1_deepseek/README_QUERY_SINGLE_HEAD_EXAMPLE.md)
- [分组实现详解](docs/V1_GROUPING_KERNEL_DETAILED_20261019.md)
- [上游来源及复用说明](UPSTREAM.md)

## 六项重点功能

|实现|语言|测试|
|---|---|---|
|fused.route|CUDA C++|example_v1_query_single_head.py：1024 Q、单头、40 页|
|fused.topk|CUDA C++|同上：2688 候选、Top2048、过滤与映射|
|_sort|Triton|test_fused_grouping_small.py：16 成员稳定排序|
|match_pages → allocate → publish → release|Triton|test_page_management_and_tail_small.py：16 页共享及回收|
|stage_tail_shared|Triton|同上：4 Q、两对共享副本|
|_gather_slots|Triton|test_gather_slots_small.py：小型 K/V gather 与 Top2048 工作区复用|

## 运行与测试范围

在有 PyTorch、Triton 和 CUDA GPU 的环境，可直接运行三个独立 Triton 测试：

```bash
python sglang/srt/layers/attention/v1_native/test_fused_grouping_small.py -v
python sglang/srt/layers/attention/v1_native/test_gather_slots_small.py -v
python sglang/srt/layers/attention/v1_deepseek/test_page_management_and_tail_small.py -v
```

融合 CUDA 示例还需要兼容的 SGLang JIT/TVM-FFI/CUDA 编译环境。请在独立的兼容 SGLang checkout 中，按相对路径合入这些文件后运行 example_v1_query_single_head.py；仅将本摘录目录加入 PYTHONPATH 并不能替代完整 SGLang 安装。建议使用 UPSTREAM.md 标明的历史版本及原有运行环境；不保证与最新版接口兼容。

测试用人工数据直接调用真实 GPU kernel，并与参考结果核对。原环境已通过：分组6项、页管理/尾部2项、gather2项及1024 Q融合查询。不是端到端模型评测或性能基准；尚未在干净安装环境重测。

源代码中的少量原环境路径属于旧实验配置，移植服务时需要自行配置；不影响三个独立 Triton 测试。原始源文件未改动，文档本地链接已转换。SOURCE_MANIFEST.json 记录导出文件哈希。
