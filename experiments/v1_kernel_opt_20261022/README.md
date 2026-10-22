# V1独立算子优化

完整说明和结果：[RESULTS.md](RESULTS.md)。

- 实现：[optimized.py](optimized.py)、[runtime.py](runtime.py)、[CUB排序](sort_cub.cu)。
- 复现：`bash run.sh`（H20/SM90和本地原CUDA环境）。
- 正确性/完整构建测试：[test_build.py](test_build.py)。
- 原实现副本：[reference/](reference/)，来源校验：[PROVENANCE.json](PROVENANCE.json)。

原SGLang、原共享精筛代码未修改。适配器限定1K桶、1个Index K head、128维、每页64成员；所有缓冲区预分配，支持动态页表/元数据和CUDA Graph。查询DeepGEMM和attention后端保持原实现；没有声称完整推理加速。

优化默认不启用Tensor Core近似点积。被拒绝的试验仍保留用于复现和解释性能取舍。

本次修改逐项说明：[CHANGES.md](CHANGES.md)。
