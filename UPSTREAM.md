# 上游来源和复用边界

`sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh` 与 SGLang 官方提交 `0ba46c88e5b16c739be0b698ec7405a147a05cb4` 对应文件逐字节一致，SHA256 为 `7706f74e012e91948a786a1a0ba62deaab6d860e1fe4e5470ad5dfdf0031a951`。

[官方历史文件](https://github.com/sgl-project/sglang/blob/0ba46c88e5b16c739be0b698ec7405a147a05cb4/python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh)

`topk_impl.cuh` 及 `v1_native/deps/*vendor*` 是随运行快照保留的 TopK 依赖；不要将其视作全部原创。vendor 中的映射适配可能不同于官方原始文件，未逐文件验证来源 commit。SGLang 的许可证见 third_party/SGLANG_LICENSE，保留各源文件现有来源说明。

V1 的定制融合位于 v1_query.cuh；其中 route 使用 NVIDIA CUB，topk 复用已有核心算法。Triton 分组排序使用 tl.sort。DeepGEMM 与 FA3 需在运行环境另外安装。

导出来源：20261017 的 bucket1k_overlap 实验快照，加当前 HELMET 后端文件。不是完整官方 SGLang 仓库，不包含模型权重、数据集、密钥或运行日志。
