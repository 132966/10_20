#include <torch/extension.h>
void route_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("route", &route_cuda); }
