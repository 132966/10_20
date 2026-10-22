#include <torch/extension.h>
void sort_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("sort",&sort_cuda);}
