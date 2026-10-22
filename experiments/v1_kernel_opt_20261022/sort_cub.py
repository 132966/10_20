from pathlib import Path
from functools import lru_cache
from torch.utils.cpp_extension import load
@lru_cache(None)
def module():
 p=Path(__file__).resolve().parent
 return load(name='v1_sort_cub_1k',sources=[str(p/'sort_binding.cpp'),str(p/'sort_cub.cu')],extra_cuda_cflags=['-O3'],extra_cflags=['-O2'],verbose=False)
def sort(p,o,s,flag,n):module().sort(p,o,s,flag,n)
