"""Independent physical-page route adapter. Compile before Graph capture.
Inputs: scores[R,P*8], published[R], budgets[R], rt[R,P].
Outputs: selected[R,M] physical pages, takes[R]. No tensor allocation in route.
Published counts and rt must describe the same event-visible snapshot as END.
Finite center scores (or -inf masks) expected; NaN scores are unsupported.
"""
from pathlib import Path
from functools import lru_cache
from torch.utils.cpp_extension import load

@lru_cache(None)
def module():
    here=Path(__file__).resolve().parent
    return load(name='v1_shared_route_physical_v1',
                sources=[str(here/'route_binding.cpp'),str(here/'route_shared.cu')],
                extra_cuda_cflags=['-O3','-std=c++17'],extra_cflags=['-O2'],verbose=False)

def route(scores,published,budgets,rt,selected,takes):
    module().route(scores,published,budgets,rt,selected,takes)
