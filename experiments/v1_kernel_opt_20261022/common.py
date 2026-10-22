import importlib.util,hashlib,json,random,statistics
from pathlib import Path
import torch,triton
from triton.testing import do_bench_cudagraph
ROOT=Path(__file__).resolve().parent
BASE=Path('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/IA/xiaobin14/zhanghao/helmet_qwen3_4b_20261010/qwen3_gqa_v1_dsa_bucket1k_overlap_20261017/sglang')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
fg=load('group_ref',ROOT/'reference/fused_grouping.py')
v=load('page_ref',ROOT/'reference/kernels.py')
def bench(fns,repeats=3):
 for f in fns.values():
  for _ in range(5):f()
 torch.cuda.synchronize()
 samples={n:{'eager_us':[],'graph_us':[]} for n in fns}
 for r in range(repeats):
  names=list(fns);random.Random(r).shuffle(names)
  for n in names:
   f=fns[n];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
   a.record()
   for _ in range(100):f()
   b.record();b.synchronize();samples[n]['eager_us'].append(a.elapsed_time(b)*10)
   samples[n]['graph_us'].append(do_bench_cudagraph(f,rep=30)*1000)
 return {n:{k:statistics.median(x) for k,x in d.items()} for n,d in samples.items()}
def tensor(x,dtype=torch.int32):return torch.tensor(x,device='cuda',dtype=dtype)
def save(name,data):(ROOT/name).write_text(json.dumps(data,indent=2)+'\n')
