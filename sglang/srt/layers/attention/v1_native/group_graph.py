"""Optional exact PyTorch grouping replay; no changed arithmetic or grouping."""
import threading
import torch
from .grouping import build_bucket

RESERVATION = 512 * (1 << 20)

class GroupGraph:
    def __init__(self, heads, device):
        self.device=device
        self.lock=threading.Lock()
        self.stream=torch.cuda.Stream(device=device)
        self.input=torch.zeros((4096,heads,128),device=device,dtype=torch.bfloat16)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.inference_mode(),torch.cuda.stream(self.stream):
            for _ in range(2):
                build_bucket(self.input,0,ids_only=True)
        self.stream.synchronize()
        before=torch.cuda.memory_reserved(device)
        self.graph=torch.cuda.CUDAGraph()
        with torch.inference_mode(),torch.cuda.graph(self.graph,stream=self.stream):
            self.output=build_bucket(self.input,0,ids_only=True)
        self.stream.synchronize()
        self.reserved_delta=max(0,torch.cuda.memory_reserved(device)-before)
        if self.reserved_delta+self.input.numel()*self.input.element_size()>RESERVATION:
            self.graph.reset()
            raise RuntimeError('Grouping graph exceeds its reserved workspace')
        self.done=torch.cuda.Event()
        self.done.record(self.stream)

    def run(self,keys,start):
        with self.lock:
            stream=torch.cuda.current_stream(keys.device)
            stream.wait_event(self.done)
            self.input.copy_(keys)
            self.graph.replay()
            result=self.output+start
            self.done.record(stream)
            return result

    def close(self):
        self.done.synchronize()
        self.graph.reset()
        self.input = self.output = None
