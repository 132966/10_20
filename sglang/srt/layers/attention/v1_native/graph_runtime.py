"""Shared graph lifetime: fixed buffers, serialized replay and explicit budget."""
import threading
import torch


class GraphEntry:
    def __init__(self, device, ledger, reservation):
        self.device = torch.device(device)
        self.ledger, self.reservation = ledger, reservation
        self.charged = False
        self.graph = self.stream = self.done = None
        self.buffers, self.uploads = {}, []
        self.lock = threading.RLock()
        self.replays = 0
        self.ledger.acquire(reservation)
        self.charged = True

    def upload(self, name, values, dtype=torch.int64):
        cpu = torch.tensor(values, dtype=dtype, pin_memory=True)
        self.buffers[name].copy_(cpu, non_blocking=True)
        self.uploads.append(cpu)

    def capture(self):
        """Caller has drained index workers. Warmup must precede capture."""
        self.stream = torch.cuda.Stream(device=self.device)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            for _ in range(2):
                self.compute()
        self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.compute()
        self.stream.synchronize()
        self.done = torch.cuda.Event()
        self.done.record(self.stream)

    def begin(self):
        if self.graph is None:
            raise RuntimeError('Index graph has been closed')
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(self.done)
        # Pinned allocations used by nonblocking copies carry allocator events.
        self.uploads.clear()
        return stream

    def finish(self, stream, tensors=()):
        self.done.record(stream)
        self.replays += 1
        for tensor in (*self.buffers.values(), *tensors):
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                tensor.record_stream(stream)

    def close(self):
        with self.lock:
            if self.done is not None:
                self.done.synchronize()
            elif self.stream is not None:
                self.stream.synchronize()
            if self.graph is not None:
                self.graph.reset()
            self.graph = None
            self.buffers.clear()
            self.uploads.clear()
            if self.charged:
                self.ledger.release(self.reservation)
                self.charged = False
