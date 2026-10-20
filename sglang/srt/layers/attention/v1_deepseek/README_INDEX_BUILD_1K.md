# V1 1K 桶：建索引与插索引完整代码导读

更新时间：2026-10-19。

本文对应当前 Qwen3 GQA / HELMET 评测使用的 1K 桶实现。它按一个 layer × KV head 建立独立索引，page_size=64；一个完整桶包含 1024 个 K。本文摘录真实函数并逐步解释，不修改实现。当前评测使用 eager + 独立 CUDA stream，未开启建索引 CUDA Graph。

## 1. 调用链

```text
后端 submit_build()                          切换到 build_stream
  → Indexer.prepare()                       锁、等待前一次操作、完成事件
    → Indexer._prepare()                    校验请求，找新增完整桶
      → Build.run()                         上传本桶 16 项页表和元数据
        → Build.compute()                   发出完整 GPU kernel 序列
          → initialize                      读 K、反量化、归一化
          → _dots / _anchor / _sort          逐级分组，生成成员排列
          → assemble                        重排 K，生成中心、逻辑成员映射
          → fingerprint                     生成页指纹
          → match_pages                     找可复用的已存活重排页
          → allocate                        分配/复用页，更新引用与桶页表
          → publish                         写入重排数据、成员、指纹、中心
      → 更新 state.table / center_table     发布请求视角的页表
  → 记录 pending[gid] 完成事件
下一 chunk 查询同层同 head 前等待该事件
```

“建索引”主要是分组、重排和中心生成；“插索引”主要是查重、页分配、发布和请求页表更新。当前二者属于同一次 Build 调用，不需要 CPU 读回聚类结果后再插入。

## 2. 文件与入口总览

|步骤|代码入口|
|---|---|
|提交异步构建|[QwenGQAProbeBackend.submit_build](../../../../../qwen_gqa_probe_backend.py)|
|准备入口|[Indexer.prepare](runtime.py)|
|增量桶调度|[Indexer._prepare](runtime.py)|
|单桶输入更新|[Build.run](runtime.py)|
|单桶 GPU 流程|[Build.compute](runtime.py)|
|读取和归一化|[initialize](kernels.py)|
|分组打分|[_dots](../v1_native/fused_grouping.py)|
|锚点选择|[_anchor](../v1_native/fused_grouping.py)|
|稳定排序|[_sort](../v1_native/fused_grouping.py)|
|重排和中心|[assemble](kernels.py)|
|页指纹|[fingerprint](kernels.py)|
|查重|[match_pages](kernels.py)|
|分配页|[allocate](kernels.py)|
|发布内容|[publish](kernels.py)|

## 3. 贯穿全文的数据结构

以下形状以一个 Indexer 为单位；当前后端每个 layer/head 对应一个 Indexer。C 为该 Indexer 重排物理页容量，L=C/16 为桶记录容量。

|对象|shape / 类型|含义|
|---|---|---|
|`native`|`[P,64,1,132]`, uint8 打包视图|原生 Index K＋scale 源缓存|
|调用参数 `native_table`|`[ceil(length/64)]` 或更长，int32|当前请求逻辑页到源物理页的映射；后端还编码 layer/head 偏移|
|`Build.source`|`[16]`, int32|只包含当前 1K 桶的 16 个源页编号|
|`Build.meta`|`[3]`, int32|`[start, lease, active]`|
|`Build.unit`|`[1024,128]`, float32|用于分组的单位向量|
|`Build.perm[0/1]`|各 `[1024]`, int32|交替写入的桶内成员排列|
|`Build.tmp`|`[16,8448]`, uint8|当前桶重排 K＋scale 临时页|
|`Build.centers`|`[2,8448]`, uint8|128 个中心，打包成 2 个中心页|
|`Build.mapping`|`[16,64]`, int32|每个重排位置对应的序列逻辑位置|
|`Build.hash`|`[16,2]`, int64|每个临时页两个指纹值|
|`Build.matches`|`[16]`, int32|可复用页编号，-1 表示未命中|
|`Build.new`|`[16]`, bool|是否新分配该页|
|`Indexer.fine`|有效持久区域 C 个 64-token 页|重排 K＋scale；独立分配模式还可能带查询工作区|
|`Indexer.centers`|`[C/8,64,1,132]`|按桶记录编号组织的中心池|
|`Indexer.members`|`[C,64]`, int32|持久成员逻辑位置映射|
|`Indexer.refs/free/free_count`|`[C] / [C] / 标量`|引用计数、空闲页栈、栈中可用页数|
|`Indexer.tables`|`[L,16]`, int32|桶记录到 16 个重排物理页的映射|
|`Indexer.center_pages`|`[L,2]`, int32|桶记录到 2 个中心物理页的映射|
|`state.leases`|CPU list|当前请求已建立桶的记录编号，按桶顺序排列|
|`state.table`|GPU int32 固定容量数组|请求按桶展开的重排物理页表|
|`state.center_table`|GPU int32 固定容量数组|请求按桶展开的中心物理页表|

一页实际为 8448 字节：前 8192 字节存 64×128 个 FP8 K，后 256 字节存 64 个 FP32 scale。`[64,1,132]` 是打包视图，不能理解为每个 token 的 scale 紧跟它的 K。

`slot` 是请求槽位，不是某个 token 的物理 slot。`lease` 是 V1 桶记录编号，不是原生 KV pool 的物理页编号。

## 4. submit_build：异步提交入口

代码：[QwenGQAProbeBackend.submit_build](../../../../../qwen_gqa_probe_backend.py)
```python
 def submit_build(self,layerid,slot,slots,total,ready):
  assert total%1024==0
  for h in range(8):
   gid=layerid*8+h;o=self.get_owner(gid)
   old=len(o.states[slot].leases) if slot in o.states else 0
   assert old+1==total//1024,('incremental bucket',layerid,h,old,total)
   # Record after owner initialization as well as the original K write.
   initialized=torch.cuda.Event();initialized.record()
   with torch.cuda.stream(self.build_stream):
    self.build_stream.wait_event(ready);self.build_stream.wait_event(initialized)
    table=(slots[:total:64]//64+gid*self.native_pages).int().contiguous()
    a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True);a.record()
    o.prepare(slot,table,total);z.record()
   self.pending[gid]=z
   self.build_records.append((dict(layer=layerid,head=h,prefix=total-1024,total=total,new_buckets=1,bucket_tokens=1024),a,z))
```

输入是层号、请求槽位、当前请求的 token slot 映射、当前总长度以及 K 已写入的事件。

具体行为：

1. 要求 total 是 1024 的整数倍；当前后端一次新增一个完整桶。
2. 对 8 个 KV heads 分别取得 Indexer，gid=`layerid*8+h`。
3. `initialized` 在主计算流记录，覆盖 owner 初始化以及此前主流操作。
4. 切换到共享的 `build_stream`，等待源 K ready 和 initialized。
5. 从每个逻辑页的第一个 token slot 推导物理页号：`slots[::64]//64`；再加该 layer/head 的源池页偏移。
6. 调用 prepare，最后记录 z 并保存在 `pending[gid]`。

这依赖当前标准分页映射：page_size=64，页内 token slot 按页对齐且连续；不是支持任意非连续的 64 个 slot 被强行当成一页。

构建任务在当前层 attention 之后提交，主要与后续模型计算重叠。8 个 head 在同一个构建 stream 排队，不是 8 个独立 stream。CPU 返回并不意味着 GPU 已完成，下一次消费由事件保证。

## 5. Indexer.prepare：锁和事件包裹

代码：[Indexer.prepare](runtime.py)
```python
    @torch.inference_mode()
    def prepare(self, owner, native_table, length):
        with self.lock:
            self._begin()
            try:
                return self._prepare(owner, native_table, length)
            finally:
                self._finish()
```

`owner` 在当前后端传入请求 slot；运行时还支持其他可哈希请求标识，但这里没有自动附加 generation。

- `self.lock` 防止同一 Indexer 的 Python 状态被并发修改。
- `_begin()` 检查是否已关闭、是否处于不支持的外层 Graph 捕获中，并让当前 stream 等待 `Indexer.done`。
- `_finish()` 在当前 stream 记录新的 done，覆盖本次 prepare 中提交的 GPU 工作。
- 此处没有正常路径上的 GPU→CPU 聚类结果回传。

锁和 CUDA event 解决不同问题：锁管理 CPU 调度状态，event 管理 GPU 执行依赖。

## 6. Indexer._prepare：找新增的完整桶

代码：[Indexer._prepare](runtime.py)
```python
    def _prepare(self, owner, native_table, length):
        if not 0 <= length <= self.config.max_context:
            raise ValueError("Index context capacity exceeded")
        if (
            native_table.dtype != torch.int32
            or native_table.device != self.device
            or native_table.ndim != 1
            or native_table.numel() < tr.cdiv(length, 64)
            or not native_table.is_contiguous()
        ):
            raise ValueError("Invalid compact native page table")
        state = self.states.get(owner)
        if state is not None and length < state.length:
            self._release(owner)
            state = None
        if state is None:
            if len(self.states) >= self.config.max_requests:
                raise ValueError("V1 request capacity exceeded")
            state = self.states[owner] = State(self)
        complete = length // 1024
        if complete > len(state.leases) and self.build is None:
            self.build = Build(self)
            self.build_capture_count += int(self.config.graphs)
        for bucket in range(len(state.leases), complete):
            if not self.free_leases:
                raise RuntimeError("V1 bucket records exhausted")
            lease = self.free_leases.pop()
            self.build.run(native_table, bucket * 1024, lease)
            state.leases.append(lease)
            state.table[bucket * 16 : (bucket + 1) * 16].copy_(self.tables[lease])
            state.center_table[bucket * 2 : (bucket + 1) * 2].copy_(
                self.center_pages[lease]
            )
        state.length = length
        return state
```

重点逻辑：

- 校验长度上限、页表 dtype/device/维度/容量/连续性。
- 请求长度回退时释放原状态后重建，避免保留超出有效长度的成员。
- `complete=length//1024`：未满 1K 的尾部不在这里建桶。
- 首次需要构建时才创建 Build 和工作区。
- 从 `len(state.leases)` 遍历到 complete，只处理未构建的桶。
- `free_leases.pop()` 分配的是 CPU 管理的桶记录编号；真正重排物理页在 GPU allocate 内分配。
- `build.run()` 后将桶页表和中心页表拷入请求状态，最后更新 CPU 上的 state.length。

这里 state.leases、state.length 描述已提交的请求状态；CPU 元数据更新不等价于 GPU 内容已经可读取，查询仍需等待完成事件。

## 7. Build.run：为一个桶设置输入

代码：[Build.run](runtime.py)
```python
    def run(self, table, start, lease):
        self.begin()
        self.source.copy_(table[start // 64 : start // 64 + 16])
        self.upload(self.meta, [start, lease, 1])
        self.execute()
        self.finish()
        table.record_stream(torch.cuda.current_stream(self.device))
```

例如 start=65536：从当前请求源页表中取 `[1024:1040]` 的 16 项，复制到固定地址的 source 工作区。

`meta=[65536,lease,1]` 表示源桶起点、目标桶记录和启用标志。upload 使用 pinned CPU 小张量并异步复制；这是 CPU→GPU 的控制数据上传，不是把分组结果读回 CPU。

`begin()` 等待前一次使用同一 Build 工作区的完成事件；execute 在本次评测直接调用 compute，若启用对应 Graph 才回放图；finish 记录 Build 自己的完成事件。`record_stream` 保护输入页表张量的跨流生命周期。

## 8. Build.compute：完整单桶 kernel 调度

代码：[Build.compute](runtime.py)
```python
    def compute(self):
        o = self.owner
        k.initialize[(64,)](
            o.native.view(torch.float8_e4m3fn),
            o.native.view(torch.float32),
            self.source,
            self.meta,
            self.unit,
            self.perm[0],
            self.flag,
            num_warps=4,
        )
        current = 0
        for n in (1024, 512, 256, 128, 64, 32, 16):
            for mode in (0, 1, 2):
                fg._dots[(64, 1)](
                    self.unit,
                    self.perm[current],
                    self.anchor,
                    self.scores,
                    self.flag,
                    n,
                    0,
                    mode,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                if mode < 2:
                    fg._anchor[(1024 // n, 1)](
                        self.perm[current],
                        self.anchor,
                        self.scores,
                        self.flag,
                        n,
                        mode,
                        num_warps=4,
                    )
            fg._sort[(1024 // n, 1)](
                self.perm[current],
                self.perm[1 - current],
                self.scores,
                self.flag,
                n,
                num_warps=4 if n <= 1024 else 8,
            )
            current = 1 - current
        k.assemble[(16, 8)](
            o.native.view(torch.float8_e4m3fn),
            o.native.view(torch.float32),
            self.source,
            self.meta,
            self.perm[current],
            self.tmp.view(torch.float8_e4m3fn),
            self.tmp.view(torch.float32),
            self.centers.view(torch.float8_e4m3fn),
            self.centers.view(torch.float32),
            self.mapping,
            num_warps=4,
        )
        k.fingerprint[(16,)](self.tmp, self.mapping, self.meta, self.hash, num_warps=8)
        k.match_pages[(16,)](
            self.tmp,
            self.mapping,
            self.hash,
            self.meta,
            o.fine,
            o.members,
            o.hashes,
            o.refs,
            self.matches,
            o.capacity,
            tr.next_power_of_2(o.capacity),
            num_warps=8,
        )
        k.allocate[(1,)](
            self.meta,
            self.matches,
            o.free,
            o.free_count,
            o.refs,
            o.tables,
            self.new,
            debug=True,
        )
        k.publish[(16,)](
            self.tmp,
            self.centers,
            self.centers.view(torch.float32),
            self.mapping,
            self.hash,
            self.meta,
            o.tables,
            self.new,
            o.fine,
            o.centers,
            o.centers.view(torch.float32),
            o.members,
            o.hashes,
            num_warps=8,
        )
```

该函数按同一 stream 的提交顺序串起全部 GPU 操作。分组子阶段需要前一阶段结果，因此它们不是可以随意并行的独立任务。下面逐步解释。

## 9. initialize：读取、反量化、归一化

代码：[initialize](kernels.py)
```python
@tr.jit
def initialize(K, KS, TABLE, META, UNIT, PERM, FLAG):
    active = tl.load(META + 2)
    if tl.program_id(0) == 0:
        tl.store(FLAG, tl.where(active != 0, -1, 0))
    if active:
        i = tl.program_id(0) * 16 + tl.arange(0, 16)
        d = tl.arange(0, 128)
        page = tl.load(TABLE + i // 64).to(tl.int64)
        scale = tl.load(KS + page * 2112 + 2048 + i % 64)
        x = (
            tl.load(K + page[:, None] * 8448 + (i % 64)[:, None] * 128 + d[None, :]).to(
                tl.float32
            )
            * scale[:, None]
        )
        norm = tl.sqrt(tl.sum(x * x, 1))
        tl.store(
            UNIT + i[:, None] * 128 + d[None, :], x / tl.maximum(norm[:, None], 1.0e-12)
        )
        tl.store(PERM + i, i)
```

每个 program 处理 16 个 K，64 个 program 覆盖 1024 个成员。

按 source 页表读取 FP8 K 与 scale，反量化得到 x，再计算 `unit=x/max(norm(x),1e-12)`。初始化排列为 0..1023；active 时 flag=-1 允许分组。

输出是临时 unit 和 perm，并不修改原生 K。后面重排保存的仍是原有 FP8 K＋scale，不是归一化后的向量。

## 10. 分组 kernel：逐级平衡二分

调用位置见 Build.compute。底层函数：

- [_dots](../v1_native/fused_grouping.py)：MODE=0 对当前首成员打分；MODE=1 对 b 打分；MODE=2 计算到 a 与到 b 的点积差。
- [_anchor](../v1_native/fused_grouping.py)：选最小点积对应成员，依次得到 b、a；同分取当前排列靠左的位置。
- [_sort](../v1_native/fused_grouping.py)：按 margin 稳定降序排列成员 ID，双缓冲交替写入。

当前 n 依次为 `1024,512,256,128,64,32,16`，每级排序后由下一层按半长划分，最终得到 128 个 8-member 小组。后续每 8 个小组合成一个 64-member 重排页。

当前直接调用这些 kernel，而不是调用旧 4K 的 `FusedGroups.run()`。grid 的 head 维为 1，h=0，因此旧 kernel 的 head stride 常量不会在本路径中产生非零偏移。不能直接扩成多 head 的 1K 紧凑工作区而不改 stride。

更细的算法、shape、同分规则和数值例子见：

[V1_GROUPING_KERNEL_DETAILED_20261019.md](../../../../../docs/V1_GROUPING_KERNEL_DETAILED_20261019.md)

## 11. assemble：产生临时重排页、中心、成员映射

代码：[assemble](kernels.py)
```python
@tr.jit
def assemble(K, KS, TABLE, META, IDS, TMP, TMPS, CENTERS, CS, MAP):
    if tl.load(META + 2):
        page, g = tl.program_id(0), tl.program_id(1)
        i = g * 8 + tl.arange(0, 8)
        d = tl.arange(0, 128)
        src = tl.load(IDS + page * 64 + i)
        native = tl.load(TABLE + src // 64).to(tl.int64)
        k = tl.load(K + native[:, None] * 8448 + (src % 64)[:, None] * 128 + d[None, :])
        scale = tl.load(KS + native * 2112 + 2048 + src % 64)
        tl.store(TMP + page * 8448 + i[:, None] * 128 + d[None, :], k)
        tl.store(TMPS + page * 2112 + 2048 + i, scale)
        tl.store(MAP + page * 64 + i, tl.load(META) + src)
        mean = tl.sum(k.to(tl.float32) * scale[:, None], 0) / 8.0
        cs = tl.exp2(
            tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(mean), 0), 1.0e-4) / 448.0))
        )
        c = page * 8 + g
        tl.store(
            CENTERS + (c // 64) * 8448 + (c % 64) * 128 + d,
            (mean / cs).to(tl.float8e4nv),
        )
        tl.store(CS + (c // 64) * 2112 + 2048 + c % 64, cs)
```

grid=(16,8)：16 个重排页，每页 8 个小组，每组 8 个成员。

- 根据最终排列读取原生 K 与 scale，写入 tmp。这里复制量化数据，不用单位向量替代 K。
- mapping 写入 `start+src`，保存序列逻辑位置，而非原生物理 slot。
- 每组 8 个反量化 K 求均值，重新量化为 FP8 中心和独立 scale。
- 最终有 16 个重排临时页和 2 个中心临时页。

注意此时数据仍是临时结果，尚未分配或发布持久重排页。

## 12. fingerprint：计算临时页指纹

代码：[fingerprint](kernels.py)
```python
@tr.jit
def fingerprint(TMP, MAP, META, HASH):
    if tl.load(META + 2):
        p = tl.program_id(0)
        i = tl.arange(0, 16384)
        x = tl.load(TMP + p * 8448 + i, i < 8448, other=0).to(tl.int64)
        m = tl.load(MAP + p * 64 + i, i < 64, other=0).to(tl.int64)
        a = tl.sum(x * (i.to(tl.int64) + 1), 0) + tl.sum(m * 104729, 0)
        b = tl.sum(x * ((i.to(tl.int64) + 7) * 1000003), 0) + tl.sum(
            m * (i.to(tl.int64) + 1) * 65537, 0
        )
        tl.store(HASH + p * 2, a)
        tl.store(HASH + p * 2 + 1, b)
```

输入是每页完整 8448 字节（含 scale）以及 64 个成员逻辑位置；输出每页两个 int64 指纹。

这是用来快速筛选可能相同的页，不是密码学哈希，也不独自证明两个页相同。包含逻辑成员位置意味着仅 K 内容相同、逻辑映射不同的页不能直接共用同一 members 映射。

## 13. match_pages：在已存活页中查找可复用页

代码：[match_pages](kernels.py)
```python
@tr.jit
def match_pages(
    TMP,
    MAP,
    HASH,
    META,
    FINE,
    MEMBERS,
    HASHES,
    REFS,
    MATCH,
    CAP: tl.constexpr,
    SCAN: tl.constexpr,
):
    if tl.load(META + 2):
        p = tl.program_id(0)
        j = tl.arange(0, SCAN)
        refs = tl.load(REFS + j, j < CAP, other=0)
        h0 = tl.load(HASHES + j * 2, (j < CAP) & (refs > 0), other=0)
        h1 = tl.load(HASHES + j * 2 + 1, (j < CAP) & (refs > 0), other=0)
        good = (
            (j < CAP)
            & (refs > 0)
            & (h0 == tl.load(HASH + p * 2))
            & (h1 == tl.load(HASH + p * 2 + 1))
        )
        dst = tl.min(tl.where(good, j, CAP), 0)
        found = -1
        byte = tl.arange(0, 16384)
        a = tl.load(TMP + p * 8448 + byte, byte < 8448, other=0)
        ma = tl.load(MAP + p * 64 + byte, byte < 64, other=0)
        while (dst < CAP) & (found < 0):
            b = tl.load(FINE + dst.to(tl.int64) * 8448 + byte, byte < 8448, other=0)
            mb = tl.load(MEMBERS + dst * 64 + byte, byte < 64, other=0)
            same = (
                tl.sum((a != b).to(tl.int32), 0) + tl.sum((ma != mb).to(tl.int32), 0)
                == 0
            )
            found = tl.where(same, dst, -1)
            good = good & (j != dst)
            dst = tl.min(tl.where(good, j, CAP), 0)
        tl.store(MATCH + p, found)
```

每个 program 处理一个临时页：

1. 扫描该 Indexer 的 C 个候选重排页。
2. 仅 refs>0 的存活页参与，先比较两个指纹。
3. 对指纹命中的页再比较全部 8448 字节和 64 个成员逻辑位置。
4. 找到完全一致页返回物理页编号，否则返回 -1。

所以指纹碰撞不会直接造成错误复用。当前是容量范围扫描，不是哈希表 O(1) 查找。

匹配范围受当前 Indexer 限制，不是跨所有层、所有 heads 的全局去重池。当前单请求、Radix 关闭测试不能直接证明跨请求共享收益。

同一桶的 16 个新临时页先一起做 match，之后才 allocate/publish；不能把这一实现描述成“对本批所有新页也保证互相去重”。

## 14. allocate：选择物理页并登记引用

代码：[allocate](kernels.py)
```python
@tr.jit
def allocate(META, MATCH, FREE, COUNT, REFS, TABLES, NEW):
    if tl.load(META + 2):
        i = tl.arange(0, 16)
        dst = tl.load(MATCH + i)
        fresh = dst < 0
        rank = tl.cumsum(fresh.to(tl.int32), 0) - 1
        needed = tl.sum(fresh.to(tl.int32), 0)
        count = tl.load(COUNT)
        tl.device_assert(count >= needed, "V1 DeepSeek page arena exhausted")
        allocated = tl.load(FREE + count - 1 - rank, fresh, other=0)
        dst = tl.where(fresh, allocated, dst)
        tl.atomic_add(REFS + dst, 1, sem="relaxed")
        tl.store(COUNT, count - needed)
        tl.store(TABLES + tl.load(META + 1) * 16 + i, dst)
        tl.store(NEW + i, fresh)
```

输入 matches 中已有页编号直接复用；-1 的位置从 free 栈弹出新页。fresh 的前缀和给各新页分配不同的栈位置。

- 空闲页不足时触发设备断言；调用处传入 debug=True。
- 对所有使用的目标页，refs 加 1，包括复用页。
- free_count 减去真正新增页数。
- `tables[lease,:]` 写入本桶 16 个重排物理页编号。
- new 标记哪些页需要写入新内容。

这是在预分配重排池内部选页，不是在每页构建时执行 cudaMalloc。

此时页表和引用计数可能已经更新，但持久数据尚未 publish；外部消费者不能只看 refs 或 CPU lease 列表就读取，必须遵守 stream/event 依赖。

## 15. publish：写入持久重排池和中心池

代码：[publish](kernels.py)
```python
@tr.jit
def publish(
    TMP, TC, TCS, MAP, HASH, META, TABLES, NEW, FINE, CENTERS, CS, MEMBERS, HASHES
):
    if tl.load(META + 2):
        p = tl.program_id(0)
        if tl.load(NEW + p):
            dst = tl.load(TABLES + tl.load(META + 1) * 16 + p)
            i = tl.arange(0, 16384)
            tl.store(
                FINE + dst.to(tl.int64) * 8448 + i,
                tl.load(TMP + p * 8448 + i, i < 8448, other=0),
                i < 8448,
            )
            tl.store(
                MEMBERS + dst * 64 + i,
                tl.load(MAP + p * 64 + i, i < 64, other=-1),
                i < 64,
            )
            tl.store(
                HASHES + dst * 2 + i, tl.load(HASH + p * 2 + i, i < 2, other=0), i < 2
            )
        # Centers belong to the bucket record, independently of shared fine pages.
        # Eight consecutive center pages per record enable native paged GEMM.
        x = tl.arange(0, 1024)
        src_c = p * 8 + x // 128
        dst_c = tl.load(META + 1).to(tl.int64) * 128 + src_c
        val = tl.load(TC + (src_c // 64) * 8448 + (src_c % 64) * 128 + x % 128)
        tl.store(CENTERS + (dst_c // 64) * 8448 + (dst_c % 64) * 128 + x % 128, val)
        g = tl.arange(0, 8)
        src_c = p * 8 + g
        dst_c = tl.load(META + 1).to(tl.int64) * 128 + src_c
        val = tl.load(TCS + (src_c // 64) * 2112 + 2048 + src_c % 64)
        tl.store(CS + (dst_c // 64) * 2112 + 2048 + dst_c % 64, val)
```

对于 new=True 的重排页：写入 fine、members、hashes。对于复用页：跳过这些已有内容的重复写入。

中心按桶记录单独写入：一个 lease 固定占 128 个中心，即 2 个中心物理页。即使某个 fine 页复用，当前桶记录的中心仍按自己的位置写入。

因此：

```text
重排页编号由 GPU allocator 决定，可离散分布。
中心页编号由桶记录 lease 决定，本实现是 [lease*2, lease*2+1]。
```

这两套编号没有“一页对一页使用相同编号”的关系。

源码内保留了一条旧的 “Eight consecutive center pages per record” 注释；当前 1K 代码实际为每记录 2 个中心页，应以地址公式和当前容量为准。

## 16. 更新请求重排页表与中心页表

位置在 Indexer._prepare 的循环末尾，关键代码为：

```python
state.leases.append(lease)
state.table[bucket*16:(bucket+1)*16].copy_(self.tables[lease])
state.center_table[bucket*2:(bucket+1)*2].copy_(self.center_pages[lease])
state.length = length
```

`tables[lease]` 是桶记录视角；state.table 是请求按逻辑桶顺序展开后的视角。查询使用请求页表，不必按所有桶记录逐个查找。

这些 GPU copy 与前面的 publish 在同一构建流按顺序执行。Indexer.done 和后端 pending 事件记录在相关提交之后。下一 chunk 查询该 layer/head 前调用 wait_event，因此不会先读到已登记但尚未写完的数据。

## 17. 一个新增 1K 桶的完整例子

假设当前请求已构建 65536 个 token 的索引，这次长度变为 66560：

|项目|值|
|---|---|
|已有完整桶数|64|
|新的完整桶数|65|
|本次新增 bucket|64，零基编号|
|start|65536|
|处理逻辑位置|65536..66559|
|从原生请求页表读取|第 1024..1039 项，共 16 页|
|新分组结果|128 个 8-member 小组|
|重排页|16 页；视匹配情况复用或新分配|
|中心页|2 页，属于本次 lease|
|请求重排页表更新范围|`state.table[1024:1040]`|
|请求中心页表更新范围|`state.center_table[128:130]`|

若本次分配 lease=64，则中心池页为 128、129。重排页则可能是任意可用的 16 个物理页，不能从 lease 推出它们的编号。

后续只增加到 67000 时，runtime 的 complete 仍为 65，不会给不足 1K 的尾部另建完整桶。当前后端只在 rows=1024 时提交建桶；请求最终短尾没有新桶，decode 使用全量 attention。

## 18. 当前实现边界

- 每桶独立分组，不跨相邻桶重新聚类。
- 建桶不读取 V，不读取 Q；保留原生 KV，额外维护查询所需的重排 Index K、中心和成员映射。
- 归一化只在分组工作区内发生；中心是未归一化反量化 K 的均值。
- runtime 可包含 Graph 支持，但当前评测显式关闭。
- overlap 表示与后续模型工作有重叠机会，不保证任意负载下完全隐藏；下次查询未等到构建完成仍须等待。
- 本 README 是当前代码导读，未新增正确性、性能测试，也未把此前测试外推到多请求、Radix 或所有 GPU 架构。
