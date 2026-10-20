"""CPU-only policy, allocation sizing and lifecycle helpers."""
import math
import os
from dataclasses import dataclass

BACKENDS = ("qwen_v1_native", "qwen_dsa_native")
SINK, WINDOW, BUCKET, PAGE, GROUP, TOPK = 64, 2048, 4096, 64, 8, 2048
DECODE_QUERY_FIXED_PAGES = math.ceil((SINK + BUCKET) / PAGE)


def build_graph_enabled():
    return (os.environ.get('SGLANG_V1_INDEX_GRAPHS', '0') == '1'
            or os.environ.get('SGLANG_V1_BUILD_GRAPH', '0') == '1')


def query_graph_enabled(decode=False):
    return (os.environ.get('SGLANG_V1_INDEX_GRAPHS', '0') == '1'
            or os.environ.get('SGLANG_V1_QUERY_GRAPH', '0') == '1'
            or (decode and os.environ.get('SGLANG_V1_DECODE_QUERY_GRAPH', '0') == '1'))


def query_graph_sizes():
    sizes = tuple(sorted(set(int(n) for n in os.environ.get(
        'SGLANG_V1_QUERY_GRAPH_SIZES', '1,2,4,8,16').split(','))))
    if not sizes or sizes[0] != 1 or sizes[-1] > 1024:
        raise ValueError('Query graph sizes must start at 1 and not exceed 1024')
    return sizes


def query_graph_capacity(pages, context):
    step = int(os.environ.get('SGLANG_V1_GRAPH_INDEX_STEP_TOKENS', '65536'))
    if step < BUCKET or step % BUCKET:
        raise ValueError('Graph index capacity step must be a positive multiple of 4096')
    maximum = math.ceil(context / BUCKET) * (BUCKET // PAGE)
    if pages < 1 or pages > maximum:
        raise ValueError('Query pages exceed configured index capacity')
    step //= PAGE
    return min(maximum, math.ceil(pages / step) * step)


def query_graph_fixed_pages(config):
    return math.ceil((SINK + BUCKET + config.chunk) / PAGE)


def shared_query_graph_bytes(config, pages, rows):
    mib = 1 << 20
    amount = (8 << 20) + 32 * PAGE * rows * (
        pages + config.budget(pages) + query_graph_fixed_pages(config))
    return math.ceil(amount / mib) * mib


def shared_build_graph_bytes(heads, mode):
    if mode == 'reference':
        return max(1, math.ceil(heads / 8)) * (512 << 20)
    return (32 << 20) + heads * BUCKET * (128 * 4 + 32)


def enabled(args):
    return any(getattr(args, name, None) in BACKENDS for name in (
        "attention_backend", "prefill_attention_backend", "decode_attention_backend"))


def reserve_bytes():
    value = float(os.environ.get("SGLANG_V1_INDEX_POOL_GIB", "8"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("SGLANG_V1_INDEX_POOL_GIB must be finite and positive")
    return int(value * (1 << 30))


def workspace_bytes():
    """BF16 gathered K and V; independent of persistent index reservation."""
    rows = int(os.environ.get("SGLANG_V1_QUERY_TILE", "1024"))
    if rows < 1:
        raise ValueError("SGLANG_V1_QUERY_TILE must be positive")
    return rows * TOPK * 128 * 2 * 2


def decode_query_graph_reservation_bytes(pages, take):
    """Legacy single-query sizing helper; shared pools use shared_query_graph_bytes."""
    if pages < 1 or not 1 <= take <= pages:
        raise ValueError("Decode query graph requires nonempty valid page counts")
    # Graph-private logits, route/sort temporaries, allocator rounding and
    # stable Q/page-table/metadata inputs. No BF16 gather workspace here.
    mib = 1 << 20
    amount = (8 << 20) + 32 * PAGE * (pages + take + DECODE_QUERY_FIXED_PAGES)
    return ((amount + mib - 1) // mib) * mib


def decode_query_graph_budget(config, heads, layers, requests, *, enabled,
                             full_bucket_append):
    """Worst live graph charge for this rank's admitted request capacity.

    One capacity is retained per Q bucket, shared across requests/layers/heads.
    Admission includes prefill when the general query graph switch is enabled.
    """
    max_prefix = config.context - 1  # Current decode token occupies one slot.
    if (not enabled or not full_bucket_append or not config.share_active_pages
            or config.mode != "v1" or not (config.sparse(max_prefix, decode=True)
                or (query_graph_enabled() and config.sparse(max_prefix)))):
        return 0
    pages = (max_prefix // BUCKET) * (BUCKET // PAGE)
    if pages == 0:
        return 0  # No complete bucket: select_decode stays eager.
    # Graphs/workspaces are shared across requests, layers and heads. Each Q
    # bucket retains one capacity, replaced only when a larger capacity is needed.
    return sum(shared_query_graph_bytes(config, query_graph_capacity(pages, config.context), rows)
               for rows in query_graph_sizes())


def local_index_geometry(q_heads, kv_heads, head_dim, attn_tp, local_layers):
    """Only partition complete four-Q routing groups; never change selection."""
    if head_dim != 128 or q_heads != 4 * kv_heads:
        raise ValueError("Native routing requires head_dim=128 and four Q heads per KV head")
    if attn_tp < 1 or kv_heads % attn_tp:
        raise ValueError("TP must partition whole KV groups; replicated/split groups need collective routing")
    if local_layers < 1:
        raise ValueError("Native index requires a nonempty local attention stage")
    return kv_heads // attn_tp, local_layers


@dataclass(frozen=True)
class Config:
    mode: str
    context: int
    chunk: int
    threshold: int = 65536
    fraction: float = 0.5
    microbatch: int = 1024
    async_index: bool = True
    decode_threshold: int = 512 * 1024
    share_active_pages: bool = False

    def __post_init__(self):
        if self.mode not in ("v1", "dsa"):
            raise ValueError(self.mode)
        if self.share_active_pages and self.mode != "v1":
            raise ValueError("Active page sharing is a V1-only opt-in path")
        if self.context < TOPK or self.chunk < 1 or self.microbatch < 1:
            raise ValueError("Invalid context/chunk/microbatch")
        if not 0 < self.fraction <= 1:
            raise ValueError("Page fraction must be in (0, 1]")
        if self.decode_threshold < TOPK:
            raise ValueError("Decode threshold must be at least TopK")

    def sparse(self, prefix, decode=False):
        # prefix excludes the current query's newly written key.
        return (self.mode == "v1" and prefix > self.decode_threshold
                if decode else prefix >= self.threshold)

    @property
    def middle_pages(self):
        return max(1, math.ceil(max(0, self.context - SINK - WINDOW) / PAGE))

    @property
    def fixed_pages(self):
        return math.ceil((SINK + WINDOW + self.chunk) / PAGE)

    @property
    def fine_pages(self):
        return (self.middle_pages + self.fixed_pages if self.mode == "v1"
                else math.ceil(self.context / PAGE))

    @property
    def center_pages(self):
        return math.ceil(self.middle_pages * 8 / PAGE) if self.mode == "v1" else 0

    def state_bytes(self, heads):
        if self.mode == "dsa":
            return heads * self.fine_pages * PAGE * 132
        return heads * (self.fine_pages * (PAGE * 132 + PAGE * 4)
                        + self.center_pages * (PAGE * 132 + PAGE))

    def budget(self, pages):
        return min(pages, math.ceil(pages * self.fraction))


def bucket_plan(committed, prefix):
    """Return only new complete buckets and the mutable boundary bucket."""
    end = max(SINK, prefix - WINDOW)
    result = []
    while committed + BUCKET <= end:
        result.append((committed, committed + BUCKET, True))
        committed += BUCKET
    if committed < end:
        result.append((committed, end, False))
    return result, committed


class Ledger:
    def __init__(self, limit):
        self.limit, self.used = limit, 0

    def acquire(self, count):
        if self.used + count > self.limit:
            raise RuntimeError("Native index pool budget exceeded; reduce max-running-requests "
                               "or increase SGLANG_V1_INDEX_POOL_GIB and restart")
        self.used += count

    def release(self, count):
        if not 0 <= count <= self.used:
            raise RuntimeError("Index pool accounting underflow")
        self.used -= count
