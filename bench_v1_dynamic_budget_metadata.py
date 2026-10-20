"""Benchmark-only budget input; production metadata logic otherwise unchanged."""
import triton as tr
import triton.language as tl
@tr.jit
def budget_metadata(LENS, NT, FLENS, TAKES, DG_LENS, CENTER_LENS, TAIL_ROWS, BUDGET,
                   NP: tl.constexpr, TOPK: tl.constexpr, NUM: tl.constexpr,
                   DEN: tl.constexpr, SINK: tl.constexpr, WINDOW: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    n = tl.load(LENS + row)
    pages = tl.maximum(n-1,0) // 1024 * 16
    take = tl.minimum(pages, tl.load(BUDGET))
    sink = tl.minimum(SINK, n)
    tail = tl.maximum(sink, tl.minimum(tl.maximum(n-1,0) // 1024 * 1024, tl.maximum(sink, n - WINDOW)))
    fine = take * 64 + sink + n - tail
    tl.store(FLENS + row, fine)
    tl.store(TAKES + row, take)
    tl.store(DG_LENS + row, tl.maximum(fine, 1))
    tl.store(CENTER_LENS + row, tl.maximum(pages * 8, 1))
    # Share bytes only after comparing current device lengths AND all involved
    # physical page IDs. This remains correct after migration, growth and replay.
    i = tl.arange(0, 128)
    page = tl.where(i == 0, 0, tail // 64 + i - 1)
    valid = (n > 0) & ((i == 0) | ((i > 0) & (page < ((n + 63) // 64)))) & (page < NP)
    mine = tl.load(NT + row * NP + page, valid, other=-1)
    first = tl.load(NT + page, valid, other=-1)
    share = (n == tl.load(LENS)) & (tl.sum((mine != first).to(tl.int32), 0) == 0)
    tl.store(TAIL_ROWS + row, tl.where(share, 0, row))


class BudgetMetadata:
 def __init__(self,budget):self.budget=budget
 def __getitem__(self,grid):
  def launch(*args,**kwargs):return budget_metadata[grid](*args[:7],self.budget,*args[7:],**kwargs)
  return launch
