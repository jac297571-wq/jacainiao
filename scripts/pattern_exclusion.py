"""检验“前置排除”：某种形态（连号、等差、纯小号…）刚开出后，接下来几期是否更少再出？

对每种形态比较：
  理论概率（全部 324,632 种前区组合精确计数）
  无条件实际出现率
  上一形态出现后第 1/2/3 期的出现率
并回测策略：“最近 k 期出现过的形态，下一期排除”——被排除空间占比 vs 下一期真正落在排除区的比例。
两者相等 = 排除没有任何作用（只是同比例丢掉中奖机会）。
"""
import csv
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
D = list(csv.DictReader((ROOT / "data" / "dlt_history.csv").open(encoding="utf-8")))
R = [tuple(sorted(int(d[f"r{i}"]) for i in range(1, 6))) for d in D]
B = [tuple(sorted(int(d[f"b{i}"]) for i in (1, 2))) for d in D]
YEAR = [int(d["date"][:4]) for d in D]


def max_run(r):
    best = cur = 1
    for a, b in zip(r, r[1:]):
        cur = cur + 1 if b - a == 1 else 1
        best = max(best, cur)
    return best


def max_ap(r):
    """最长等差子序列长度（公差>0）"""
    s, best = set(r), 2
    for a, b in combinations(r, 2):
        d, n, x = b - a, 2, b + (b - a)
        while x in s:
            n, x = n + 1, x + d
        best = max(best, n)
    return best


PATTERNS = {
    "有连号(≥1对)": lambda r: max_run(r) >= 2,
    "三连号及以上": lambda r: max_run(r) >= 3,
    "≥2组连号": lambda r: sum(b - a == 1 for a, b in zip(r, r[1:])) >= 2,
    "含3项等差": lambda r: max_ap(r) >= 3,
    "含4项等差": lambda r: max_ap(r) >= 4,
    "纯小号(全≤17)": lambda r: r[-1] <= 17,
    "偏小(最大号≤22)": lambda r: r[-1] <= 22,
    "和值≤65": lambda r: sum(r) <= 65,
    "全奇或全偶": lambda r: len({x % 2 for x in r}) == 1,
}

ALL = list(combinations(range(1, 36), 5))
THEORY = {k: sum(map(f, ALL)) / len(ALL) for k, f in PATTERNS.items()}

out = []
p = out.append


def segment(lo):
    idx = [i for i in range(len(D)) if YEAR[i] >= lo]
    return idx


for label, lo in (("全部年份", 2007), ("2017 年后", 2017)):
    idx = segment(lo)
    p(f"\n## 形态出现后，下几期是否更少再出？· {label}（{len(idx)} 期）\n")
    p("| 形态 | 理论概率 | 实际出现率 | 出现后第1期 | 第2期 | 第3期 | 第1期 p |")
    p("|---|---|---|---|---|---|---|")
    for k, f in PATTERNS.items():
        flag = [f(R[i]) for i in range(len(D))]
        base = np.mean([flag[i] for i in idx])
        row = [k, f"{THEORY[k]:.3f}", f"{base:.3f}"]
        p1 = None
        for lag in (1, 2, 3):
            after = [flag[i] for i in idx if i - lag >= 0 and flag[i - lag]]
            rate = np.mean(after)
            # 原假设：仍为理论概率（独立）
            pv = stats.binomtest(int(sum(after)), len(after), THEORY[k]).pvalue
            if lag == 1:
                p1 = pv
            row.append(f"{rate:.3f} (n={len(after)})")
        row.append(f"{p1:.3f}")
        p("| " + " | ".join(row) + " |")

# ---- 排除策略回测 ----
p("\n## 回测“排除最近 k 期出现过的形态”（2017 年后）\n")
p("“排除空间占比”是被排除的组合占全部组合的比例；“落入排除区”是下一期真实开奖恰好落在被排除区里的比例。")
p("若落入排除区 ≈ 排除空间占比，说明排除只是同比例丢掉中奖机会，单注中奖率没有任何提高。\n")
p("| k | 平均排除空间占比 | 下一期落入排除区 | 比值 | 理论上单注中奖率变化 |")
p("|---|---|---|---|---|")
STRONG = ["三连号及以上", "≥2组连号", "含4项等差", "纯小号(全≤17)", "和值≤65", "全奇或全偶"]
combo_flags = {k: np.array([PATTERNS[k](c) for c in ALL]) for k in STRONG}
idx = segment(2017)
for kwin in (1, 3, 5):
    space, fell = [], []
    for i in idx:
        recent = {k for k in STRONG for j in range(i - kwin, i) if PATTERNS[k](R[j])}
        if not recent:
            continue
        mask = np.zeros(len(ALL), bool)
        for k in recent:
            mask |= combo_flags[k]
        space.append(mask.mean())
        fell.append(any(PATTERNS[k](R[i]) for k in recent))
    s, fl = np.mean(space), np.mean(fell)
    p(f"| {kwin} | {s:.3f} | {fl:.3f} | {fl / s:.2f} | ×{(1 - fl) / (1 - s):.3f} |")

# ---- 后区 ----
p("\n## 后区：上期后区对下期的影响（2017 年后）\n")
p("| 条件 | 下期出现率 | 理论 | p |")
p("|---|---|---|---|")
pairs = [(B[i - 1], B[i]) for i in idx]
rep = [len(set(a) & set(b)) >= 1 for a, b in pairs]
th_rep = 1 - comb(10, 2) / comb(12, 2)
p(f"| 与上期后区至少重1个 | {np.mean(rep):.3f} | {th_rep:.3f} | {stats.binomtest(sum(rep), len(rep), th_rep).pvalue:.3f} |")
cons_prev = [b for a, b in pairs if a[1] - a[0] == 1]
th_cons = 11 / 66
rate = np.mean([b[1] - b[0] == 1 for b in cons_prev])
p(f"| 上期后区连号 → 下期后区连号 | {rate:.3f} (n={len(cons_prev)}) | {th_cons:.3f} | "
  f"{stats.binomtest(sum(b[1] - b[0] == 1 for b in cons_prev), len(cons_prev), th_cons).pvalue:.3f} |")
small_prev = [b for a, b in pairs if a[1] <= 6]
th_small = comb(6, 2) / 66
k_ = sum(b[1] <= 6 for b in small_prev)
p(f"| 上期后区全≤6 → 下期全≤6 | {k_ / len(small_prev):.3f} (n={len(small_prev)}) | {th_small:.3f} | "
  f"{stats.binomtest(k_, len(small_prev), th_small).pvalue:.3f} |")

text = "# 形态前置排除检验\n" + "\n".join(out) + "\n"
(ROOT / "report_pattern.md").write_text(text, encoding="utf-8")
print(text)
