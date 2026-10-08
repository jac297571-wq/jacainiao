"""按摇奖球套号（3 套）分组检验。输出 report_ballset.md。

数据：data/dlt_ballset.csv（福建体彩网 data_api，YaoJiangQiu 字段，2011 年起）。
假设：如果偏差来自物理（某些球偏重/偏轻），它应当（1）集中在某一套；（2）在同一套的不同时段可复现。
"""
import csv
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
rows = [r for r in csv.DictReader((ROOT / "data" / "dlt_ballset.csv").open(encoding="utf-8")) if r["ballset"] in "123" and r["ballset"]]
ISS = [r["issue"] for r in rows]
SET = np.array([int(r["ballset"]) for r in rows])
YR = np.array([int(r["date"][:4]) for r in rows])
SORTED = [list(map(int, r["sorted"].split())) for r in rows]
R = np.array([s[:5] for s in SORTED])
B = np.array([s[5:] for s in SORTED])
X = np.zeros((len(R), 35))
for i, r in enumerate(R):
    X[i, r - 1] = 1
XB = np.zeros((len(B), 12))
for i, b in enumerate(B):
    XB[i, b - 1] = 1
rng = np.random.default_rng(7)
SIMS = 2000

out = []
p = out.append
SEGS = {"2011-16": (2011, 2016), "2017后": (2017, 2099), "2022后": (2022, 2099), "全部": (2011, 2099)}


def chi_freq(M):
    c = M.sum(0)
    e = c.sum() / len(c)
    return float(((c - e) ** 2 / e).sum())


def mc_freq(n, k, m, obs):
    """n 期、k 选 m 的真随机下 χ² >= obs 的比例"""
    sims = np.empty(SIMS)
    for s in range(SIMS):
        idx = np.argsort(rng.random((n, k)), axis=1)[:, :m]
        c = np.bincount(idx.ravel(), minlength=k)
        e = n * m / k
        sims[s] = ((c - e) ** 2 / e).sum()
    return (np.sum(sims >= obs) + 1) / (SIMS + 1)


p("# 摇奖球套号检验\n")
p(f"数据：{len(rows)} 期（{ISS[0]} ~ {ISS[-1]}），套号分布 {dict(sorted(Counter(SET.tolist()).items()))}\n")

# ---------- A. 每套单独：均匀性、小号、连号 ----------
P_CONS = 1 - comb(31, 5) / comb(35, 5)
var_s = 5 * (12 / 35) * (23 / 35) * (30 / 34)
p("## A. 每套单独检验\n")
p("| 时段 | 套 | 期数 | 前区 χ² p | 后区 χ² p | 每期 1-12 号个数(理论1.714) | z | 有连号比例(理论0.477) | z |")
p("|---|---|---|---|---|---|---|---|---|")
for sname, (lo, hi) in SEGS.items():
    for s in (1, 2, 3):
        m = (SET == s) & (YR >= lo) & (YR <= hi)
        n = m.sum()
        pf = mc_freq(n, 35, 5, chi_freq(X[m]))
        pb = mc_freq(n, 12, 2, chi_freq(XB[m]))
        sm = (R[m] <= 12).sum(1)
        zs = (sm.mean() - 60 / 35) / np.sqrt(var_s / n)
        cs = (np.diff(R[m], axis=1) == 1).any(1)
        zc = (cs.mean() - P_CONS) / np.sqrt(P_CONS * (1 - P_CONS) / n)
        p(f"| {sname} | 第{s}套 | {n} | {pf:.3f} | {pb:.3f} | {sm.mean():.3f} | {zs:+.2f} | {cs.mean():.3f} | {zc:+.2f} |")

# ---------- B. 三套之间是否不同（置换检验） ----------
p("\n## B. 三套之间号码分布是否不同（套号标签置换检验）\n")
p("统计量：3×35（后区 3×12）列联表 χ²。把套号标签随机打乱 2000 次得到零分布。\n")
p("| 时段 | 前区 χ² | p | 后区 χ² | p |")
p("|---|---|---|---|---|")


def between(M, labels):
    T = np.vstack([M[labels == s].sum(0) for s in (1, 2, 3)])
    E = np.outer(T.sum(1), T.sum(0)) / T.sum()
    return float(((T - E) ** 2 / E).sum())


for sname, (lo, hi) in SEGS.items():
    m = (YR >= lo) & (YR <= hi)
    lab = SET[m]
    res = []
    for M in (X[m], XB[m]):
        obs = between(M, lab)
        sims = np.array([between(M, rng.permutation(lab)) for _ in range(SIMS)])
        res += [obs, (np.sum(sims >= obs) + 1) / (SIMS + 1)]
    p(f"| {sname} | {res[0]:.1f} | {res[1]:.3f} | {res[2]:.1f} | {res[3]:.3f} |")

# ---------- C. 同一套的号码偏差能否跨时段复现 ----------
p("\n## C. 同一套的“号码指纹”能否跨时段复现？\n")
p("把每套的开奖按时间分成前后两半，计算 35 个号码各自的偏差（实际 − 期望），看前半段与后半段的相关系数。")
p("真有偏重/偏轻的球，相关应显著为正。p 值来自置换（打乱后半段号码）。\n")
p("| 时段 | 套 | 前半 | 后半 | 前区相关 r | p | 后区相关 r | p |")
p("|---|---|---|---|---|---|---|---|")


def dev(M):
    c = M.sum(0)
    return c - c.mean()


def perm_r(a, b):
    r = np.corrcoef(a, b)[0, 1]
    sims = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(SIMS)])
    return r, (np.sum(sims >= r) + 1) / (SIMS + 1)


for sname in ("2011-16", "2017后", "全部"):
    lo, hi = SEGS[sname]
    for s in (1, 2, 3):
        idx = np.flatnonzero((SET == s) & (YR >= lo) & (YR <= hi))
        h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
        rf, pf = perm_r(dev(X[h1]), dev(X[h2]))
        rb, pb = perm_r(dev(XB[h1]), dev(XB[h2]))
        p(f"| {sname} | 第{s}套 | {ISS[h1[0]]}~{ISS[h1[-1]]} | {ISS[h2[0]]}~{ISS[h2[-1]]} | {rf:+.3f} | {pf:.3f} | {rb:+.3f} | {pb:.3f} |")

# 跨套对照：不同套之间同期指纹的相关（应约为 0；若也正，说明是共同因素而非套特有）
p("\n对照：同时段不同套之间的相关（若偏差是套特有，应≈0）\n")
p("| 时段 | 套对 | 前区 r | p |")
p("|---|---|---|---|")
for sname in ("2011-16", "2017后"):
    lo, hi = SEGS[sname]
    for a, b in ((1, 2), (1, 3), (2, 3)):
        ma = (SET == a) & (YR >= lo) & (YR <= hi)
        mb = (SET == b) & (YR >= lo) & (YR <= hi)
        r, pv = perm_r(dev(X[ma]), dev(X[mb]))
        p(f"| {sname} | {a}-{b} | {r:+.3f} | {pv:.3f} |")

# ---------- D. 套号能否预测 ----------
p("\n## D. 下一期用哪套能预测吗？\n")
T = np.zeros((3, 3), int)
for a, b in zip(SET[:-1], SET[1:]):
    T[a - 1, b - 1] += 1
p("转移矩阵（行=上期套号，列=本期套号）：\n")
p("| 上期\\本期 | 1 | 2 | 3 | 同套连用比例 |")
p("|---|---|---|---|---|")
for i in range(3):
    p(f"| {i + 1} | {T[i, 0]} | {T[i, 1]} | {T[i, 2]} | {T[i, i] / T[i].sum():.3f} |")
x2, pv, _, _ = stats.chi2_contingency(T)
p(f"\n独立性 χ²={x2:.1f}，p={pv:.3f}（理论同套连用比例 1/3）")
runs = Counter()
cur = 1
for a, b in zip(SET[:-1], SET[1:]):
    if a == b:
        cur += 1
    else:
        runs[cur] += 1
        cur = 1
p(f"连续使用同一套的长度分布：{dict(sorted(runs.items()))}")

text = "\n".join(out) + "\n"
(ROOT / "report_ballset.md").write_text(text, encoding="utf-8")
print(text)
