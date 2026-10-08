"""全历史组合层面深度分析。输出 report_combo.md。

A. 重复与近似重复：完全重复、与历史最近邻的重合数、4 码/3 码子组合重复次数（蒙特卡洛对照）
B. 共现极值：最常同出的号码对、三码组合是否超出随机极值
C. 联合结构类：(三区比, 奇数个数, 连号对数) 联合分布 vs 精确理论
D. 组合评分的样本外检验（PIT）：逐期只用过去数据给组合打分，看真实开奖在随机组合中的百分位
E. 组合空间地图：全部 324,632 种前区组合与历史最近一期的重合数
"""
import csv
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
D = list(csv.DictReader((ROOT / "data" / "dlt_history.csv").open(encoding="utf-8")))
R = np.array([sorted(int(d[f"r{i}"]) for i in range(1, 6)) for d in D])
B = np.array([sorted(int(d[f"b{i}"]) for i in (1, 2)) for d in D])
DATES = np.array([d["date"] for d in D])
ISS = [d["issue"] for d in D]
N = len(R)
rng = np.random.default_rng(2026)
SIMS = 300
ALL = np.array(list(combinations(range(1, 36), 5)))
TOTAL = len(ALL)


def mask(rows):
    return (np.left_shift(1, rows - 1).astype(np.int64)).sum(-1)


def popcount(x):
    return np.bitwise_count(x.astype(np.uint64)) if hasattr(np, "bitwise_count") else np.array([bin(int(v)).count("1") for v in x])


def rand_draws(n):
    return np.sort(np.argsort(rng.random((n, 35)), axis=1)[:, :5] + 1, axis=1)


RM = mask(R)
out = []
p = out.append
p("# 全历史组合深度分析\n")
p(f"样本 {N} 期（{ISS[0]} ~ {ISS[-1]}），前区组合空间 {TOTAL:,}，前后区合计 {TOTAL * 66:,}。蒙特卡洛 {SIMS} 次。\n")


# ---------------- A. 重复与近似重复 ----------------
def nn_overlap(Mk):
    """每期与之前所有期的最大重合数分布"""
    res = np.zeros(len(Mk), int)
    for i in range(1, len(Mk)):
        res[i] = popcount(Mk[:i] & Mk[i]).max()
    return res[1:]


def subset_repeats(rows, k):
    c = Counter()
    for r in rows:
        for s in combinations(r.tolist(), k):
            c[s] += 1
    vals = np.array(list(c.values()))
    return int((vals >= 2).sum()), int(vals.max())


obs_nn = nn_overlap(RM)
obs_nn_c = Counter(obs_nn.tolist())
full_rep = N - len(set(RM.tolist()))
fb = Counter(zip(RM.tolist(), map(tuple, B.tolist())))
full_rep_fb = sum(v - 1 for v in fb.values() if v > 1)
sims_nn = {k: [] for k in range(6)}
sims_rep = []
sims_sub = {3: [], 4: []}
for _ in range(SIMS):
    Rs = rand_draws(N)
    Ms = mask(Rs)
    c = Counter(nn_overlap(Ms).tolist())
    for k in range(6):
        sims_nn[k].append(c.get(k, 0))
    sims_rep.append(N - len(set(Ms.tolist())))
    for k in (3, 4):
        sims_sub[k].append(subset_repeats(Rs, k))
p("## A. 重复与近似重复\n")
exp_rep = comb(N, 2) / TOTAL
p(f"- 前区完全重复的期数：**{full_rep}**（随机期望 ≈ {exp_rep:.1f}，模拟均值 {np.mean(sims_rep):.1f}）")
if full_rep:
    seen = {}
    for i, m in enumerate(RM.tolist()):
        if m in seen:
            p(f"  - {ISS[seen[m]]} 与 {ISS[i]}：{' '.join(f'{x:02d}' for x in R[i])}")
        seen.setdefault(m, i)
p(f"- 前后区完全重复（一等奖号码重现）：**{full_rep_fb}**（随机期望 ≈ {comb(N, 2) / (TOTAL * 66):.3f}）\n")
p("每期与之前所有期的**最大重合数**分布（越大说明越“像历史”）：\n")
p("| 最大重合 | 实际期数 | 随机模拟均值 | 模拟 90% 区间 | 双侧 p |")
p("|---|---|---|---|---|")
for k in range(6):
    s = np.array(sims_nn[k])
    o = obs_nn_c.get(k, 0)
    pv = 2 * min((np.sum(s >= o) + 1) / (SIMS + 1), (np.sum(s <= o) + 1) / (SIMS + 1))
    p(f"| {k} | {o} | {s.mean():.1f} | [{np.percentile(s, 5):.0f}, {np.percentile(s, 95):.0f}] | {min(pv, 1):.3f} |")
for k in (4, 3):
    o_rep, o_max = subset_repeats(R, k)
    s = np.array(sims_sub[k])
    p(f"\n{k} 码子组合：重复出现过的子组合 {o_rep} 个（模拟 {s[:, 0].mean():.0f}，p={(np.sum(s[:, 0] >= o_rep) + 1) / (SIMS + 1):.3f}）；"
      f"单个子组合最多出现 {o_max} 次（模拟最大值均值 {s[:, 1].mean():.1f}，p={(np.sum(s[:, 1] >= o_max) + 1) / (SIMS + 1):.3f}）")

# ---------------- B. 共现极值 ----------------
p("\n## B. 共现极值：最常/最少同出的号码对、三码组合\n")
X = np.zeros((N, 35))
X[np.arange(N)[:, None], R - 1] = 1
PM = X.T @ X
iu = np.triu_indices(35, 1)
pairs = PM[iu]
tri = Counter(s for r in R.tolist() for s in combinations(r, 3))
sims_pmax, sims_pmin, sims_tmax = [], [], []
for _ in range(SIMS):
    Rs = rand_draws(N)
    Xs = np.zeros((N, 35))
    Xs[np.arange(N)[:, None], Rs - 1] = 1
    ps = (Xs.T @ Xs)[iu]
    sims_pmax.append(ps.max())
    sims_pmin.append(ps.min())
    sims_tmax.append(max(Counter(s for r in Rs.tolist() for s in combinations(r, 3)).values()))
top_p = np.argsort(-pairs)[:5]
bot_p = np.argsort(pairs)[:5]
p(f"号码对期望同出 {N * 20 / (35 * 34):.1f} 次。")
p(f"- 最常同出：" + "，".join(f"{iu[0][i] + 1:02d}-{iu[1][i] + 1:02d}({pairs[i]:.0f})" for i in top_p)
  + f"；最大值 p={(np.sum(np.array(sims_pmax) >= pairs.max()) + 1) / (SIMS + 1):.3f}")
p(f"- 最少同出：" + "，".join(f"{iu[0][i] + 1:02d}-{iu[1][i] + 1:02d}({pairs[i]:.0f})" for i in bot_p)
  + f"；最小值 p={(np.sum(np.array(sims_pmin) <= pairs.min()) + 1) / (SIMS + 1):.3f}")
tm = tri.most_common(5)
p(f"- 三码组合期望 {N * comb(32, 2) / comb(35, 5):.2f} 次；最常：" + "，".join(f"{'-'.join(f'{x:02d}' for x in t)}({c})" for t, c in tm)
  + f"；最大值 p={(np.sum(np.array(sims_tmax) >= tm[0][1]) + 1) / (SIMS + 1):.3f}")

# ---------------- C. 联合结构类 ----------------


def cls(rows):
    z = (rows <= 12).sum(1) * 100 + ((rows > 12) & (rows <= 24)).sum(1) * 10 + (rows > 24).sum(1)
    odd = (rows % 2).sum(1)
    con = (np.diff(rows, axis=1) == 1).sum(1)
    return z * 100 + odd * 10 + np.minimum(con, 2)


TH = Counter(cls(ALL).tolist())
p("\n## C. 联合结构类：(三区比, 奇数个数, 连号对数) 联合分布\n")
p(f"共 {len(TH)} 个形态类，概率由全部组合精确计数。统计量 G（似然比），p 值用蒙特卡洛。\n")
p("| 时段 | 期数 | 出现过的类 | G | p | 偏离最大的类（实际/期望） |")
p("|---|---|---|---|---|---|")
keys = np.array(sorted(TH))
pk = np.array([TH[k] for k in keys]) / TOTAL
ALLC = cls(ALL)


def gstat(cnt, n):
    o = np.array([cnt.get(k, 0) for k in keys])
    e = pk * n
    nz = o > 0
    return 2 * (o[nz] * np.log(o[nz] / e[nz])).sum(), o, e


for name, m in (("全部", DATES >= "2007"), ("2016前", DATES < "2017"), ("2017后", DATES >= "2017"), ("新机2019-09起", DATES >= "2019-09-07")):
    n = m.sum()
    G, o, e = gstat(Counter(cls(R[m]).tolist()), n)
    sims = []
    for _ in range(SIMS):
        idx = rng.choice(TOTAL, n)
        sims.append(gstat(Counter(ALLC[idx].tolist()), n)[0])
    pv = (np.sum(np.array(sims) >= G) + 1) / (SIMS + 1)
    z = (o - e) / np.sqrt(e)
    j = np.argsort(-np.abs(z))[:3]
    desc = "；".join(f"区{keys[i] // 100:03d}/奇{keys[i] // 10 % 10}/连{keys[i] % 10}: {o[i]}/{e[i]:.1f}" for i in j)
    p(f"| {name} | {n} | {(o > 0).sum()} | {G:.1f} | {pv:.3f} | {desc} |")

# ---------------- D. 组合评分的样本外检验（PIT） ----------------
p("\n## D. 组合评分的样本外检验\n")
p("对每一期 t，只用 t 之前的数据给组合打分；把真实开奖的分数放进 5000 个随机组合的分数里，得到百分位。")
p("打分方法真有预测力 → 平均百分位 > 0.5；完全无用 → 百分位均匀分布（均值 0.5）。KS 检验均匀性，t 检验均值。\n")
K_RAND = 5000
START = 300
TRI3 = np.array(list(combinations(range(5), 3)))
PR2 = np.array(list(combinations(range(5), 2)))


def scores_at(t, C, Pc, T3, clsfreq):
    """C: (m,5) 组合；返回各打分方法的分数"""
    pair = Pc[C[:, PR2[:, 0]] - 1, C[:, PR2[:, 1]] - 1].sum(1)
    trip = T3[C[:, TRI3[:, 0]] - 1, C[:, TRI3[:, 1]] - 1, C[:, TRI3[:, 2]] - 1].sum(1)
    cm = mask(C)
    sim10 = popcount(cm[:, None] & RM[None, t - 10:t]).sum(1) if len(C) else 0
    cc = cls(C)
    cf = np.array([clsfreq.get(k, 0) / TH[k] for k in cc.tolist()])
    small = (C <= 11).sum(1)
    last = popcount(cm & RM[t - 1])
    return {"号码对共现支持": pair, "三码共现支持": trip, "与近10期相似度": sim10,
            "形态类历史频率/理论": cf, "01-11 个数": small, "与上期重号数": last}


Pc = np.zeros((35, 35))
T3 = np.zeros((35, 35, 35))
clsfreq = Counter()
pits = {}
for t in range(N):
    if t >= START:
        Cr = rand_draws(K_RAND)
        sr = scores_at(t, Cr, Pc, T3, clsfreq)
        so = scores_at(t, R[t:t + 1], Pc, T3, clsfreq)
        u = rng.random()
        for k in sr:
            # 随机化 PIT 处理并列分数
            less = (sr[k] < so[k][0]).mean()
            eq = (sr[k] == so[k][0]).mean()
            pits.setdefault(k, []).append(less + u * eq)
    r = R[t] - 1
    for a, b in combinations(r, 2):
        Pc[a, b] += 1
        Pc[b, a] += 1
    for a, b, c in combinations(r, 3):
        for x, y, z in ((a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)):
            T3[x, y, z] += 1
    clsfreq[int(cls(R[t:t + 1])[0])] += 1
dates_pit = DATES[START:]
p("| 打分方法 | 时段 | 期数 | 平均百分位 | t 检验 p | KS p |")
p("|---|---|---|---|---|---|")
for k, v in pits.items():
    v = np.array(v)
    for name, m in (("全部", dates_pit >= "2007"), ("2017后", dates_pit >= "2017"), ("新机", dates_pit >= "2019-09-07")):
        x = v[m]
        tp = stats.ttest_1samp(x, 0.5).pvalue
        kp = stats.kstest(x, "uniform").pvalue
        p(f"| {k} | {name} | {len(x)} | {x.mean():.4f} | {tp:.3f} | {kp:.3f} |")

# ---------------- E. 组合空间地图 ----------------
p("\n## E. 组合空间地图：每一种可能的前区组合，与历史上最接近的一期重几个号？\n")
AM = mask(ALL)
best = np.zeros(TOTAL, int)
for i in range(0, TOTAL, 20000):
    blk = AM[i:i + 20000]
    best[i:i + 20000] = np.max(popcount((blk[:, None] & RM[None, :]).ravel()).reshape(len(blk), N), axis=1)
bc = Counter(best.tolist())
p("| 与历史最接近一期的重合数 | 组合数 | 占比 |")
p("|---|---|---|")
for k in sorted(bc, reverse=True):
    p(f"| {k} | {bc[k]:,} | {bc[k] / TOTAL:.2%} |")
never = bc.get(2, 0) + bc.get(1, 0) + bc.get(0, 0)
p(f"\n- 已开出过的前区组合：{bc.get(5, 0):,} 种，占全部组合的 {bc.get(5, 0) / TOTAL:.2%}。")
p(f"- 和历史上任何一期都最多只重 2 个号的“处女地”组合：{never:,} 种。")
p("- 注意：在开奖真随机的前提下，无论是“已开过的”还是“处女地”组合，下一期开出的概率都完全相同（见 D 节的检验）。")

text = "\n".join(out) + "\n"
(ROOT / "report_combo.md").write_text(text, encoding="utf-8")
print(text)
