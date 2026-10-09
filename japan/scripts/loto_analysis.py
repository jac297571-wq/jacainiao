"""日本ロト系列（ロト6 / ロト7 / ミニロト）第一轮检验：整体均匀性、变点、セット球（A-J）指纹、套号可预测性、分套模型样本外打分。
用法：python3 japan/scripts/loto6_analysis.py [loto6|loto7|miniloto] → japan/report_<game>.md
零假设：每回从 1-K 等概率无放回抽 M 个本数字。
"""
import sys
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
GAME = sys.argv[1] if len(sys.argv) > 1 else "loto6"
TITLE, K, M = {"loto6": ("ロト6", 43, 6), "loto7": ("ロト7", 37, 7), "miniloto": ("ミニロト", 31, 5)}[GAME]
D = list(csv.DictReader((ROOT / "data" / f"{GAME}.csv").open(encoding="utf-8")))
N = len(D)
R = np.array([[int(d[f"n{i}"]) for i in range(1, M + 1)] for d in D])
HAS_ORDER = "o1" in D[0]
O = np.array([[int(d[f"o{i}"]) for i in range(1, M + 1)] for d in D]) - 1 if HAS_ORDER else None
SETS = np.array(["ABCDEFGHIJ".index(d["setball"]) for d in D])
DATE = np.array([d["date"] for d in D])
X = np.zeros((N, K))
X[np.arange(N)[:, None], R - 1] = 1
rng = np.random.default_rng(43)
SIMS = 2000
out = []
p = out.append


def chi(c):
    e = c.mean()
    return float(((c - e) ** 2 / e).sum())


def mc_chi(n, obs):
    s = np.array([chi(np.bincount(np.argsort(rng.random((n, K)), 1)[:, :M].ravel(), minlength=K)) for _ in range(SIMS)])
    return (np.sum(s >= obs) + 1) / (SIMS + 1)


p(f"# {TITLE} 第一轮检验\n")
p(f"数据：{N} 回（{DATE[0]} ~ {DATE[-1]}），每回本数字 {M} 个（1–{K}），セット球 A–J 共 10 套。\n")

# ---------- A. 整体与分时段均匀性 ----------
p("## A. 号码频率均匀性（分时段）\n")
p("| 时段 | 回数 | χ² | p | 最热 | 最冷 |")
p("|---|---|---|---|---|---|")
edges = np.linspace(0, N, 6).astype(int)
for a, b in list(zip(edges[:-1], edges[1:])) + [(0, N)]:
    c = X[a:b].sum(0)
    x2 = chi(c)
    hot = np.argsort(-c)[:3] + 1
    cold = np.argsort(c)[:3] + 1
    p(f"| 第{a + 1}–{b}回（{DATE[a][:4]}–{DATE[b - 1][:4]}） | {b - a} | {x2:.1f} | {mc_chi(b - a, x2):.3f} | {hot.tolist()} | {cold.tolist()} |")

# ---------- B. 变点（三等分号段个数） ----------
LO, HI = K // 3, K - K // 3 + 1
p(f"\n## B. 无监督变点（每回 1–{LO} 号个数 / {HI}–{K} 号个数 / 合计）\n")


def cusum(x, lo=60):
    n = len(x)
    c = np.cumsum(x - x.mean())
    k = np.arange(1, n)
    st = np.abs(c[:-1]) / np.sqrt(k * (n - k) / n)
    j = int(np.argmax(st[lo:n - lo])) + lo
    return j + 1, st[j]


for name, f in ((f"1–{LO} 号个数", lambda Rm: (Rm <= LO).sum(1)), (f"{HI}–{K} 号个数", lambda Rm: (Rm >= HI).sum(1)),
                ("本数字合计", lambda Rm: Rm.sum(1))):
    x = f(R).astype(float)
    j, s = cusum(x)
    perm = [cusum(rng.permutation(x))[1] for _ in range(1000)]
    pv = (np.sum(np.array(perm) >= s) + 1) / 1001
    p(f"- {name}：最强变点 第{D[j]['id']}回（{DATE[j]}），前 {x[:j].mean():.3f} / 后 {x[j:].mean():.3f}，置换 p={pv:.3f}")

# ---------- C. セット球 ----------
p("\n## C. セット球（A–J）检验\n")
p("### C1. 每套单独的号码频率\n")
p("| セット | 回数 | χ² | p |")
p("|---|---|---|---|")
pvals = []
for s in range(10):
    m = SETS == s
    x2 = chi(X[m].sum(0))
    pv = mc_chi(m.sum(), x2)
    pvals.append(pv)
    p(f"| {'ABCDEFGHIJ'[s]} | {m.sum()} | {x2:.1f} | {pv:.3f} |")
p(f"\n10 套中 p<0.05 的有 {sum(v < 0.05 for v in pvals)} 套（纯随机期望 0.5 套）。")


def between(Xm, lab):
    T = np.vstack([Xm[lab == s].sum(0) for s in range(10)])
    E = np.outer(T.sum(1), T.sum(0)) / T.sum()
    return float(((T - E) ** 2 / E).sum())


obs = between(X, SETS)
sims = [between(X, rng.permutation(SETS)) for _ in range(1000)]
p(f"\n### C2. 10 套之间号码分布是否不同（套号置换检验）\n\n10×{K} 列联表 χ²={obs:.1f}，置换 p={(np.sum(np.array(sims) >= obs) + 1) / 1001:.3f}")

p("\n### C3. 每套的“号码指纹”能否前后复现？（セット球理论的核心）\n")
p(f"每套按时间分前后两半，比较 {K} 个号码偏差的相关；真有固定指纹则 r 显著为正。\n")
p("| セット | 前半 | 后半 | r | 置换 p |")
p("|---|---|---|---|---|")
rs = []
for s in range(10):
    idx = np.flatnonzero(SETS == s)
    h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
    a, b = X[h1].sum(0), X[h2].sum(0)
    a, b = a - a.mean(), b - b.mean()
    r = np.corrcoef(a, b)[0, 1]
    pr = (np.sum([np.corrcoef(a, rng.permutation(b))[0, 1] >= r for _ in range(2000)]) + 1) / 2001
    rs.append(r)
    p(f"| {'ABCDEFGHIJ'[s]} | 第{D[h1[0]]['id']}–{D[h1[-1]]['id']}回 | 第{D[h2[0]]['id']}–{D[h2[-1]]['id']}回 | {r:+.3f} | {pr:.3f} |")
p(f"\n10 套平均 r={np.mean(rs):+.3f}（纯随机期望 ≈0）；Fisher 合并：z={np.mean(np.arctanh(rs)) * np.sqrt(10 * (K - 3)):+.2f}")

p("\n### C4. 下回用哪套能预测吗？\n")
T = np.zeros((10, 10), int)
for a, b in zip(SETS[:-1], SETS[1:]):
    T[a, b] += 1
x2, pv, _, _ = stats.chi2_contingency(T)
rep = np.trace(T) / T.sum()
p(f"转移矩阵独立性 χ²={x2:.1f}，p={pv:.3f}；同套连用比例 {rep:.3f}（随机 0.100）。")
last_seen = {}
gaps = []
for i, s in enumerate(SETS):
    if s in last_seen:
        gaps.append(i - last_seen[s])
    last_seen[s] = i
gc = Counter(gaps)
p(f"同一套再次使用的间隔分布（回）：" + "，".join(f"{k}:{gc[k]}" for k in sorted(gc)[:15]) + " …")
# 用“最久没用的套”预测下一回命中率
hit, tot = 0, 0
last_seen = {s: -1 for s in range(10)}
for i, s in enumerate(SETS):
    if i > 20:
        cand = min(last_seen, key=lambda k: last_seen[k])
        hit += cand == s
        tot += 1
    last_seen[s] = i
p(f"\n“下回 = 最久没用过的那套”的命中率：{hit / tot:.3f}（随机 0.100）")
# 最近 k 回内用过的套是否被排除
for k in (1, 3, 5):
    excl, tot2 = 0, 0
    for i in range(k, N):
        excl += SETS[i] in SETS[i - k:i]
        tot2 += 1
    p(f"- 下回套号出现在前 {k} 回中的比例：{excl / tot2:.3f}（若独立随机约 {1 - 0.9 ** k:.3f}）")

# ---------- D. 分套 Plackett-Luce 模型样本外打分 ----------
if not HAS_ORDER:
    (ROOT / f"report_{GAME}.md").write_text("\n".join(out) + "\n\n（该彩种数据无出球顺序，跳过 D 节出球模型。）\n", encoding="utf-8")
    print("\n".join(out))
    sys.exit(0)
p("\n## D. 分套号码权重模型的样本外打分（按出球顺序，5 折时间分块）\n")


def nll(th, Ob, lab, per_set):
    LW = th.reshape(10, K) if per_set else np.tile(th, (10, 1))
    W = np.exp(LW)[lab]
    n = len(Ob)
    avail = np.ones((n, K))
    v = 0.0
    G = np.zeros((n, K))
    for k in range(M):
        den = (W * avail).sum(1)
        pk = Ob[:, k]
        v -= (np.log(W[np.arange(n), pk]) - np.log(den)).sum()
        G += W * avail / den[:, None]
        G[np.arange(n), pk] -= 1
        avail[np.arange(n), pk] = 0
    if per_set:
        g = np.vstack([G[lab == s].sum(0) for s in range(10)]).ravel()
    else:
        g = G.sum(0)
    return v, g


def fit(Ob, lab, per_set, lam):
    n0 = 10 * K if per_set else K
    def f(t):
        v, g = nll(t, Ob, lab, per_set)
        return v + lam * t @ t / 2, g + lam * t
    return minimize(f, np.zeros(n0), jac=True, method="L-BFGS-B").x


p("| 模型 | 正则 λ | CV 增益（比特） | 各块 |")
p("|---|---|---|---|")
ed = np.linspace(0, N, 6).astype(int)
for per_set, lams in ((False, (100, 300, 1000)), (True, (100, 300, 1000))):
    for lam in lams:
        g = []
        for f_ in range(5):
            te = np.arange(ed[f_], ed[f_ + 1])
            tr = np.setdiff1d(np.arange(N), te)
            th = fit(O[tr], SETS[tr], per_set, lam)
            n0 = 10 * K if per_set else K
            g.append((nll(np.zeros(n0), O[te], SETS[te], per_set)[0] - nll(th, O[te], SETS[te], per_set)[0]) / np.log(2))
        p(f"| {'每套各自的号码权重' if per_set else '共同号码权重'} | {lam} | {sum(g):+.2f} | {' '.join(f'{x:+.1f}' for x in g)} |")

text = "\n".join(out) + "\n"
(ROOT / f"report_{GAME}.md").write_text(text, encoding="utf-8")
print(text)
