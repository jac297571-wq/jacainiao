"""无预设的结构发掘。输出 report_explore.md。

A. 随机矩阵谱：期×号码指示矩阵的相关矩阵特征值 vs 纯随机零分布；滞后互协方差奇异值（时序结构）
B. 出球邻近图谱：出球顺序中“i 之后紧跟 j”的超额，奇偶期两半互验；奇异值 vs 零分布
C. 频谱：01-11 个数、和值等序列的周期图，Fisher g 检验
D. 变点：在 2007–2026 全序列上无监督地找机制切换点（二分分割 + 置换检验）
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(11)
SIMS = 500

H = list(csv.DictReader((ROOT / "data" / "dlt_history.csv").open(encoding="utf-8")))
HR = np.array([sorted(int(d[f"r{i}"]) for i in range(1, 6)) for d in H])
HD = np.array([d["date"] for d in H])
HI = [d["issue"] for d in H]
BS = [r for r in csv.DictReader((ROOT / "data" / "dlt_ballset.csv").open(encoding="utf-8")) if r["ballset"] in ("1", "2", "3")]
OD = np.array([r["date"] for r in BS])
ORD = np.array([list(map(int, r["order"].split()))[:5] for r in BS])
SET = np.array([int(r["ballset"]) for r in BS])

NEW = HD >= "2019-09-07"
out = []
p = out.append
p("# 无预设结构发掘\n")


def onehot(R, k=35):
    X = np.zeros((len(R), k))
    X[np.arange(len(R))[:, None], R - 1] = 1
    return X


def rand_draws(n):
    return np.argsort(rng.random((n, 35)), axis=1)[:, :5] + 1


# ---------------- A. 随机矩阵谱 ----------------
def spectrum(X):
    C = np.corrcoef(X, rowvar=False)
    return np.sort(np.linalg.eigvalsh(C))[::-1]


def lag_sv(X, lag):
    Z = (X - X.mean(0)) / X.std(0)
    M = Z[lag:].T @ Z[:-lag] / (len(Z) - lag)
    return np.linalg.svd(M, compute_uv=False)


p("## A. 随机矩阵谱：数据里有没有任何方向的隐藏结构？\n")
for name, mask in (("新机 2019-09 起", NEW), ("2018 起", HD >= "2018"), ("2007-2016", HD < "2017")):
    X = onehot(HR[mask])
    n = len(X)
    ev = spectrum(X)
    sims = np.array([spectrum(onehot(rand_draws(n))) for _ in range(SIMS)])
    lag_obs = {L: lag_sv(X, L) for L in (1, 2, 3)}
    lag_sims = {L: np.array([lag_sv(onehot(rand_draws(n)), L) for _ in range(SIMS // 2)]) for L in (1, 2, 3)}
    p(f"### {name}（{n} 期）\n")
    p("| 特征值序号 | 实际 | 随机 95% 上限 | p（≥实际） |")
    p("|---|---|---|---|")
    for k in range(4):
        pv = (np.sum(sims[:, k] >= ev[k]) + 1) / (SIMS + 1)
        p(f"| λ{k + 1} | {ev[k]:.3f} | {np.percentile(sims[:, k], 95):.3f} | {pv:.3f} |")
    # 第一特征向量的形状
    C = np.corrcoef(X, rowvar=False)
    w, V = np.linalg.eigh(C)
    v1 = V[:, -1] * np.sign(V[:, -1].sum() or 1)
    corr_num = np.corrcoef(v1, np.arange(1, 36))[0, 1]
    top = np.argsort(-np.abs(v1))[:8] + 1
    p(f"\n第一主方向与号码大小的相关 r={corr_num:+.2f}；载荷最大的号码：{', '.join(f'{x:02d}' for x in top)}")
    p("\n期与期之间（滞后互协方差最大奇异值）：")
    for L in (1, 2, 3):
        pv = (np.sum(lag_sims[L][:, 0] >= lag_obs[L][0]) + 1) / (SIMS // 2 + 1)
        p(f"- 滞后 {L} 期：σ1={lag_obs[L][0]:.4f}，随机 95% 上限 {np.percentile(lag_sims[L][:, 0], 95):.4f}，p={pv:.3f}")
    p("")


# ---------------- B. 出球邻近图谱 ----------------
def follow_resid(O):
    """返回 35×35 标准化残差 (O-E)/sqrt(E)，i 之后紧跟 j"""
    Ob = np.zeros((35, 35))
    E = np.zeros((35, 35))
    for o in O - 1:
        remaining = np.ones(35, bool)
        for k in range(4):
            remaining[o[k]] = False
            E[o[k], remaining] += 1 / remaining.sum()
            Ob[o[k], o[k + 1]] += 1
    S = (Ob + Ob.T) - (E + E.T)
    Es = E + E.T
    Z = np.divide(S, np.sqrt(Es), out=np.zeros_like(S), where=Es > 0)
    np.fill_diagonal(Z, 0)
    return Z


def rand_orders(n):
    return np.argsort(rng.random((n, 35)), axis=1)[:, :5] + 1


iu = np.triu_indices(35, 1)
p("## B. 出球邻近图谱：哪些球倾向于“前后脚”出来？\n")
p("统计“球 i 出来后紧跟球 j”（双向合并）相对零假设的标准化残差。真实结构应当在奇数期、偶数期两半里都出现（两半残差正相关）。\n")
p("| 时段 | 期数 | 两半残差相关 r | p | 残差矩阵最大奇异值 | 随机 95% 上限 | p |")
p("|---|---|---|---|---|---|---|")
for name, mask in (("新机", OD >= "2019-09-07"), ("2011-2019.09", OD < "2019-09-07"), ("全部 2011 起", OD >= "2011")):
    O = ORD[mask]
    n = len(O)
    Za, Zb = follow_resid(O[0::2]), follow_resid(O[1::2])
    r = np.corrcoef(Za[iu], Zb[iu])[0, 1]
    rs = []
    for _ in range(200):
        Rr = rand_orders(n)
        rs.append(np.corrcoef(follow_resid(Rr[0::2])[iu], follow_resid(Rr[1::2])[iu])[0, 1])
    pr = (np.sum(np.array(rs) >= r) + 1) / 201
    Z = follow_resid(O)
    s1 = np.linalg.svd(Z, compute_uv=False)[0]
    ss = [np.linalg.svd(follow_resid(rand_orders(n)), compute_uv=False)[0] for _ in range(200)]
    ps = (np.sum(np.array(ss) >= s1) + 1) / 201
    p(f"| {name} | {n} | {r:+.3f} | {pr:.3f} | {s1:.2f} | {np.percentile(ss, 95):.2f} | {ps:.3f} |")
    if name == "新机":
        Znew = Z
d = np.abs(iu[0] - iu[1])
p("\n新机时期，残差随号码距离 |i−j| 的平均值（码放相邻 = 距离 1）：\n")
p("| 号码距离 | 1 | 2 | 3 | 4–6 | 7–12 | 13+ |")
p("|---|---|---|---|---|---|---|")
bins = [(1, 1), (2, 2), (3, 3), (4, 6), (7, 12), (13, 34)]
p("| 平均残差 | " + " | ".join(f"{Znew[iu][(d >= a) & (d <= b)].mean():+.3f}" for a, b in bins) + " |")
top = np.argsort(-Znew[iu])[:6]
p("\n新机时期最常“前后脚”的号码对：" + "，".join(f"{iu[0][i] + 1:02d}↔{iu[1][i] + 1:02d}(z={Znew[iu][i]:+.1f})" for i in top))

# ---------------- C. 频谱 ----------------
p("\n## C. 频谱：偏差会不会随周期起伏？\n")
p("对序列做周期图，Fisher g 检验最高峰是否超出随机（不预设周期）。开奖间隔平均约 2.33 天，所以 39 期 ≈ 1 季度、13 期 ≈ 1 个月。\n")
p("| 序列 | 时段 | 最高峰周期（期） | 约合天数 | Fisher g p | 次高峰周期 |")
p("|---|---|---|---|---|---|")


def fisher_g(x):
    x = (x - x.mean()) / x.std()
    I = np.abs(np.fft.rfft(x)) ** 2
    I = I[1:len(x) // 2]
    freqs = np.fft.rfftfreq(len(x))[1:len(x) // 2]
    g = I.max() / I.sum()
    m = len(I)
    from math import comb as _c
    pv = sum((-1) ** (j - 1) * _c(m, j) * (1 - j * g) ** (m - 1) for j in range(1, int(1 / g) + 1))  # Fisher 精确公式
    pv = float(min(max(pv, 0.0), 1.0))
    order = np.argsort(-I)
    return 1 / freqs[order[0]], pv, 1 / freqs[order[1]]


series = {
    "每期 01-11 个数": lambda R: (R <= 11).sum(1).astype(float),
    "和值": lambda R: R.sum(1).astype(float),
    "跨度": lambda R: (R[:, 4] - R[:, 0]).astype(float),
    "连号对数": lambda R: (np.diff(R, axis=1) == 1).sum(1).astype(float),
}
for sname, f in series.items():
    for name, mask in (("新机", NEW), ("全部", HD >= "2007")):
        per, pv, per2 = fisher_g(f(HR[mask]))
        p(f"| {sname} | {name} | {per:.1f} | {per * 2.33:.0f} | {pv:.3f} | {per2:.1f} |")

# ---------------- D. 变点 ----------------
p("\n## D. 变点：不告诉算法任何日期，让它自己找机制切换点\n")
p("序列：每期 01-11 个数（2007–2026 全部 2932 期）。二分分割，CUSUM 统计量，置换检验 p<0.01 且每段 ≥100 期才切。\n")
s_all = (HR <= 11).sum(1).astype(float)


def cusum_split(x):
    n = len(x)
    c = np.cumsum(x - x.mean())
    k = np.arange(1, n)
    stat = np.abs(c[:-1]) / np.sqrt(k * (n - k) / n)
    j = int(np.argmax(stat[99:n - 100])) + 99 if n > 200 else None
    return (j + 1, stat[j]) if j is not None else (None, 0)


def segment(lo, hi, found):
    x = s_all[lo:hi]
    if len(x) < 200:
        return
    j, st = cusum_split(x)
    if j is None:
        return
    perm = []
    for _ in range(300):
        xp = rng.permutation(x)
        perm.append(cusum_split(xp)[1])
    pv = (np.sum(np.array(perm) >= st) + 1) / 301
    if pv < 0.01:
        found.append((lo + j, pv, x[:j].mean(), x[j:].mean()))
        segment(lo, lo + j, found)
        segment(lo + j, hi, found)


found = []
segment(0, len(s_all), found)
found.sort()
p("| 切换点（期号） | 日期 | p | 之前 01-11 均值 | 之后 01-11 均值 |")
p("|---|---|---|---|---|")
for j, pv, a, b in found:
    p(f"| {HI[j]} | {HD[j]} | {pv:.3f} | {a:.3f} | {b:.3f} |")
p(f"\n（理论均值 {55 / 35:.3f}。已知事件：2019-09-07 新摇奖机启用；2022-08-27 摇奖机异常。）")
bounds = [0] + [j for j, *_ in found] + [len(s_all)]
p("\n各段：\n")
p("| 起 | 止 | 期数 | 01-11 均值 | ×理论 |")
p("|---|---|---|---|---|")
for a, b in zip(bounds[:-1], bounds[1:]):
    m = s_all[a:b].mean()
    p(f"| {HI[a]} | {HI[b - 1]} | {b - a} | {m:.3f} | ×{m / (55 / 35):.3f} |")

text = "\n".join(out) + "\n"
(ROOT / "report_explore.md").write_text(text, encoding="utf-8")
print(text)
