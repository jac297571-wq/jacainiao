"""大乐透深度检验：分段 / 切片 / 时序 / 组合 / 反向（杀号）策略，并做多重检验校正与跨期复现。

输出 report_deep.md。依赖 numpy、scipy。
零假设：每期前区从 1-35 等概率无放回抽 5 个，后区从 1-12 抽 2 个，期与期独立。
"""
import csv
from collections import Counter
from datetime import date
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
D = list(csv.DictReader((ROOT / "data" / "dlt_history.csv").open(encoding="utf-8")))
R = np.array([sorted(int(d[f"r{i}"]) for i in range(1, 6)) for d in D])
B = np.array([sorted(int(d[f"b{i}"]) for i in (1, 2)) for d in D])
DATES = [date.fromisoformat(d["date"]) for d in D]
N = len(D)
SIMS = 1000
rng = np.random.default_rng(2026)

OUT = []


def p(*a):
    OUT.append(" ".join(str(x) for x in a))


def onehot(rows, k):
    X = np.zeros((len(rows), k), dtype=np.int8)
    np.put_along_axis(X, rows - 1, 1, axis=1)
    return X


def rand_draws(n, k, m):
    return np.argsort(rng.random((n, k)), axis=1)[:, :m] + 1


# ---------- 分段 ----------
SEGS = {
    "全部": np.arange(N),
    "2016前": np.array([i for i in range(N) if DATES[i].year <= 2016]),
    "2017后": np.array([i for i in range(N) if DATES[i].year >= 2017]),
    "2017-21": np.array([i for i in range(N) if 2017 <= DATES[i].year <= 2021]),
    "2022-26": np.array([i for i in range(N) if DATES[i].year >= 2022]),
}
SEG_NAMES = list(SEGS)

# ---------- 统计量（基于 one-hot 矩阵，便于蒙特卡洛） ----------


def chi_freq(X):
    c = X.sum(0)
    e = c.sum() / len(c)
    return float(((c - e) ** 2 / e).sum())


def chi_pairs(X):
    M = X.T.astype(np.int32) @ X
    iu = np.triu_indices(X.shape[1], 1)
    c = M[iu]
    e = c.sum() / len(c)
    return float(((c - e) ** 2 / e).sum())


def lag_overlap(X, k):
    return float((X[k:] * X[:-k]).sum(1).mean())


def num_autocorr(X):
    """每个号码出现序列的 lag-1 自相关，z² 之和"""
    Y = X - X.mean(0)
    num = (Y[1:] * Y[:-1]).sum(0)
    den = (Y ** 2).sum(0)
    r = num / den
    return float((r ** 2 * len(X)).sum())


def red_blue_dep(XR, XB):
    M = XR.T.astype(np.int32) @ XB
    e = np.outer(M.sum(1), M.sum(0)) / M.sum()
    return float(((M - e) ** 2 / e).sum())


def mc(stat, obs, n, two_sided=False):
    """蒙特卡洛 p 值：在 n 期真随机开奖下，统计量 >= 观测值的比例"""
    sims = np.array([stat(onehot(rand_draws(n, 35, 5), 35), onehot(rand_draws(n, 12, 2), 12)) for _ in range(SIMS)])
    if two_sided:
        m = sims.mean()
        return (np.sum(np.abs(sims - m) >= abs(obs - m)) + 1) / (SIMS + 1)
    return (np.sum(sims >= obs) + 1) / (SIMS + 1)


TESTS = []  # (类别, 名称, 分段, 统计量文本, p)


def add(cat, name, seg, txt, pv):
    TESTS.append((cat, name, seg, txt, float(pv)))


# ---------- 1. 号码频率 / 组合 / 时序（蒙特卡洛） ----------
for seg, idx in SEGS.items():
    XR, XB = onehot(R[idx], 35), onehot(B[idx], 12)
    n = len(idx)
    defs = [
        ("频率", "前区号码频率 χ²", lambda a, b: chi_freq(a), False),
        ("频率", "后区号码频率 χ²", lambda a, b: chi_freq(b), False),
        ("组合", "前区两两同出 χ²(595对)", lambda a, b: chi_pairs(a), False),
        ("组合", "后区两码组合 χ²(66对)", lambda a, b: chi_pairs(b), False),
        ("组合", "前后区相关性 χ²(35×12)", lambda a, b: red_blue_dep(a, b), False),
        ("时序", "前区各号码 lag1 自相关", lambda a, b: num_autocorr(a), False),
        ("时序", "后区各号码 lag1 自相关", lambda a, b: num_autocorr(b), False),
    ] + [
        ("时序", f"前区与 {k} 期前重号数", (lambda k: lambda a, b: lag_overlap(a, k))(k), True) for k in (1, 2, 3, 5, 10)
    ] + [("时序", "后区与上期重号数", lambda a, b: lag_overlap(b, 1), True)]
    for cat, name, f, two in defs:
        obs = f(XR, XB)
        add(cat, name, seg, f"{obs:.3f}", mc(f, obs, n, two))

# ---------- 2. 单期切片特征 vs 精确理论分布 ----------
COMBOS = np.array(list(combinations(range(1, 36), 5)))
BCOMBOS = np.array(list(combinations(range(1, 13), 2)))
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}


def ac_value(r):
    return len({b - a for a, b in combinations(r, 2)}) - 4


def feats(Rm):
    return {
        "和值": Rm.sum(1),
        "跨度": Rm[:, 4] - Rm[:, 0],
        "奇数个数": (Rm % 2).sum(1),
        "大号(≥18)个数": (Rm >= 18).sum(1),
        "质数个数": np.isin(Rm, list(PRIMES)).sum(1),
        "连号对数": (np.diff(Rm, axis=1) == 1).sum(1),
        "三区比(1-12/13-24/25-35)": ((Rm <= 12).sum(1) * 100 + ((Rm > 12) & (Rm <= 24)).sum(1) * 10 + (Rm > 24).sum(1)),
        "尾数种类数": np.array([len(set(x % 10)) for x in Rm]),
        "AC值": np.array([ac_value(x) for x in Rm]),
        "最小号": Rm[:, 0], "第2位": Rm[:, 1], "第3位": Rm[:, 2], "第4位": Rm[:, 3], "最大号": Rm[:, 4],
    }


def bfeats(Bm):
    return {"后区和值": Bm.sum(1), "后区间距": Bm[:, 1] - Bm[:, 0], "后区奇数个数": (Bm % 2).sum(1)}


TH = {k: Counter(v.tolist()) for k, v in feats(COMBOS).items()}
TH.update({k: Counter(v.tolist()) for k, v in bfeats(BCOMBOS).items()})


def gof(obs_vals, theory):
    """理论分布拟合优度：按取值顺序合并期望 < 5 的格子"""
    tot = sum(theory.values())
    n = len(obs_vals)
    oc = Counter(obs_vals.tolist())
    keys = sorted(theory)
    O, E, bo, be = [], [], 0, 0.0
    for k in keys:
        bo += oc.get(k, 0)
        be += theory[k] / tot * n
        if be >= 5:
            O.append(bo); E.append(be); bo, be = 0, 0.0
    if be > 0:
        O[-1] += bo; E[-1] += be
    O, E = np.array(O), np.array(E)
    x2 = float(((O - E) ** 2 / E).sum())
    return x2, len(O) - 1, stats.chi2.sf(x2, len(O) - 1)


for seg, idx in SEGS.items():
    for name, v in list(feats(R[idx]).items()) + list(bfeats(B[idx]).items()):
        x2, df, pv = gof(v, TH[name])
        add("切片", name, seg, f"χ²={x2:.1f}/df{df}", pv)

# ---------- 3. 遗漏（间隔）分布 vs 几何分布 ----------


def gap_test(X, prob):
    gaps = []
    for j in range(X.shape[1]):
        pos = np.flatnonzero(X[:, j])
        gaps += np.diff(pos).tolist()
    gaps = np.array(gaps)
    kmax = int(np.ceil(np.log(5 / len(gaps) / prob) / np.log(1 - prob)))  # 尾部期望>=5
    ks = np.arange(1, kmax)
    E = len(gaps) * prob * (1 - prob) ** (ks - 1)
    E = np.append(E, len(gaps) * (1 - prob) ** (kmax - 1))
    O = np.array([np.sum(gaps == k) for k in ks] + [np.sum(gaps >= kmax)])
    x2 = float(((O - E) ** 2 / E).sum())
    return x2, len(O) - 1, stats.chi2.sf(x2, len(O) - 1), gaps


for seg, idx in SEGS.items():
    for name, X, pr in (("前区遗漏分布", onehot(R[idx], 35), 1 / 7), ("后区遗漏分布", onehot(B[idx], 12), 1 / 6)):
        x2, df, pv, _ = gap_test(X, pr)
        add("时序", name, seg, f"χ²={x2:.1f}/df{df}", pv)

# 和值序列 Ljung-Box
for seg, idx in SEGS.items():
    s = R[idx].sum(1).astype(float)
    s -= s.mean()
    n, h = len(s), 10
    ac = [np.dot(s[k:], s[:-k]) / np.dot(s, s) for k in range(1, h + 1)]
    Q = n * (n + 2) * sum(a * a / (n - k) for k, a in enumerate(ac, 1))
    add("时序", "前区和值 Ljung-Box(10)", seg, f"Q={Q:.1f}", stats.chi2.sf(Q, h))

# ---------- 4. 星期分段 ----------
WD = {0: "周一", 2: "周三", 5: "周六"}
for seg in ("全部", "2017后"):
    idx = SEGS[seg]
    for w, wn in WD.items():
        sub = np.array([i for i in idx if DATES[i].weekday() == w])
        if len(sub) < 100:
            continue
        x2 = chi_freq(onehot(R[sub], 35))
        add("分段", f"{wn}前区频率 χ²", seg, f"χ²={x2:.1f}(n={len(sub)})", mc(lambda a, b: chi_freq(a), x2, len(sub)))

# ---------- 5. 多重检验校正（BH-FDR） ----------
pv = np.array([t[4] for t in TESTS])
order = np.argsort(pv)
m = len(pv)
q = np.empty(m)
q[order] = np.minimum.accumulate((pv[order] * m / np.arange(1, m + 1))[::-1])[::-1]
q = np.minimum(q, 1)

# ---------- 6. 反向 / 杀号 / 选号策略回测 ----------


def strategy_backtest(lo, hi, W=100):
    """返回 {策略: (成功次数, 次数, 理论概率)} —— 杀号成功=所杀号码下期全部未出"""
    XR = onehot(R, 35)
    out = {}

    def rec(k, ok, p0):
        a = out.setdefault(k, [0, 0, p0])
        a[0] += ok; a[1] += 1

    last_seen = np.full(36, -1)
    for i in range(N):
        if i >= max(lo, W) and i < hi:
            nxt = set(R[i].tolist())
            prev = R[i - 1]
            c = XR[i - W:i].sum(0)
            hot = list(np.argsort(-c, kind="stable")[:5] + 1)
            cold = list(np.argsort(c, kind="stable")[:5] + 1)
            omit = sorted(range(1, 36), key=lambda x: last_seen[x])[:5]  # 遗漏最久
            p5 = comb(30, 5) / comb(35, 5)
            p1 = 30 / 35
            rec("杀上期5码", not nxt & set(prev.tolist()), p5)
            rec("杀最热5码", not nxt & set(hot), p5)
            rec("杀最冷5码", not nxt & set(cold), p5)
            rec("杀遗漏最久5码", not nxt & set(omit), p5)
            for nm, v in (("杀(首+尾)%35", (prev[0] + prev[4]) % 35 or 35),
                          ("杀 尾-首", prev[4] - prev[0]),
                          ("杀 和值%35", prev.sum() % 35 or 35),
                          ("杀 第3位", prev[2])):
                rec(nm, v not in nxt, p1)
            # 选号：命中 >=2 个前区
            ph2 = sum(comb(5, k) * comb(30, 5 - k) for k in range(2, 6)) / comb(35, 5)
            rec("选最热5码·中≥2", len(nxt & set(hot)) >= 2, ph2)
            rec("选最冷5码·中≥2", len(nxt & set(cold)) >= 2, ph2)
            rec("选遗漏最久5码·中≥2", len(nxt & set(omit)) >= 2, ph2)
            rec("选上期5码·中≥2", len(nxt & set(prev.tolist())) >= 2, ph2)
        for x in R[i]:
            last_seen[x] = i
    return out


# ---------- 7. 滚动窗口 χ²：定位偏差何时消失 ----------
def rolling(Wn=300, step=100):
    rows = []
    for end in range(Wn, N + 1, step):
        X = onehot(R[end - Wn:end], 35)
        x2 = chi_freq(X)
        hi = X[:, 28:].sum() / X.sum() / (7 / 35)  # 29-35 号出现率相对理论
        rows.append((D[end - Wn]["issue"], D[end - 1]["issue"], x2, stats.chi2.sf(x2, 34), hi))
    return rows


# ================= 生成报告 =================
p("# 大乐透深度检验报告\n")
p(f"样本 {N} 期（{D[0]['issue']} ~ {D[-1]['issue']}），共 {m} 项检验，蒙特卡洛 {SIMS} 次。")
p("p<0.05 只是“可疑”；做了 {} 项检验，纯随机也会冒出约 {} 个。真正算数的是 **q（BH-FDR 校正后）<0.05** 且在 2017-21 与 2022-26 两段**都复现**。\n".format(m, round(m * 0.05)))

p("## A. 总览：每个检验在各分段的 p 值\n")
p("| 类别 | 检验 | " + " | ".join(SEG_NAMES) + " |")
p("|---|---|" + "---|" * len(SEG_NAMES))
keys = []
for t in TESTS:
    if (t[0], t[1]) not in keys:
        keys.append((t[0], t[1]))
look = {(t[0], t[1], t[2]): (t[4], q[i]) for i, t in enumerate(TESTS)}


def cell(pv_, qv):
    s = f"{pv_:.3f}"
    if qv < 0.05:
        return f"**{s}**‼"
    if pv_ < 0.05:
        return f"{s}*"
    return s


for c_, n_ in keys:
    p(f"| {c_} | {n_} | " + " | ".join(cell(*look[(c_, n_, s)]) if (c_, n_, s) in look else "—" for s in SEG_NAMES) + " |")
p("\n`*` p<0.05（未校正）；`**‼**` 校正后仍显著（q<0.05）。\n")

p("## B. 校正后仍显著的结果\n")
sig = [(TESTS[i], q[i]) for i in np.argsort(pv) if q[i] < 0.05]
if sig:
    for t, qv in sig:
        p(f"- [{t[2]}] {t[1]}：{t[3]}，p={t[4]:.2g}，q={qv:.2g}")
else:
    p("- 无")
post = [(TESTS[i], q[i]) for i in range(m) if TESTS[i][2] in ("2017后", "2017-21", "2022-26") and q[i] < 0.05]
p(f"\n**2017 年后的分段里，校正后显著的检验数：{len(post)}**\n")

p("## C. 滚动窗口（300 期，步长 100）：偏差何时消失\n")
p("| 起 | 止 | 前区 χ² | p | 29-35号出现率/理论 |")
p("|---|---|---|---|---|")
for a, b, x2, pv_, hi in rolling():
    flag = " ←" if pv_ < 0.01 else ""
    p(f"| {a} | {b} | {x2:.1f} | {pv_:.4f}{flag} | {hi:.2f} |")
p("")

for label, lo, hi in (("全部年份", 0, N), ("2017 年后", int(SEGS["2017后"][0]), N)):
    bt = strategy_backtest(lo, hi)
    p(f"## D. 反向 / 杀号 / 选号策略回测 · {label}\n")
    p("| 策略 | 成功率 | 理论 | z | p |")
    p("|---|---|---|---|---|")
    for k, (ok, n, p0) in bt.items():
        z = (ok / n - p0) / np.sqrt(p0 * (1 - p0) / n)
        p(f"| {k} | {ok / n:.4f} | {p0:.4f} | {z:+.2f} | {2 * stats.norm.sf(abs(z)):.3f} |")
    p("")

p("## E. 2016 年前偏差的形态（切片特征均值 vs 理论）\n")
p("| 特征 | 理论均值 | 2016前 | 2017后 |")
p("|---|---|---|---|")
thf = feats(COMBOS)
f_pre, f_post = feats(R[SEGS["2016前"]]), feats(R[SEGS["2017后"]])
for k in ("和值", "跨度", "奇数个数", "大号(≥18)个数", "最小号", "第3位", "最大号", "AC值"):
    p(f"| {k} | {thf[k].mean():.2f} | {f_pre[k].mean():.2f} | {f_post[k].mean():.2f} |")

p("\n## F. 小号偏移跟踪（每期 1-12 号个数，理论 5×12/35=1.714）\n")
p("| 时段 | 期数 | 均值 | z | 单边 p |")
p("|---|---|---|---|---|")
var_s = 5 * (12 / 35) * (23 / 35) * (30 / 34)
YR = np.array([d.year for d in DATES])
for lo_, hi_ in ((2007, 2016), (2017, 2021), (2022, 2024), (2025, 2026)):
    s_ = (R[(YR >= lo_) & (YR <= hi_)] <= 12).sum(1)
    z_ = (s_.mean() - 60 / 35) / np.sqrt(var_s / len(s_))
    p(f"| {lo_}-{hi_} | {len(s_)} | {s_.mean():.3f} | {z_:+.2f} | {stats.norm.sf(z_):.4f} |")

(ROOT / "report_deep.md").write_text("\n".join(OUT) + "\n", encoding="utf-8")
print("\n".join(OUT))
