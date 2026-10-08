"""Plackett-Luce 出球模型：从出球顺序反推每颗球的“有效权重”，并做时间分块交叉验证。输出 report_pl.md。

模型：每期前区按出球顺序逐个不放回抽取，第 k 个球取到 i 的概率 = w_i / Σ_{剩余 j} w_j。
  log w_{i,s} = a_i + b_{i,s}    a_i：号码 i 在三套球上共同的倾向；b_{i,s}：第 s 套特有的偏差
  先验（等价于 L2 惩罚）：a ~ N(0, 1/λa)，b ~ N(0, 1/λb)
比较的模型：
  M0 均匀；M1 台阶（01-11 一个参数）；M2 每号独立 a_i；M3 a_i + 每套 b_{i,s}
评估：按时间切成 5 块，每块用其余块拟合、在本块上算对数似然（比特），相对 M0 的增益 > 0 才算真有预测力。
"""
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
rows = [r for r in csv.DictReader((ROOT / "data" / "dlt_ballset.csv").open(encoding="utf-8")) if r["ballset"] in ("1", "2", "3")]
ORDER = np.array([list(map(int, r["order"].split()))[:5] for r in rows]) - 1
BORDER = np.array([list(map(int, r["order"].split()))[5:] for r in rows]) - 1
SETS = np.array([int(r["ballset"]) - 1 for r in rows])
DATES = np.array([r["date"] for r in rows])
NEW = DATES >= "2019-09-07"


def design(order, sets, n_balls, kind):
    """返回 logw(params) 的构造函数和参数个数"""
    if kind == "step":
        return 1, lambda th: np.tile(np.where(np.arange(n_balls) < 11, th[0], 0.0), (3, 1))
    if kind == "free":
        return n_balls, lambda th: np.tile(th, (3, 1))
    if kind == "set":
        return n_balls * 4, lambda th: th[:n_balls][None] + th[n_balls:].reshape(3, n_balls)
    return 0, lambda th: np.zeros((3, n_balls))


def nll_grad(th, order, sets, n_balls, kind, lam_a, lam_b):
    npar, f = design(order, sets, n_balls, kind)
    LW = f(th)  # (3, n)
    W = np.exp(LW)[sets]  # (N, n)
    N, K = order.shape
    avail = np.ones((N, n_balls))
    nll = 0.0
    gW = np.zeros((N, n_balls))  # d nll / d logw (按期)
    rows_ = np.arange(N)
    for k in range(K):
        denom = (W * avail).sum(1)
        pick = order[:, k]
        nll -= (np.log(W[rows_, pick]) - np.log(denom)).sum()
        gW += W * avail / denom[:, None]
        gW[rows_, pick] -= 1
        avail[rows_, pick] = 0
    # 汇总到 (3, n)
    G = np.zeros((3, n_balls))
    for s in range(3):
        G[s] = gW[sets == s].sum(0)
    if kind == "step":
        g = np.array([G[:, :11].sum()])
        pen, gp = lam_a * th @ th / 2, lam_a * th
    elif kind == "free":
        g = G.sum(0)
        pen, gp = lam_a * th @ th / 2, lam_a * th
    elif kind == "set":
        g = np.concatenate([G.sum(0), G.ravel()])
        a, b = th[:n_balls], th[n_balls:]
        pen = lam_a * a @ a / 2 + lam_b * b @ b / 2
        gp = np.concatenate([lam_a * a, lam_b * b])
    else:
        return nll, np.zeros(0)
    return nll + pen, g + gp


def fit(order, sets, n_balls, kind, lam_a=1.0, lam_b=1.0):
    npar, _ = design(order, sets, n_balls, kind)
    if npar == 0:
        return np.zeros(0)
    r = minimize(nll_grad, np.zeros(npar), args=(order, sets, n_balls, kind, lam_a, lam_b), jac=True, method="L-BFGS-B")
    return r.x


def loglik_bits(th, order, sets, n_balls, kind):
    v, _ = nll_grad(th, order, sets, n_balls, kind, 0, 0) if kind != "uniform" else nll_grad(np.zeros(0), order, sets, n_balls, kind, 0, 0)
    return -v / np.log(2)


def cv(order, sets, n_balls, kind, lam_a, lam_b, folds=5):
    N = len(order)
    edges = np.linspace(0, N, folds + 1).astype(int)
    tot = 0.0
    per_fold = []
    for f in range(folds):
        te = np.arange(edges[f], edges[f + 1])
        tr = np.setdiff1d(np.arange(N), te)
        th = fit(order[tr], sets[tr], n_balls, kind, lam_a, lam_b)
        g = loglik_bits(th, order[te], sets[te], n_balls, kind) - loglik_bits(np.zeros(0), order[te], sets[te], n_balls, "uniform")
        per_fold.append(g)
        tot += g
    return tot, per_fold


out = []
p = out.append
p("# Plackett-Luce 出球模型：每颗球的有效权重\n")
for label, mask, order_all, nb in (("新机（2019-09-07 起）· 前区", NEW, ORDER, 35),
                                    ("旧机（2011-01 ~ 2019-09）· 前区", ~NEW, ORDER, 35),
                                    ("新机 · 后区", NEW, BORDER, 12)):
    o, s = order_all[mask], SETS[mask]
    p(f"\n## {label}（{len(o)} 期）\n")
    p("交叉验证：5 个时间块，逐块留出；增益为相对均匀模型的比特数，> 0 表示能预测样本外。\n")
    p("| 模型 | 正则 λ | CV 增益(比特) | 各块增益 |")
    p("|---|---|---|---|")
    configs = [("step", 0.01, 0)] if nb == 35 else []
    configs += [("free", lam, 0) for lam in (30, 100, 300, 1000)]
    configs += [("set", lam, lb) for lam in (100, 300) for lb in (100, 300, 1000)]
    best = None
    for kind, la, lb in configs:
        tot, pf = cv(o, s, nb, kind, la, lb)
        name = {"step": "M1 台阶(01-11)", "free": "M2 每号独立", "set": "M3 每号+每套"}[kind]
        lamtxt = f"{la}" if kind != "set" else f"a:{la} b:{lb}"
        p(f"| {name} | {lamtxt} | {tot:+.2f} | {' '.join(f'{x:+.1f}' for x in pf)} |")
        print(label, name, lamtxt, f"{tot:+.2f}", flush=True)
        if best is None or tot > best[0]:
            best = (tot, kind, la, lb)
    # 全样本拟合最好的“共同倾向”模型并报告权重
    th = fit(o, s, nb, "free", 300 if nb == 35 else 100)
    w = np.exp(th)
    w /= w.mean()
    p(f"\n全样本拟合（M2，λ={300 if nb == 35 else 100}，收缩后）的相对权重：\n")
    p("```")
    for i in range(0, nb, 7):
        p("  ".join(f"{j + 1:02d}:{w[j]:.3f}" for j in range(i, min(i + 7, nb))))
    p("```")
    if nb == 35:
        th1 = fit(o, s, nb, "step", 0.01)
        p(f"\nM1 台阶参数：01–11 号权重 ×{np.exp(th1[0]):.3f}（相对其余号码）")
        # 自助法置信区间
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(200):
            idx = rng.integers(0, len(o), len(o))
            bs.append(np.exp(fit(o[idx], s[idx], nb, "step", 0.01)[0]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p(f"，95% 自助置信区间 [{lo:.3f}, {hi:.3f}]")

text = "\n".join(out) + "\n"
(ROOT / "report_pl.md").write_text(text, encoding="utf-8")
print(text)


# ---------- 严格版：分界点也只用训练数据选（嵌套交叉验证），消除“先看数据再选 11”的泄漏 ----------
def nll_cut(th, order, n_balls, cut):
    w = np.where(np.arange(n_balls) < cut, np.exp(th[0]), 1.0)
    N, K = order.shape
    avail = np.ones((N, n_balls))
    rows_ = np.arange(N)
    v = 0.0
    for k in range(K):
        denom = (w[None] * avail).sum(1)
        v -= (np.log(w[order[:, k]]) - np.log(denom)).sum()
        avail[rows_, order[:, k]] = 0
    return v


def fit_cut(order, cut):
    r = minimize(lambda t: nll_cut(t, order, 35, cut), [0.0], method="Nelder-Mead")
    return r.x, r.fun


def nested_cv(order, folds=5, cuts=range(3, 33)):
    N = len(order)
    edges = np.linspace(0, N, folds + 1).astype(int)
    res = []
    for f in range(folds):
        te = np.arange(edges[f], edges[f + 1])
        tr = np.setdiff1d(np.arange(N), te)
        best = min(((fit_cut(order[tr], c)[1], c) for c in cuts))
        cut = best[1]
        th, _ = fit_cut(order[tr], cut)
        g = (nll_cut(np.zeros(1), order[te], 35, cut) - nll_cut(th, order[te], 35, cut)) / np.log(2)
        res.append((cut, float(np.exp(th[0])), g))
    return res


nres = nested_cv(ORDER[NEW])
lines = ["\n## 严格版：嵌套交叉验证（新机 · 前区）\n",
         "每一折只用训练数据扫描分界点（1-3 … 1-32）并拟合台阶权重，再在留出块上打分。\n",
         "| 留出块 | 训练数据选出的分界 | 台阶权重 | 留出块增益(比特) |", "|---|---|---|---|"]
for f, (c, w, g) in enumerate(nres, 1):
    lines.append(f"| {f} | 01-{c:02d} | ×{w:.3f} | {g:+.2f} |")
lines.append(f"\n合计 {sum(g for *_, g in nres):+.2f} 比特，{sum(g > 0 for *_, g in nres)}/5 块为正。")
with (ROOT / "report_pl.md").open("a", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
