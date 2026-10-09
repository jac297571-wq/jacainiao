"""双色球：寻找“像大乐透绿球那样”的阶段性物理漏洞。输出 report_ssq.md。

数据：data/other/ssq.txt（17500.cn，2003 年至今，含红球出球顺序）。
双色球红球同为红色、蓝球同为蓝色，没有颜色效应；但摇奖球约两年更换一次，若某批球有制造偏差，
应表现为一段时期内某些号码持续偏热、换球后消失。
A. 滚动窗口（300 期）红/蓝球频率 χ²：定位异常时期
B. 无监督变点：每个红球号码出现率的 CUSUM（扫描 33 个号码，按最大值做多重校正）
C. 前瞻预测：只用最近 W 期估计号码权重（Plackett-Luce，收缩），逐 50 期滚动预测下一段，比特增益
D. 若 C 有正增益：换算成头奖概率倍数
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "other" / "ssq.txt"
rows = []
for line in SRC.open(encoding="utf-8", errors="ignore"):
    f = line.split()
    if len(f) < 15:
        continue
    try:
        red = [int(x) for x in f[2:8]]
        blue = int(f[8])
        order = [int(x) for x in f[9:15]]
    except ValueError:
        continue
    if sorted(order) != sorted(red):
        order = None
    rows.append((f[0], f[1], red, blue, order))
N = len(rows)
ISS = [r[0] for r in rows]
DATE = [r[1] for r in rows]
R = np.array([r[2] for r in rows])
B = np.array([r[3] for r in rows])
XR = np.zeros((N, 33))
XR[np.arange(N)[:, None], R - 1] = 1
XB = np.zeros((N, 16))
XB[np.arange(N), B - 1] = 1
rng = np.random.default_rng(33)
out = []
p = out.append
p("# 双色球：寻找阶段性物理漏洞\n")
p(f"数据：{N} 期（{ISS[0]} ~ {ISS[-1]}），有出球顺序的 {sum(r[4] is not None for r in rows)} 期。\n")


def chi(c):
    e = c.mean()
    return float(((c - e) ** 2 / e).sum())


# ---------- A ----------
p("## A. 滚动窗口（300 期，步长 150）\n")
p("| 起 | 止 | 红球 χ²（df32） | p | 最热红球 | 蓝球 χ²（df15） | p |")
p("|---|---|---|---|---|---|---|")
from scipy import stats

Wn = 300
flag = []
for end in range(Wn, N + 1, 150):
    cr = XR[end - Wn:end].sum(0)
    cb = XB[end - Wn:end].sum(0)
    xr, xb = chi(cr), chi(cb)
    # 无放回 6/33 的 χ² 方差略小于多项式，用 df 近似并乘 (33-1)/(33-6) 修正
    pr = stats.chi2.sf(xr * (32 / 27), 32)
    pb = stats.chi2.sf(xb, 15)
    flag.append((pr, ISS[end - Wn], ISS[end - 1]))
    p(f"| {ISS[end - Wn]} | {ISS[end - 1]} | {xr:.1f} | {pr:.4f}{' ←' if pr < 0.01 else ''} | {(np.argsort(-cr)[:3] + 1).tolist()} | {xb:.1f} | {pb:.4f}{' ←' if pb < 0.01 else ''} |")

# ---------- B ----------
p("\n## B. 每个红球号码的无监督变点（CUSUM，按 33 个号码的最大值校正）\n")


def cusum_max(x, lo=150):
    n = len(x)
    c = np.cumsum(x - x.mean())
    k = np.arange(1, n)
    st = np.abs(c[:-1]) / np.sqrt(k * (n - k) / n) / max(x.std(), 1e-9)
    j = int(np.argmax(st[lo:n - lo])) + lo
    return j + 1, st[j]


obs = [cusum_max(XR[:, i]) for i in range(33)]
null_max = []
for _ in range(300):
    perm = rng.permutation(N)
    null_max.append(max(cusum_max(XR[perm, i])[1] for i in range(0, 33, 3)))  # 抽样 11 个号码近似
null_max = np.array(null_max)
p("| 号码 | 变点期号 | 统计量 | 之前出现率 | 之后出现率 | 校正 p |")
p("|---|---|---|---|---|---|")
for i in np.argsort([-o[1] for o in obs])[:6]:
    j, s = obs[i]
    pv = (np.sum(null_max >= s) + 1) / (len(null_max) + 1)
    p(f"| {i + 1:02d} | {ISS[j]}（{DATE[j]}） | {s:.2f} | {XR[:j, i].mean():.3f} | {XR[j:, i].mean():.3f} | {pv:.3f} |")
p(f"\n（理论出现率 6/33 = {6 / 33:.3f}）")

# ---------- C ----------
p("\n## C. 前瞻预测：只用最近 W 期估计红球号码权重，滚动预测下一段\n")
p("增益 > 0 代表“最近一段时间偏热的号码，接下来也确实更容易出”，即存在可利用的阶段性偏差。\n")
OR = [r[4] for r in rows]
has = np.array([o is not None for o in OR])
Oarr = np.array([o if o is not None else [0] * 6 for o in OR]) - 1


def nll(th, idx):
    w = np.exp(th)
    v, g = 0.0, np.zeros(33)
    for i in idx:
        avail = np.ones(33, bool)
        for x in Oarr[i]:
            den = w[avail].sum()
            v -= th[x] - np.log(den)
            g += np.where(avail, w, 0) / den
            g[x] -= 1
            avail[x] = False
    return v, g


def fit(idx, lam):
    return minimize(lambda t: (nll(t, idx)[0] + lam * t @ t / 2, nll(t, idx)[1] + lam * t), np.zeros(33), jac=True,
                    method="L-BFGS-B").x


valid = np.flatnonzero(has)
p("| 窗口 W | 收缩 λ | 评估期数 | 总增益（比特） | z | 相当于头奖概率倍数（6 红） |")
p("|---|---|---|---|---|---|")
best = None
for W in (150, 300, 600):
    for lam in (30, 100):
        gains = []
        start = W
        for s in range(start, len(valid), 50):
            tr = valid[max(0, s - W):s]
            te = valid[s:s + 50]
            th = fit(tr, lam)
            for i in te:
                gains.append((nll(np.zeros(33), [i])[0] - nll(th, [i])[0]) / np.log(2))
        g = np.array(gains)
        z = g.sum() / (g.std(ddof=1) * np.sqrt(len(g)))
        mult = 2 ** g.mean()
        p(f"| {W} | {lam} | {len(g)} | {g.sum():+.2f} | {z:+.2f} | ×{mult:.4f} |")
        if best is None or g.sum() > best[0]:
            best = (g.sum(), W, lam, z)
p(f"\n最好的设置：W={best[1]}，λ={best[2]}，总增益 {best[0]:+.2f} 比特（z={best[3]:+.2f}）。")

text = "\n".join(out) + "\n"
(ROOT / "report_ssq.md").write_text(text, encoding="utf-8")
print(text)
