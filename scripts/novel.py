"""新切入点：出球顺序 / 混合不充分假设 / 预测模型的样本外打分。输出 report_novel.md。

数据：data/dlt_order.csv（17500.cn，2011 年起含出球顺序）、data/dlt_history.csv。
"""
import csv
from collections import Counter
from math import comb, log2
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OD = list(csv.DictReader((ROOT / "data" / "dlt_order.csv").open(encoding="utf-8")))
O = np.array([[int(d[f"o{i}"]) for i in range(1, 6)] for d in OD])
OB = np.array([[int(d["ob1"]), int(d["ob2"])] for d in OD])
OY = np.array([int(d["date"][:4]) for d in OD])

HD = list(csv.DictReader((ROOT / "data" / "dlt_history.csv").open(encoding="utf-8")))
H = np.array([sorted(int(d[f"r{i}"]) for i in range(1, 6)) for d in HD])
HY = np.array([int(d["date"][:4]) for d in HD])

out = []
p = out.append
SEGS = {"2011-16": (2011, 2016), "2017后": (2017, 2099), "2022后": (2022, 2099)}


def seg(Y, s):
    lo, hi = SEGS[s]
    return (Y >= lo) & (Y <= hi)


# ---------------- A. 出球顺序 ----------------
p("# 新切入点检验\n")
p(f"出球顺序数据：{len(OD)} 期（{OD[0]['issue']} ~ {OD[-1]['issue']}）\n")
p("## A. 按出球顺序：第 k 个出的球是否均匀？\n")
p("零假设下，第 k 个出的球在 35 个号码上均匀分布（逐位置 χ²，df=34）。也看 1–12 号在每个位置的占比（理论 0.343）。\n")
p("| 时段 | 位置 | χ² | p | 1-12 号占比 | z |")
p("|---|---|---|---|---|---|")
for s in SEGS:
    m = seg(OY, s)
    n = m.sum()
    for k in range(5):
        c = Counter(O[m, k].tolist())
        e = n / 35
        x2 = sum((c.get(i, 0) - e) ** 2 / e for i in range(1, 36))
        sm = (O[m, k] <= 12).mean()
        z = (sm - 12 / 35) / np.sqrt(12 / 35 * 23 / 35 / n)
        p(f"| {s} | 第{k + 1}球 | {x2:.1f} | {stats.chi2.sf(x2, 34):.3f} | {sm:.3f} | {z:+.2f} |")
    for k in range(2):
        c = Counter(OB[m, k].tolist())
        e = n / 12
        x2 = sum((c.get(i, 0) - e) ** 2 / e for i in range(1, 13))
        p(f"| {s} | 后区第{k + 1}球 | {x2:.1f} | {stats.chi2.sf(x2, 11):.3f} | — | — |")

# ---------------- B. 混合不充分：相邻号码是否接连出球 ----------------
p("\n## B. 混合不充分假设：相邻号码是否更容易接连出球、同期出现？\n")
p("摇奖球按号码顺序码放。若混合不充分，号码相近的球会更容易接连出、同期出。\n")
p("| 时段 | 指标 | 实际 | 理论 | z | p |")
p("|---|---|---|---|---|---|")
# 接连两球 |差|=1 的理论概率：有序不放回取两个，2*34/(35*34)
P_ADJ = 2 / 35
# |差|<=3
P_NEAR = sum(1 for a in range(1, 36) for b in range(1, 36) if a != b and abs(a - b) <= 3) / (35 * 34)
# 同期至少一对连号（无序组合）
P_CONS = 1 - comb(31, 5) / comb(35, 5)  # 无连号的组合数 = C(35-4, 5)
for s in SEGS:
    m = seg(OY, s)
    d = np.abs(np.diff(O[m], axis=1))
    n = d.size
    for name, obs, th in (("接连两球号码相邻(|差|=1)", (d == 1).mean(), P_ADJ),
                          ("接连两球号码相近(|差|≤3)", (d <= 3).mean(), P_NEAR)):
        z = (obs - th) / np.sqrt(th * (1 - th) / n)
        p(f"| {s} | {name} | {obs:.4f} | {th:.4f} | {z:+.2f} | {2 * stats.norm.sf(abs(z)):.3f} |")
for s in SEGS:
    m = seg(HY, s)
    obs = (np.diff(H[m], axis=1) == 1).any(1)
    z = (obs.mean() - P_CONS) / np.sqrt(P_CONS * (1 - P_CONS) / m.sum())
    p(f"| {s} | 同期有连号 | {obs.mean():.4f} | {P_CONS:.4f} | {z:+.2f} | {2 * stats.norm.sf(abs(z)):.3f} |")
m = HY >= 2007
obs = (np.diff(H, axis=1) == 1).any(1)
z = (obs.mean() - P_CONS) / np.sqrt(P_CONS * (1 - P_CONS) / len(H))
p(f"| 全部(2007-) | 同期有连号 | {obs.mean():.4f} | {P_CONS:.4f} | {z:+.2f} | {2 * stats.norm.sf(abs(z)):.3f} |")
# 同期连号对数均值
th_pairs = 4 * comb(33, 3) / comb(35, 5) * 34 / 4 / 34 * 34  # 每对(i,i+1)同时出现概率×34
th_pairs = 34 * comb(33, 3) / comb(35, 5)
for s in SEGS:
    m = seg(HY, s)
    k = (np.diff(H[m], axis=1) == 1).sum(1)
    var = k.var(ddof=1)
    z = (k.mean() - th_pairs) / np.sqrt(var / len(k))
    p(f"| {s} | 每期连号对数 | {k.mean():.4f} | {th_pairs:.4f} | {z:+.2f} | {2 * stats.norm.sf(abs(z)):.3f} |")

# ---------------- C. 预测模型样本外打分 ----------------
p("\n## C. 预测模型的样本外打分（逐期滚动，只用过去数据）\n")
p("每个模型给出 35 个号码各自的出现概率（总和=5），用对数损失给下一期打分，与均匀模型（每个号 1/7）比较。")
p("**比特增益 > 0 才说明模型真的比随机更会“猜”**；用 2017 年后 1457 期评估。\n")

XH = np.zeros((len(H), 35))
for i, r in enumerate(H):
    XH[i, r - 1] = 1
U = np.full(35, 5 / 35)


def ll(prob, y):
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return (y * np.log2(prob) + (1 - y) * np.log2(1 - prob)).sum()


def m_freq(W, alpha):
    def f(i):
        c = XH[max(0, i - W):i].sum(0)
        n = min(i, W)
        q = (c + alpha * 5 / 35) / (n + alpha)
        return q / q.sum() * 5
    return f


def m_zone(W, shrink):
    def f(i):
        s = XH[max(0, i - W):i, :12].sum() / (5 * min(i, W))
        r = 1 + (s / (12 / 35) - 1) * shrink
        q = np.where(np.arange(35) < 12, r, (1 - 12 / 35 * r) / (23 / 35))
        return q * 5 / 35
    return f


def m_cold(W):
    def f(i):
        c = XH[max(0, i - W):i].sum(0)
        q = (c.max() - c + 1)
        q = q / q.sum() * 5
        return 0.9 * U + 0.1 * q
    return f


def m_hot(W):
    def f(i):
        c = XH[max(0, i - W):i].sum(0) + 1
        q = c / c.sum() * 5
        return 0.9 * U + 0.1 * q
    return f


MODELS = {
    "小号偏移(500期,收缩0.5)": m_zone(500, 0.5),
    "小号偏移(300期,收缩1)": m_zone(300, 1.0),
    "号码频率(100期,α=100)": m_freq(100, 100),
    "号码频率(300期,α=300)": m_freq(300, 300),
    "号码频率(1000期,α=1000)": m_freq(1000, 1000),
    "号码频率(全部历史,α=500)": m_freq(10 ** 6, 500),
    "热号倾斜(100期)": m_hot(100),
    "冷号倾斜(100期)": m_cold(100),
}
START = int(np.argmax(HY >= 2017))
p("| 模型 | 总比特增益 | 每期 | 标准误 | z | 相当于头奖概率倍数 |")
p("|---|---|---|---|---|---|")
for name, f in MODELS.items():
    g = np.array([ll(f(i), XH[i]) - ll(U, XH[i]) for i in range(START, len(H))])
    se = g.std(ddof=1) * np.sqrt(len(g))
    p(f"| {name} | {g.sum():+.2f} | {g.mean():+.5f} | {se:.2f} | {g.sum() / se:+.2f} | ×{2 ** (g.mean() * 1):.4f} |")
p("\n“相当于头奖概率倍数”是粗略换算：每期平均增益 b 比特 ≈ 对真实开奖组合的概率估计提高 2^b 倍。")

text = "\n".join(out) + "\n"
(ROOT / "report_novel.md").write_text(text, encoding="utf-8")
print(text)
