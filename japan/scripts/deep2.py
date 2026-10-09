"""日本ロト第二轮深挖（之前没做过的角度）。输出 japan/report_deep2.md。

A. 追热球的历史回测：把雷达（下注档/证明档 CUSUM）放回全部历史逐回运行，每次报警后看该号码接下来 L 回的出现率。
   这直接回答"追热球到底有没有用"。买含热球的一注，头奖概率倍数 = 报警后出现率 / 理论出现率。
   同时跑大乐透（分 2014 前后）和双色球做对照：大乐透 2007–2013 绿球时代是已知的真实物理偏差。
B. 跨彩种一致性：三种ロト很可能用同一家厂商、同样印刷工艺的球。如果某个号码的球天生偏重/偏轻，
   它在三种彩票里应该同时偏冷或偏热。比较 1–31 号在三种彩票里的偏差相关性（置换检验）。
C. 印刷墨量：号码印得越"黑"（8、0 比 1 用墨多），球的质量分布越不均匀。用字形像素面积对偏差回归，
   五种彩票合并检验。
D. 出球早晚的物理一致性：同一个号码，"在本数字中出得晚"和"当上ボーナス数字（最后出的球）"是对同一种物理
   属性（轻/难被抓）的两次独立测量。两者若正相关，说明存在稳定的出球早晚偏差；再用奇偶回拆半检验稳定性。
E. ロト6 星期一 vs 星期四（不同的开奖批次/人员/机器预热）。
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
import radar  # noqa: E402

rng = np.random.default_rng(77)
TH = json.loads((REPO / "data" / "radar_thresholds.json").read_text())
JP = {"loto6": ("ロト6", 43, 6), "loto7": ("ロト7", 37, 7), "miniloto": ("ミニロト", 31, 5)}
out = ["# 日本ロト第二轮深挖\n"]
p = out.append


def jp_rows(game):
    M = JP[game][2]
    D = list(csv.DictReader((ROOT / "data" / f"{game}.csv").open(encoding="utf-8")))
    return D, np.array([[int(d[f"n{i}"]) for i in range(1, M + 1)] for d in D])


def onehot(R, K):
    X = np.zeros((len(R), K))
    X[np.arange(len(R))[:, None], np.asarray(R) - 1] = 1
    return X


def zvec(X, K, M):
    n = len(X)
    p0 = M / K
    return (X.sum(0) - n * p0) / np.sqrt(n * p0 * (1 - p0) * (K - M) / (K - 1))


# ---------------- A ----------------
def backtest(X, K, M, H, L):
    """逐回运行 CUSUM；号码 S 越过 H 时记一次报警，统计之后 L 回的出现次数；报警后该号码冷却 L 回、S 归零。"""
    up, down = radar.increments(K, M)
    S = np.zeros(K)
    cool = np.zeros(K, int)
    hits = n = ev = 0
    for t in range(len(X) - 1):
        S = np.maximum(0.0, S + np.where(X[t] > 0, up, down))
        cool -= 1
        for i in np.flatnonzero((S >= H) & (cool <= 0)):
            fut = X[t + 1:t + 1 + L, i]
            hits += fut.sum()
            n += len(fut)
            ev += 1
            cool[i] = L
            S[i] = 0
    return ev, n, hits


p("## A. 追热球的历史回测（把雷达放回全部历史逐回运行）\n")
p("每次报警后，看这个号码接下来 L 回的出现率。“倍数”= 出现率 / 理论，也就是买一注含该号码的票，头奖概率被放大的倍数。"
  "报警后该号码冷却 L 回，避免同一段热度被重复计数。\n")
games = []
for g, (title, K, M) in JP.items():
    games.append((title, K, M, onehot(jp_rows(g)[1], K)))
dlt = radar.load("dlt")
Xd = onehot([r for _, r in dlt], 35)
cut = next(i for i, (iss, _) in enumerate(dlt) if iss >= "14001")
games.append(("大乐透 2007–2013（已知真实偏差，阳性对照）", 35, 5, Xd[:cut]))
games.append(("大乐透 2014–今", 35, 5, Xd[cut:]))
games.append(("双色球", 33, 6, onehot([r for _, r in radar.load("ssq")], 33)))
p("| 彩种 | 档位 | L | 报警次数 | 报警后出现率 | 理论 | 倍数 | z |")
p("|---|---|---|---|---|---|---|---|")
summary = {}
for title, K, M, X in games:
    for tier in ("bet", "proof"):
        H = TH[f"{K}_{M}"][tier]["H"]
        for L in (10, 30):
            ev, n, h = backtest(X, K, M, H, L)
            p0 = M / K
            rate = h / n if n else float("nan")
            z = (h - n * p0) / np.sqrt(n * p0 * (1 - p0)) if n else float("nan")
            p(f"| {title} | {'下注档' if tier == 'bet' else '证明档'} | {L} | {ev} | {rate:.3f} | {p0:.3f} | ×{rate / p0:.3f} | {z:+.2f} |")
            summary[(title, tier, L)] = (rate / p0, z)
p("\n日本三种合并（Stouffer）：" + "；".join(
    f"{'下注档' if t == 'bet' else '证明档'} L={L} z={sum(summary[(g[0], t, L)][1] for g in JP.values()) / np.sqrt(3):+.2f}"
    for t in ("bet", "proof") for L in (10, 30)))

# ---------------- B ----------------
p("\n## B. 跨彩种一致性：同一个号码在三种ロト里是否同冷同热\n")
Z = {}
for g, (title, K, M) in JP.items():
    Z[title] = zvec(onehot(jp_rows(g)[1], K), K, M)
p("| 对比 | 共同号码 | 相关 r | 置换 p |")
p("|---|---|---|---|")
pairs = [("ロト6", "ロト7"), ("ロト6", "ミニロト"), ("ロト7", "ミニロト")]
rs = []
for a, b in pairs:
    m = min(len(Z[a]), len(Z[b]))
    r = np.corrcoef(Z[a][:m], Z[b][:m])[0, 1]
    null = np.array([np.corrcoef(Z[a][:m], rng.permutation(Z[b][:m]))[0, 1] for _ in range(20000)])
    pv = (np.sum(np.abs(null) >= abs(r)) + 1) / (len(null) + 1)
    rs.append(r)
    p(f"| {a} vs {b} | 1–{m} | {r:+.3f} | {pv:.3f} |")
zsum = Z["ロト6"][:31] + Z["ロト7"][:31] + Z["ミニロト"][:31]
top = np.argsort(-zsum)[:5] + 1
bot = np.argsort(zsum)[:5] + 1
p(f"\n三种合计最热：{', '.join(f'{x:02d}' for x in top)}；最冷：{', '.join(f'{x:02d}' for x in bot)}")
zall = zsum / np.sqrt(3)
p(f"合计 z 的 χ²（df31）= {np.sum(zall ** 2):.1f}，p={stats.chi2.sf(np.sum(zall ** 2), 31):.3f}（若存在跨彩种共同的号码偏差，此值应偏大）")

# ---------------- C ----------------
p("\n## C. 印刷墨量与出现率\n")
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 120)


def ink(s):
    im = Image.new("L", (400, 200), 0)
    ImageDraw.Draw(im).text((10, 10), s, fill=255, font=font)
    return np.asarray(im).astype(float).sum() / 255


p("用粗体字形的像素面积近似每个号码的用墨量，分“1”和“01”两种印法。对每种彩票的号码偏差 z 做相关，再用 Stouffer 合并。\n")
p("| 彩种 | r（印成 1） | r（印成 01） |")
p("|---|---|---|")
allz = {"ロト6": Z["ロト6"], "ロト7": Z["ロト7"], "ミニロト": Z["ミニロト"],
        "大乐透 2014–今": zvec(Xd[cut:], 35, 5),
        "双色球": zvec(onehot([r for _, r in radar.load("ssq")], 33), 33, 6)}
comb_ = {1: [], 2: []}
for title, z in allz.items():
    K = len(z)
    rr = []
    for w, fmt in ((1, "{}"), (2, "{:02d}")):
        a = np.array([ink(fmt.format(i)) for i in range(1, K + 1)])
        r = np.corrcoef(a, z)[0, 1]
        comb_[w].append(np.arctanh(r) * np.sqrt(K - 3))
        rr.append(r)
    p(f"| {title} | {rr[0]:+.3f} | {rr[1]:+.3f} |")
for w, lab in ((1, "1"), (2, "01")):
    zz = np.sum(comb_[w]) / np.sqrt(len(comb_[w]))
    p(f"\n合并（印成 {lab}）：z={zz:+.2f}，p={2 * stats.norm.sf(abs(zz)):.3f}")

# ---------------- D ----------------
p("\n## D. 出球早晚：同一号码的两次独立测量是否一致\n")
p("“晚出度”= 号码作为本数字开出时，在出球顺序里的平均位置（标准化）；“ボーナス率”= 作为最后出的ボーナス数字的频率（标准化）。"
  "若某些球物理上就是难被抓到，两者应正相关；“奇偶拆半”= 奇数回与偶数回各算一次晚出度的相关，检验它是否稳定。\n")
p("| 彩种 | 晚出度 vs ボーナス率 r | p | 晚出度奇偶拆半 r | p |")
p("|---|---|---|---|---|")
late_all = {}
for g, nb in (("loto6", ["nbo"]), ("loto7", ["nbo1", "nbo2"])):
    title, K, M = JP[g]
    D, _ = jp_rows(g)
    O = np.array([[int(d[f"o{i}"]) for i in range(1, M + 1)] for d in D])
    B = np.array([[int(d[k]) for k in nb] for d in D])
    pos = np.arange(M) - (M - 1) / 2
    sd_pos = pos.std()

    def lateness(rows):
        s, c = np.zeros(K), np.zeros(K)
        for o in O[rows]:
            s[o - 1] += pos
            c[o - 1] += 1
        return s / np.maximum(c, 1) / sd_pos * np.sqrt(c)

    late = lateness(np.arange(len(O)))
    late_all[title] = late
    bz = zvec(onehot(B, K), K, len(nb))
    r1 = np.corrcoef(late, bz)[0, 1]
    e, o_ = lateness(np.arange(0, len(O), 2)), lateness(np.arange(1, len(O), 2))
    r2 = np.corrcoef(e, o_)[0, 1]
    pv = lambda r: 2 * stats.norm.sf(abs(np.arctanh(r)) * np.sqrt(K - 3))
    p(f"| {title} | {r1:+.3f} | {pv(r1):.3f} | {r2:+.3f} | {pv(r2):.3f} |")
r = np.corrcoef(late_all["ロト6"][:37], late_all["ロト7"])[0, 1]
p(f"\n跨彩种：1–37 号在ロト6 与ロト7 的晚出度相关 r={r:+.3f}，p={2 * stats.norm.sf(abs(np.arctanh(r)) * np.sqrt(34)):.3f}")

# ---------------- E ----------------
p("\n## E. ロト6 星期一 vs 星期四\n")
D6, R6 = jp_rows("loto6")
from datetime import date as _d

wd = np.array([_d.fromisoformat(d["date"]).weekday() for d in D6])
X6 = onehot(R6, 43)
tab = np.vstack([X6[wd == 0].sum(0), X6[wd == 3].sum(0)])
chi, pv, *_ = stats.chi2_contingency(tab)
p(f"星期一 {int((wd == 0).sum())} 回，星期四 {int((wd == 3).sum())} 回；2×43 列联表 χ²={chi:.1f}，p={pv:.3f}")

text = "\n".join(out) + "\n"
(ROOT / "report_deep2.md").write_text(text, encoding="utf-8")
print(text)
