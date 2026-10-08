"""只用 2018 年起的数据：发现期（2018-01 ~ 2023-12）大规模扫描假设 → 验证期（2024-01 起）只检验前 10 名。
输出 report_recent.md。

每个假设是一个号码集合 S（可附带条件：某套球 / 出球顺序第 k 位 / 后区），统计量为
    每期落在 S 中的号码个数之和，零假设下精确的超几何均值与方差 → z 值。
形态类假设（连号、等差…）为二项事件，理论概率由全部组合精确计数。
"""
import csv
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
rows = [r for r in csv.DictReader((ROOT / "data" / "dlt_ballset.csv").open(encoding="utf-8")) if r["date"] >= "2018-01-01"]
DATE = np.array([r["date"] for r in rows])
ISS = [r["issue"] for r in rows]
SET = np.array([int(r["ballset"]) for r in rows])
SORTED = np.array([list(map(int, r["sorted"].split())) for r in rows])
ORDER = np.array([list(map(int, r["order"].split())) for r in rows])
R, B = SORTED[:, :5], SORTED[:, 5:]
O = ORDER[:, :5]
DISC = DATE < "2024-01-01"
HOLD = ~DISC
ALL = np.array(list(combinations(range(1, 36), 5)))

H = []  # (家族, 名称, 计数函数(idx)->(观测, 期望, 方差, n))


def set_count(S, pool=35, k=5, which="front", cond=None):
    S = np.array(sorted(S))
    s = len(S)

    def f(idx):
        if cond is not None:
            idx = idx[cond[idx]]
        n = len(idx)
        if which == "front":
            c = np.isin(R[idx], S).sum()
        elif which == "back":
            c = np.isin(B[idx], S).sum()
        else:  # 出球顺序第 which 位，单球
            c = np.isin(O[idx, which], S).sum()
            mu = n * s / 35
            return c, mu, n * s / 35 * (1 - s / 35), n
        kk = k
        mu = n * kk * s / pool
        var = n * kk * (s / pool) * (1 - s / pool) * (pool - kk) / (pool - 1)
        return c, mu, var, n
    return f


def event(fn, p0):
    def f(idx):
        n = len(idx)
        c = sum(fn(R[i]) for i in idx)
        return c, n * p0, n * p0 * (1 - p0), n
    return f


fmt = lambda S: f"{min(S):02d}-{max(S):02d}" if list(S) == list(range(min(S), max(S) + 1)) else ",".join(f"{x:02d}" for x in S)
# 1. 前区号段
for a in range(1, 36):
    for b in range(a + 2, min(a + 20, 36)):
        H.append(("前区号段", f"前区 {a:02d}-{b:02d}", set_count(range(a, b + 1))))
# 2. 单号
for x in range(1, 36):
    H.append(("前区单号", f"前区 {x:02d}", set_count([x])))
# 3. 余数类、尾数
for m in range(2, 8):
    for r in range(m):
        H.append(("余数类", f"前区 mod{m}={r}", set_count([x for x in range(1, 36) if x % m == r])))
for t in range(10):
    H.append(("尾数", f"前区 尾数{t}", set_count([x for x in range(1, 36) if x % 10 == t])))
# 4. 每套球的号段（长度 5-15）
for s in (1, 2, 3):
    for a in range(1, 36):
        for b in range(a + 4, min(a + 15, 36)):
            H.append(("分套号段", f"第{s}套 前区 {a:02d}-{b:02d}", set_count(range(a, b + 1), cond=(SET == s))))
# 5. 出球顺序第 k 位的号段（三等分 + 01-11/12-35）
for k in range(5):
    for a, b in ((1, 11), (12, 23), (24, 35), (1, 12), (13, 24), (25, 35)):
        H.append(("出球位置", f"第{k + 1}球 {a:02d}-{b:02d}", set_count(range(a, b + 1), which=k)))
# 6. 后区
for x in range(1, 13):
    H.append(("后区单号", f"后区 {x:02d}", set_count([x], 12, 2, "back")))
for a in range(1, 13):
    for b in range(a + 1, min(a + 6, 13)):
        H.append(("后区号段", f"后区 {a:02d}-{b:02d}", set_count(range(a, b + 1), 12, 2, "back")))
for m in (2, 3, 4):
    for r in range(m):
        H.append(("后区余数", f"后区 mod{m}={r}", set_count([x for x in range(1, 13) if x % m == r], 12, 2, "back")))


# 7. 形态事件
def max_run(r):
    best = cur = 1
    for a, b in zip(r, r[1:]):
        cur = cur + 1 if b - a == 1 else 1
        best = max(best, cur)
    return best


def has_ap3(r):
    s = set(r)
    return any(2 * b - a in s for a, b in combinations(r, 2))


EVENTS = {
    "有连号": lambda r: max_run(r) >= 2,
    "三连号": lambda r: max_run(r) >= 3,
    "≥2组连号": lambda r: (np.diff(r) == 1).sum() >= 2,
    "含3项等差": has_ap3,
    "全奇或全偶": lambda r: len(set(x % 2 for x in r)) == 1,
    "跨度≤20": lambda r: r[-1] - r[0] <= 20,
    "跨度≥30": lambda r: r[-1] - r[0] >= 30,
    "和值≤70": lambda r: sum(r) <= 70,
    "和值≥110": lambda r: sum(r) >= 110,
}
for name, fn in EVENTS.items():
    p0 = np.mean([fn(c) for c in ALL])
    H.append(("形态", name, event(fn, p0)))

disc_idx, hold_idx = np.flatnonzero(DISC), np.flatnonzero(HOLD)


def zstat(f, idx):
    c, mu, var, n = f(idx)
    return (c - mu) / np.sqrt(var), c / mu if mu else np.nan, n


res = []
for fam, name, f in H:
    z, ratio, n = zstat(f, disc_idx)
    res.append((fam, name, f, z, ratio))
zs = np.array([r[3] for r in res])
pv = 2 * stats.norm.sf(np.abs(zs))
m = len(pv)
order = np.argsort(pv)
q = np.empty(m)
q[order] = np.minimum.accumulate((pv[order] * m / np.arange(1, m + 1))[::-1])[::-1]

out = []
p = out.append
p("# 2018 年起深度扫描：发现期 → 验证期\n")
p(f"发现期 {ISS[disc_idx[0]]}–{ISS[disc_idx[-1]]}（{len(disc_idx)} 期），验证期 {ISS[hold_idx[0]]}–{ISS[hold_idx[-1]]}（{len(hold_idx)} 期）。")
p(f"共扫描 **{m} 个假设**。纯随机下，发现期也会有约 {m * 0.05:.0f} 个 p<0.05、约 {m * 0.001:.1f} 个 p<0.001。\n")
p("## 发现期：各家族最强的假设\n")
p("| 家族 | 假设数 | 最强假设 | z | 出现率/理论 | q(FDR) |")
p("|---|---|---|---|---|---|")
fams = []
for r in res:
    if r[0] not in fams:
        fams.append(r[0])
for fam in fams:
    idx = [i for i, r in enumerate(res) if r[0] == fam]
    j = max(idx, key=lambda i: abs(zs[i]))
    p(f"| {fam} | {len(idx)} | {res[j][1]} | {zs[j]:+.2f} | ×{res[j][4]:.3f} | {q[j]:.3f} |")

# 选前 10（去掉高度重叠：同家族同方向、号段重叠 >70% 的只保留最强）
def members(name):
    import re
    mm = re.findall(r"(\d{2})-(\d{2})", name)
    if mm:
        a, b = map(int, mm[-1])
        return set(range(a, b + 1))
    return {name}


picked = []
for i in np.argsort(-np.abs(zs)):
    fam, name = res[i][0], res[i][1]
    ctx = name.split(" ")[0] if fam in ("分套号段", "出球位置") else fam.replace("号段", "").replace("单号", "")
    dup = False
    for j in picked:
        fj, nj = res[j][0], res[j][1]
        cj = nj.split(" ")[0] if fj in ("分套号段", "出球位置") else fj.replace("号段", "").replace("单号", "")
        a, b = members(name), members(nj)
        if ctx == cj and np.sign(zs[i]) == np.sign(zs[j]) and len(a & b) / max(1, min(len(a), len(b))) > 0.7:
            dup = True
            break
    if not dup:
        picked.append(i)
    if len(picked) == 10:
        break

p("\n## 验证期：只检验发现期的前 10 名（去重后），方向须一致\n")
p("Bonferroni：单侧 p < 0.05/10 = 0.005 才算通过验证。\n")
p("| 名次 | 假设 | 发现期 z | 发现期 ×理论 | 验证期 z | 验证期 ×理论 | 验证期单侧 p | 结论 |")
p("|---|---|---|---|---|---|---|---|")
for rank, i in enumerate(picked, 1):
    fam, name, f, zd, rd = res[i]
    zh, rh, nh = zstat(f, hold_idx)
    pone = stats.norm.sf(zh * np.sign(zd))
    verdict = "✅ 通过" if pone < 0.005 else ("◐ 同向但不显著" if pone < 0.5 else "✗ 未复现")
    p(f"| {rank} | {name} | {zd:+.2f} | ×{rd:.3f} | {zh:+.2f} | ×{rh:.3f} | {pone:.4f} | {verdict} |")

# 01-11 的位置
k111 = next(i for i, r in enumerate(res) if r[1] == "前区 01-11")
rank111 = int((np.abs(zs) > abs(zs[k111])).sum()) + 1
zh, rh, _ = zstat(res[k111][2], hold_idx)
p(f"\n“前区 01-11”在发现期的排名：第 {rank111} / {m}（z={zs[k111]:+.2f}，q={q[k111]:.3f}）；验证期 z={zh:+.2f}，×{rh:.3f}。")

# ---------- 逐年趋势 ----------
p("\n## 01-11 信号逐年趋势\n")
p("| 年份 | 期数 | 每期 01-11 个数（理论 1.571） | ×理论 | z | 主力摇奖机 |")
p("|---|---|---|---|---|---|")
for y in range(2018, 2027):
    mk = np.array([d.startswith(str(y)) for d in DATE])
    if mk.sum() == 0:
        continue
    s = (R[mk] <= 11).sum(1)
    n = mk.sum()
    var = 5 * (11 / 35) * (24 / 35) * (30 / 34)
    z = (s.mean() - 55 / 35) / np.sqrt(var / n)
    mach = "旧" if y < 2019 else ("旧→新（9月）" if y == 2019 else "新")
    p(f"| {y} | {n} | {s.mean():.3f} | ×{s.mean() / (55 / 35):.3f} | {z:+.2f} | {mach} |")

text = "\n".join(out) + "\n"
(ROOT / "report_recent.md").write_text(text, encoding="utf-8")
print(text)
