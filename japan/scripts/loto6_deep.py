"""ロト6 深入检验：リハーサル（固定 A セット）、同夜状态、开奖物理（出球位置/相邻/会场）、2019 换球前后。
输出 japan/report_loto6_deep.md。
"""
import csv
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
K, M = 43, 6
rng = np.random.default_rng(6)
SIMS = 3000
D = {int(d["id"]): d for d in csv.DictReader((ROOT / "data" / "loto6.csv").open(encoding="utf-8"))}
REH = {int(r["id"]): [int(r[f"r{i}"]) for i in range(1, 7)] + [int(r["rb"])]
       for r in csv.DictReader((ROOT / "data" / "loto6_rehearsal.csv").open(encoding="utf-8"))}
ids = sorted(D)
main = {i: [int(D[i][f"n{k}"]) for k in range(1, 7)] for i in ids}
order = {i: [int(D[i][f"o{k}"]) for k in range(1, 7)] for i in ids}
bonus = {i: int(D[i]["nbo"]) for i in ids}
setb = {i: D[i]["setball"] for i in ids}
date = {i: D[i]["date"] for i in ids}
NEWBALL = min(i for i in ids if date[i] >= "2019-04-01")
out = []
p = out.append


def chi(c):
    e = c.mean()
    return float(((c - e) ** 2 / e).sum())


def mc_p(n, m, obs):
    s = [chi(np.bincount(np.argsort(rng.random((n, K)), 1)[:, :m].ravel(), minlength=K)) for _ in range(SIMS)]
    return (np.sum(np.array(s) >= obs) + 1) / (SIMS + 1)


def counts(rows, m):
    c = np.zeros(K)
    for r in rows:
        for x in r[:m]:
            c[x - 1] += 1
    return c


p("# ロト6 深入检验\n")
p(f"正式开奖 {len(ids)} 回；リハーサル {len(REH)} 回（固定使用 A セット球；第 {min(REH)}–{max(REH)} 回中有缺失）。新セット球自第 {NEWBALL} 回（{date[NEWBALL]}）起。\n")

# ---------------- A. リハーサル ----------------
p("## A. リハーサル数字（A セット专用的大样本）\n")
p("### A1. A セット本身是否均匀？（リハーサル 6+1 个球）\n")
p("| 时段 | リハーサル回数 | χ²（含ボーナス共 7 球） | p | 最热 | 最冷 |")
p("|---|---|---|---|---|---|")
for name, cond in (("全部", lambda i: True), ("旧球（2019-04 前）", lambda i: i < NEWBALL), ("新球（2019-04 起）", lambda i: i >= NEWBALL)):
    rows = [REH[i] for i in REH if cond(i)]
    c = counts(rows, 7)
    x2 = chi(c)
    p(f"| {name} | {len(rows)} | {x2:.1f} | {mc_p(len(rows), 7, x2):.3f} | {(np.argsort(-c)[:3] + 1).tolist()} | {(np.argsort(c)[:3] + 1).tolist()} |")

p("\n### A2. A セットの“指纹”：リハーサル频率 vs 正式开奖中使用 A セット的回\n")
p("若 A セット某些球物理上更易被抽中，两组独立数据中这些号码都应偏多（相关 r>0）。\n")
p("| 时段 | リハーサル回数 | 正式 A セット回数 | r | 置换 p |")
p("|---|---|---|---|---|")
for name, cond in (("全部", lambda i: True), ("旧球", lambda i: i < NEWBALL), ("新球", lambda i: i >= NEWBALL)):
    rr = [REH[i] for i in REH if cond(i)]
    aa = [main[i] + [bonus[i]] for i in ids if setb[i] == "A" and cond(i)]
    a, b = counts(rr, 7), counts(aa, 7)
    a, b = a / a.sum() - 1 / K, b / b.sum() - 1 / K
    r = np.corrcoef(a, b)[0, 1]
    pr = (np.sum([np.corrcoef(a, rng.permutation(b))[0, 1] >= r for _ in range(SIMS)]) + 1) / (SIMS + 1)
    p(f"| {name} | {len(rr)} | {len(aa)} | {r:+.3f} | {pr:.3f} |")
# 对照：リハーサル频率 vs 正式开奖中非 A セット的回（应无相关）
rr = [REH[i] for i in REH]
na = [main[i] + [bonus[i]] for i in ids if setb[i] != "A"]
a, b = counts(rr, 7), counts(na, 7)
r0 = np.corrcoef(a / a.sum(), b / b.sum())[0, 1]
p(f"\n对照：リハーサル频率 vs 正式开奖中**非 A** セット的回：r={r0:+.3f}（应约为 0）")

p("\n### A3. 同一晚：リハーサル与正式开奖的重合\n")
th = M * M / K
p(f"零假设下，正式本数字与リハーサル本数字的重合个数期望 {th:.3f}。\n")
p("| 条件 | 回数 | 平均重合 | z | p（双侧） |")
p("|---|---|---|---|---|")
var1 = M * (M / K) * ((K - M) / K) * ((K - M) / (K - 1))
for name, cond in (("全部", lambda i: True), ("正式也用 A セット（同一套球）", lambda i: setb[i] == "A"),
                   ("正式用其他套", lambda i: setb[i] != "A")):
    ov = [len(set(main[i]) & set(REH[i][:6])) for i in REH if i in main and cond(i)]
    n = len(ov)
    z = (np.mean(ov) - th) / np.sqrt(var1 / n)
    p(f"| {name} | {n} | {np.mean(ov):.3f} | {z:+.2f} | {2 * stats.norm.sf(abs(z)):.3f} |")
# 出球顺序同位置是否相同 / 相近
same_pos = [sum(1 for a, b in zip(order[i], REH[i][:6]) if a == b) for i in REH if i in order]
p(f"\n同一出球位置号码完全相同的次数：平均 {np.mean(same_pos):.4f}（随机 {6 / K:.4f}）")

# ---------------- B. 开奖物理 ----------------
p("\n## B. 开奖物理（离心式“夢ロト君”）\n")
p("### B1. 出球顺序第 k 个球是否均匀\n")
p("| 位置 | χ² | p | 1–14 占比（理论 0.326） |")
p("|---|---|---|---|")
for k in range(6):
    c = np.bincount([order[i][k] - 1 for i in ids], minlength=K).astype(float)
    x2 = chi(c)
    sims = [chi(np.bincount(rng.integers(0, K, len(ids)), minlength=K).astype(float)) for _ in range(SIMS)]
    pv = (np.sum(np.array(sims) >= x2) + 1) / (SIMS + 1)
    low = np.mean([order[i][k] <= 14 for i in ids])
    p(f"| 第{k + 1}球 | {x2:.1f} | {pv:.3f} | {low:.3f} |")

p("\n### B2. 接连出球的号码是否相邻（球在机器中按号码码放时的残留）\n")
d = np.array([abs(order[i][k + 1] - order[i][k]) for i in ids for k in range(5)])
th_adj = 2 * (K - 1) / (K * (K - 1))
th_near = sum(1 for a in range(1, K + 1) for b in range(1, K + 1) if a != b and abs(a - b) <= 3) / (K * (K - 1))
for name, obs, t in (("|差|=1", (d == 1).mean(), th_adj), ("|差|≤3", (d <= 3).mean(), th_near)):
    z = (obs - t) / np.sqrt(t * (1 - t) / len(d))
    p(f"- {name}：实际 {obs:.4f}，理论 {t:.4f}，z={z:+.2f}")
cons = np.array([(np.diff(sorted(main[i])) == 1).any() for i in ids])
th_c = 1 - comb(K - M + 1, M) / comb(K, M)
z = (cons.mean() - th_c) / np.sqrt(th_c * (1 - th_c) / len(ids))
p(f"- 同回有连号：实际 {cons.mean():.4f}，理论 {th_c:.4f}，z={z:+.2f}")

p("\n### B3. 会场：东京 vs 大阪\n")
for st, name in (("1", "东京"), ("2", "大阪")):
    rows = [main[i] for i in ids if D[i]["stage_id"] == st]
    c = counts(rows, 6)
    x2 = chi(c)
    p(f"- {name}：{len(rows)} 回，χ²={x2:.1f}，p={mc_p(len(rows), 6, x2):.3f}")

p("\n### B4. 2019-04 换球前后逐号码对比\n")
a = counts([main[i] + [bonus[i]] for i in ids if i < NEWBALL], 7)
b = counts([main[i] + [bonus[i]] for i in ids if i >= NEWBALL], 7)
T = np.vstack([a, b])
x2, pv, _, _ = stats.chi2_contingency(T)
p(f"2×43 列联表 χ²={x2:.1f}，p={pv:.3f}（若换球改变了某些号码的被抽概率，应显著）")

text = "\n".join(out) + "\n"
(ROOT / "report_loto6_deep.md").write_text(text, encoding="utf-8")
print(text)
