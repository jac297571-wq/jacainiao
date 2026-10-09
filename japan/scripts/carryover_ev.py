"""キャリーオーバー（奖池滚存）与每注期望回报。输出 japan/report_carryover.md。

已实现回报率 R_t = Σ_k (第 k 等中奖注数 × 单注奖金) / 当回销售额。
按“上回滚入的キャリーオーバー / 当回销售额”分箱，箱内平均 R 即为该条件下的期望回报
（中不中头奖的运气在多回平均中抵消）。置信区间用自助法（按回重抽样）。
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(0)
out = ["# キャリーオーバー与期望回报\n",
       "R = 当回全部奖金支出 / 当回销售额（每投入 1 日元平均拿回多少）。R>1 才是正期望。\n"]

for game, tiers, price in (("loto6", 5, 200), ("loto7", 6, 300), ("miniloto", 4, 200)):
    D = list(csv.DictReader((ROOT / "data" / f"{game}.csv").open(encoding="utf-8")))
    rows = []
    prev_co = 0.0
    for d in D:
        sales = float(d.get("sales") or 0)
        co_in = prev_co
        prev_co = float(d.get("co") or 0)
        if sales <= 0:
            continue
        paid = sum(float(d[f"t{k}"] or 0) * float(d[f"m{k}"] or 0) for k in range(1, tiers + 1))
        jack = float(d["t1"] or 0) * float(d["m1"] or 0)
        rows.append((d["id"], d["date"], sales, co_in, paid / sales, jack / sales, int(d["t1"] or 0)))
    if not rows:
        out.append(f"## {game}\n\n无销售额数据。\n")
        continue
    A = np.array([(r[2], r[3], r[4], r[5], r[6]) for r in rows])
    sales, co, R, J, t1 = A.T
    ratio = co / sales
    out.append(f"## {game}（{len(rows)} 回有销售额，{rows[0][1]} ~ {rows[-1][1]}；每注 {price} 日元）\n")
    out.append(f"全体平均回报率 R = {R.mean():.3f}；其中头奖部分 {J.mean():.3f}。\n")
    out.append("| 滚入キャリーオーバー / 当回销售额 | 回数 | 平均回报率 R | 95% CI | 头奖出现率 |")
    out.append("|---|---|---|---|---|")
    bins = [(-1e-9, 1e-9, "0（无滚存）"), (1e-9, 0.25, "0–0.25"), (0.25, 0.5, "0.25–0.5"), (0.5, 1.0, "0.5–1"),
            (1.0, 2.0, "1–2"), (2.0, 99, "≥2")]
    for lo, hi, lab in bins:
        m = (ratio > lo) & (ratio <= hi) if lo >= 0 else (ratio <= hi)
        if m.sum() < 5:
            continue
        r = R[m]
        bs = [rng.choice(r, len(r)).mean() for _ in range(3000)]
        lo_ci, hi_ci = np.percentile(bs, [2.5, 97.5])
        out.append(f"| {lab} | {m.sum()} | **{r.mean():.3f}** | [{lo_ci:.3f}, {hi_ci:.3f}] | {(t1[m] > 0).mean():.2f} |")
    top = np.argsort(-ratio)[:5]
    out.append("\n滚存比例最高的几回：")
    for i in top:
        out.append(f"- 第{rows[i][0]}回（{rows[i][1]}）：滚入 {co[i] / 1e8:.2f} 亿日元，销售 {sales[i] / 1e8:.2f} 亿，"
                   f"比例 {ratio[i]:.2f}，实际回报率 {R[i]:.3f}，头奖 {int(t1[i])} 注")
    out.append("")

text = "\n".join(out) + "\n"
(ROOT / "report_carryover.md").write_text(text, encoding="utf-8")
print(text)
