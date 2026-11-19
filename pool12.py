"""大乐透 12+6 收敛模型：回测“怎么选 12 个前区、6 个后区”，以及选好之后怎么用最少的注数把池子罩住。

一、选池子（严格前瞻：每期只用此前的数据）。比较几种选法，看开奖号码平均有几个落在池里：
   随机期望：前区 5×12/35 = 1.714 个，后区 2×6/12 = 1.000 个；前区 5 个全进池的概率 0.244%，后区 2 个全进 22.7%。
二、罩池子（旋转矩阵）：池子选好后，用贪心法找最少的注数，保证“池里开出 k 个时，至少有一注中 t 个”。
用法：python3 pool12.py   → report_pool12.md
"""
import csv
import itertools
from math import comb, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NF, NB = 12, 6
START = "14001"         # 当前这台机器（2014 年起）
WARM = 300              # 前 300 期只用于积累历史，不参与评分


def load():
    rows = [r for r in csv.DictReader((ROOT / "data" / "dlt_history.csv").open(encoding="utf-8")) if r["issue"] >= START]
    F = np.zeros((len(rows), 35), bool)
    B = np.zeros((len(rows), 12), bool)
    for i, r in enumerate(rows):
        F[i, [int(r[f"r{k}"]) - 1 for k in range(1, 6)]] = True
        B[i, [int(r["b1"]) - 1, int(r["b2"]) - 1]] = True
    return rows, F, B


def methods(H, rng):
    """H：此前各期的 0/1 矩阵（期 × 号码）。返回 {方法名: 每个号码的分数}，分数高的进池"""
    n = len(H)
    last = np.array([n - 1 - np.max(np.nonzero(H[:, j])[0]) if H[:, j].any() else n for j in range(H.shape[1])])
    out = {"随机": rng.random(H.shape[1])}
    for w in (10, 30, 100, 300):
        out[f"近 {w} 期最热"] = H[-w:].sum(0) + rng.random(H.shape[1]) * 1e-3
    out["近 100 期最冷"] = -H[-100:].sum(0) + rng.random(H.shape[1]) * 1e-3
    out["2014 年起累计最热"] = H.sum(0) + rng.random(H.shape[1]) * 1e-3
    out["遗漏最久"] = last + rng.random(H.shape[1]) * 1e-3
    out["刚开过（遗漏最短）"] = -last + rng.random(H.shape[1]) * 1e-3
    return out


def backtest(X, size, M, rng):
    res = {}
    for t in range(WARM, len(X)):
        for name, sc in methods(X[:t], rng).items():
            pool = np.argsort(-sc)[:size]
            res.setdefault(name, []).append(X[t, pool].sum())
    return {k: np.array(v) for k, v in res.items()}


def greedy_cover(n, m, k, t):
    """n 个号里选 m 个一注；保证：开出的 k 个号都在池里时，至少一注中 t 个。返回注列表（贪心，不一定最少）"""
    tickets = [frozenset(c) for c in itertools.combinations(range(n), m)]
    targets = [frozenset(c) for c in itertools.combinations(range(n), k)]
    left = set(targets)
    chosen = []
    while left:
        best = max(tickets, key=lambda T: sum(len(T & s) >= t for s in left))
        chosen.append(best)
        left = {s for s in left if len(best & s) < t}
    return chosen


def main():
    rng = np.random.default_rng(0)
    rows, F, B = load()
    nt = len(rows) - WARM
    out = ["# 大乐透 12+6 收敛模型回测\n",
           f"数据：2014 年起 {len(rows)} 期，前 {WARM} 期只做历史，后 {nt} 期（{rows[WARM]['issue']}–{rows[-1]['issue']}）逐期前瞻评分。\n",
           "## 一、怎么选池子\n",
           "每期按各方法选出前区 12 个、后区 6 个，看开奖号码平均落进几个。z 是相对随机期望的标准分（|z|>2.9 才算过了 8 种方法的多重比较）。\n",
           "| 选法 | 前区平均进池 | z | 前区 ≥4 个进池 | 前区 5 个全进 | 后区平均进池 | z | 后区 2 个全进 |", "|---|---|---|---|---|---|---|---|"]
    ef, eb = 5 * NF / 35, 2 * NB / 12
    vf = 5 * (NF / 35) * (1 - NF / 35) * (35 - 5) / 34
    vb = 2 * (NB / 12) * (1 - NB / 12) * (12 - 2) / 11
    rf, rb = backtest(F, NF, 5, rng), backtest(B, NB, 2, rng)
    p4 = (comb(NF, 4) * comb(35 - NF, 1) + comb(NF, 5)) / comb(35, 5)
    out.append(f"| （理论随机） | {ef:.3f} | — | {p4 * 100:.2f}% | {comb(NF, 5) / comb(35, 5) * 100:.3f}% | {eb:.3f} | — | {comb(NB, 2) / comb(12, 2) * 100:.1f}% |")
    for name in rf:
        a, b = rf[name], rb[name]
        zf = (a.mean() - ef) / sqrt(vf / len(a))
        zb = (b.mean() - eb) / sqrt(vb / len(b))
        out.append(f"| {name} | {a.mean():.3f} | {zf:+.2f} | {(a >= 4).mean() * 100:.2f}% | {(a == 5).mean() * 100:.2f}%（{(a == 5).sum()} 次） | "
                   f"{b.mean():.3f} | {zb:+.2f} | {(b == 2).mean() * 100:.1f}% |")
        print(name, f"{a.mean():.3f} z={zf:+.2f}  back {b.mean():.3f} z={zb:+.2f}", flush=True)
    out += ["\n## 二、怎么罩池子（旋转矩阵）\n",
            "池子定好后，不必买全 C(12,5)=792 注。下表是贪心法找到的注数（每注 2 元；后区按“6 选 2”另行组合）：\n",
            "| 前区保证 | 前区注数 | 后区保证 | 后区组数 | 合计注数 | 金额 | 罩住后的含义 |", "|---|---|---|---|---|---|---|"]
    fronts = [("开 5 中 5（全包）", None, 792), ("开 5 中 4", (5, 4), None), ("开 4 中 4", (4, 4), None), ("开 5 中 3", (5, 3), None),
              ("开 4 中 3", (4, 3), None), ("开 3 中 3", (3, 3), None)]
    backs = [("开 2 中 2（全包）", 15), ("开 2 中 1", 3)]
    for fn, kt, cnt in fronts:
        if cnt is None:
            cnt = len(greedy_cover(NF, 5, *kt))
        for bn, bc in backs:
            out.append(f"| {fn} | {cnt} | {bn} | {bc} | {cnt * bc} | {cnt * bc * 2:,} 元 | 前区池里开出够数、后区池里开出够数时，保证有一注达到 |")
        print(fn, cnt, flush=True)
    out += ["\n要点：旋转矩阵只保证“号码进池时”的最低中奖，不改变号码进池的概率，也不改变每注的期望回报（每注仍约 50%）。",
            "能不能赚，取决于第一部分：池子是否真的比随机多罩住开奖号码。"]
    (ROOT / "report_pool12.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("→ report_pool12.md")


if __name__ == "__main__":
    main()
