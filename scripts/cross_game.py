"""跨彩种对照：“小号偏多”是大乐透特有，还是同类设备/所有彩种都有？输出 report_cross.md。

数据（17500.cn）：双色球 ssq（福彩，33 选 6）、快乐8 kl8（福彩，80 选 20）、排列5 pl5（体彩，每位 0-9）。
对每个彩种：按 2019-09-07（体彩摇奖机统一更新日）前后分段，
  1) 小号段（号码最小的约 1/3）的出现率 / 理论，z 值；
  2) 扫描所有“从最小号开始的台阶”，取最强者，用蒙特卡洛校正扫描带来的多重比较；
  3) 号码频率整体 χ²（蒙特卡洛 p）。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "other"
rng = np.random.default_rng(5)
SIMS = 1000


def load(game):
    rows = []
    for line in (SRC / f"{game}.txt").open(encoding="utf-8", errors="ignore"):
        f = line.split()
        if len(f) < 3:
            continue
        rows.append(f)
    return rows


def parse(game):
    rows = load(game)
    if game == "ssq":
        return [(r[1], [int(x) for x in r[2:8]]) for r in rows], 33, 6, 1
    if game == "kl8":
        return [(r[1], [int(x) for x in r[2:22]]) for r in rows], 80, 20, 1
    if game == "pl5":  # 每一位独立 0-9，拆成 5 个“位”分别看，号码 +1 映射为 1-10
        return [(r[1], [int(x) + 1 for x in r[2:7]]) for r in rows], 10, 5, 0


def sim_counts(n, K, m, with_replacement):
    if with_replacement:
        draws = rng.integers(1, K + 1, (n, m))
    else:
        draws = np.argsort(rng.random((n, K)), axis=1)[:, :m] + 1
    return np.bincount(draws.ravel(), minlength=K + 1)[1:]


def analyse(game):
    data, K, m, norep = parse(game)
    wr = norep == 0
    out = []
    low = round(K / 3)
    for name, lo, hi in (("2019-09-07 前", "0000", "2019-09-07"), ("2019-09-07 起", "2019-09-07", "9999")):
        D = np.array([x for d, x in data if lo <= d < hi])
        if len(D) < 100:
            continue
        n = len(D)
        c = np.bincount(D.ravel(), minlength=K + 1)[1:]
        e = n * m / K
        # 小号段
        p_low = low / K
        var = n * m * p_low * (1 - p_low) * ((K - m) / (K - 1) if not wr else 1)
        z_low = (c[:low].sum() - n * m * p_low) / np.sqrt(var)
        # 台阶扫描
        def best_step(cc):
            best = (0, 0, 0)
            for k in range(2, K - 1):
                pk = k / K
                v = n * m * pk * (1 - pk) * ((K - m) / (K - 1) if not wr else 1)
                z = (cc[:k].sum() - n * m * pk) / np.sqrt(v)
                if abs(z) > abs(best[0]):
                    best = (z, k, cc[:k].sum() / (n * m * pk))
            return best
        zb, kb, rb = best_step(c)
        chi = ((c - e) ** 2 / e).sum()
        sims_step, sims_chi = [], []
        for _ in range(SIMS):
            cs = sim_counts(n, K, m, wr)
            sims_step.append(abs(best_step(cs)[0]))
            sims_chi.append(((cs - e) ** 2 / e).sum())
        p_step = (np.sum(np.array(sims_step) >= abs(zb)) + 1) / (SIMS + 1)
        p_chi = (np.sum(np.array(sims_chi) >= chi) + 1) / (SIMS + 1)
        lab = (lambda k: f"{k - 1}" if wr else f"{k:02d}")
        out.append(f"| {name} | {n} | {lab(1)}–{lab(low)}（×{c[:low].sum() / (n * m * p_low):.3f}，z={z_low:+.2f}） | "
                   f"{lab(1)}–{lab(kb)}：×{rb:.3f}，z={zb:+.2f}，扫描校正 p={p_step:.3f} | {p_chi:.3f} |")
    return out, K, m


GAMES = {"ssq": "双色球（福彩，红球 33 选 6）", "kl8": "快乐8（福彩，80 选 20）", "pl5": "排列5（体彩托帕斯，每位 0–9，可重复）"}
lines = ["# 跨彩种对照：小号偏多是不是大乐透特有？\n",
         "体彩各摇奖机于 2019-09-07 统一更新（大乐透维纳斯、排列3/5 托帕斯）。福彩为不同机构、不同设备，作为对照组。\n"]
for g, title in GAMES.items():
    out, K, m = analyse(g)
    lines.append(f"## {title}\n")
    lines.append("| 时段 | 期数 | 小号段（最小的约 1/3） | 最强台阶（从最小号起） | 整体频率 χ² p |")
    lines.append("|---|---|---|---|---|")
    lines += out
    lines.append("")
lines.append("对照：大乐透新机时期 01–11（35 个号中最小的 11 个）×1.07，z≈+3.5；PL 权重 ×1.11。")
text = "\n".join(lines) + "\n"
(ROOT / "report_cross.md").write_text(text, encoding="utf-8")
print(text)
