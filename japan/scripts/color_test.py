"""日本ロト颜色效应检验。颜色表来自 loto6.jp / loto7.jp / miniloto.jp 的“セット球の色の確認”页（japan/data/setball_colors.json）：
号码按 mod 7 分成 7 组，每套セット球把 7 种颜色轮换分配给这 7 组。颜色与号码因此在各套之间“解耦”，形成天然实验：
  颜色效应 → 按颜色重新编码后出现一致偏差；号码效应 → 按号码组（不轮换）出现偏差。
输出 japan/report_color.md。
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
COLORS = json.load((ROOT / "data" / "setball_colors.json").open(encoding="utf-8"))
PALETTE = ["赤", "橙", "黄", "緑", "青", "紺", "紫"]
GAMES = {"loto6": ("ロト6", 43, 6), "loto7": ("ロト7", 37, 7), "miniloto": ("ミニロト", 31, 5)}
NEW_BALLS = "2019-04-01"  # 2019 年 4 月起启用新セット球（配色变更）
rng = np.random.default_rng(77)
SIMS = 2000
out = ["# 日本ロト：颜色效应检验\n",
       "每套セット球把 7 种颜色轮换分配给号码的 7 个 mod-7 组，同一号码在不同套里颜色不同。",
       "**颜色检验**：按当回所用套的颜色给开出的球计数；**号码组对照**：按 mod-7 号码组计数（不随套轮换）。",
       "统计量为 7 类的 χ²（每回期望按该类球数精确计算），p 值来自保留真实套号序列的蒙特卡洛模拟。\n"]


def setup(game):
    title, K, M = GAMES[game]
    D = list(csv.DictReader((ROOT / "data" / f"{game}.csv").open(encoding="utf-8")))
    groups = COLORS[game]["groups"]
    grp_of = np.zeros(K + 1, int)
    for g, nums in enumerate(groups):
        for x in nums:
            grp_of[x] = g
    color_of = {}  # (set, number) -> color index
    for s, cols in COLORS[game]["sets"].items():
        color_of[s] = np.array([PALETTE.index(cols[grp_of[x]]) if x > 0 else -1 for x in range(K + 1)])
    R = np.array([[int(d[f"n{i}"]) for i in range(1, M + 1)] for d in D])
    S = [d["setball"] for d in D]
    O = np.array([[int(d[f"o{i}"]) for i in range(1, M + 1)] for d in D]) if "o1" in D[0] else None
    dates = np.array([d["date"] for d in D])
    return title, K, M, R, S, O, dates, grp_of, color_of


def stat(R, S, K, M, keyfun):
    """7 类 χ²：O_c 与按每回实际类大小计算的期望 E_c"""
    Oc = np.zeros(7)
    Ec = np.zeros(7)
    for r, s in zip(R, S):
        lab = keyfun(s)
        sizes = np.bincount(lab[1:], minlength=7)
        Ec += M * sizes / K
        Oc += np.bincount(lab[r], minlength=7)
    return float(((Oc - Ec) ** 2 / Ec).sum()), Oc / Ec


def mc(R, S, K, M, keyfun, obs):
    n = len(R)
    sims = []
    for _ in range(SIMS):
        Rs = np.argsort(rng.random((n, K)), 1)[:, :M] + 1
        sims.append(stat(Rs, S, K, M, keyfun)[0])
    return (np.sum(np.array(sims) >= obs) + 1) / (SIMS + 1)


def pl_cv(O, S, K, M, keyfun):
    """7 个类别权重的 Plackett-Luce 模型，5 折时间分块 CV（比特）"""
    labs = np.array([keyfun(s) for s in S])  # (n, K+1)

    def nll(th, idx):
        lw = np.concatenate([[0.0], th])  # 第 0 类为基准
        v = 0.0
        g = np.zeros(6)
        for i in idx:
            w = np.exp(lw[labs[i][1:]])
            avail = np.ones(K, bool)
            for x in O[i]:
                tot = w[avail].sum()
                c = labs[i][x]
                v -= lw[c] - np.log(tot)
                # 梯度
                for cc in range(1, 7):
                    g[cc - 1] += (w[avail & (labs[i][1:] == cc)]).sum() / tot
                if c > 0:
                    g[c - 1] -= 1
                avail[x - 1] = False
        return v, g

    n = len(O)
    ed = np.linspace(0, n, 6).astype(int)
    gains = []
    for f in range(5):
        te = np.arange(ed[f], ed[f + 1])
        tr = np.setdiff1d(np.arange(n), te)
        th = minimize(lambda t: tuple(np.add(nll(t, tr), (t @ t / 2, t))) if False else
                      (nll(t, tr)[0] + t @ t / 2, nll(t, tr)[1] + t), np.zeros(6), jac=True, method="L-BFGS-B").x
        gains.append((nll(np.zeros(6), te)[0] - nll(th, te)[0]) / np.log(2))
    return gains


for game in GAMES:
    title, K, M, R, S, O, dates, grp_of, color_of = setup(game)
    out.append(f"## {title}（{len(R)} 回）\n")
    out.append("| 时段 | 回数 | 检验 | χ² | p | 各类 实际/期望（赤 橙 黄 緑 青 紺 紫 ／ 号码组 1–7） |")
    out.append("|---|---|---|---|---|---|")
    color_key = lambda s: color_of[s]
    group_key = lambda s: grp_of
    for name, m in (("全部", dates >= "0"), ("2019-04 前（旧球）", dates < NEW_BALLS), ("2019-04 起（新球）", dates >= NEW_BALLS)):
        if m.sum() < 50:
            continue
        Rm = R[m]
        Sm = [s for s, k in zip(S, m) if k]
        for lab, key in (("颜色", color_key), ("号码组", group_key)):
            x2, ratio = stat(Rm, Sm, K, M, key)
            pv = mc(Rm, Sm, K, M, key, x2)
            out.append(f"| {name} | {m.sum()} | {lab} | {x2:.1f} | {pv:.3f} | {' '.join(f'{v:.3f}' for v in ratio)} |")
    if O is not None:
        for name, m in (("2019-04 前", dates < NEW_BALLS), ("2019-04 起", dates >= NEW_BALLS)):
            Om, Sm = O[m], [s for s, k in zip(S, m) if k]
            g_col = pl_cv(Om, Sm, K, M, color_key)
            g_grp = pl_cv(Om, Sm, K, M, group_key)
            out.append(f"\n出球顺序模型 5 折 CV（{name}）：颜色权重 {sum(g_col):+.2f} 比特（{' '.join(f'{x:+.1f}' for x in g_col)}）；"
                       f"号码组权重 {sum(g_grp):+.2f} 比特（{' '.join(f'{x:+.1f}' for x in g_grp)}）")
    out.append("")

text = "\n".join(out) + "\n"
(ROOT / "report_color.md").write_text(text, encoding="utf-8")
print(text)
