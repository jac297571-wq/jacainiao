"""三个选号器共用：动态贝叶斯 Plackett-Luce（deep/dynpl.py）的最优 5 注，每注都含雷达热球。

做法：对 36 个超参数格子逐期滤波到最新一期，按前瞻证据做贝叶斯模型平均（BMA），
在当前后验下（对参数不确定性做蒙特卡洛积分）计算候选组合的头奖概率倍数，取最高的 5 注。
有热球时，候选组合都必须含热球（= 把最优组合里最弱的号码换成热球），并报告这样做损失了多少。
"""
import itertools
import sys
from math import erf, log, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "deep"))
import dynpl  # noqa: E402

NAMES = {"大乐透": "大乐透 2014–今", "双色球": "双色球", "ロト6": "ロト6", "ロト7": "ロト7", "ミニロト": "ミニロト"}


def posterior(game, rng):
    for name, K, M, gsize, rows, _ in dynpl.series():
        if name == NAMES[game]:
            break
    res = {c: dynpl.run(rows, K, M, gsize, c, rng, keep=True) for c in dynpl.GRID}
    lw = np.array([res[c][0].sum() for c in dynpl.GRID]) * log(2)
    bw = np.exp(lw - lw.max())
    bw /= bw.sum()
    mix = [(bw[i], res[c][2], res[c][3]) for i, c in enumerate(dynpl.GRID) if bw[i] >= 0.05]
    tot = sum(w for w, _, _ in mix)
    mix = [(w / tot, m, P) for w, m, P in mix]
    best = dynpl.GRID[int(np.argmax(bw))]
    return K, M, mix, best, float(bw.max())


def bayes_tickets(game, hot=None, n=5, seed=None):
    rng = np.random.default_rng(seed)
    K, M, mix, best, bwmax = posterior(game, rng)
    mbar = sum(w * m for w, m, _ in mix)
    sd = np.sqrt(sum(w * np.diag(P) for w, _, P in mix))
    order = [int(i) for i in np.argsort(-mbar)]

    def mults(cands):
        out = np.zeros(len(cands))
        for w, m, P in mix:
            out += w * dynpl.ticket_mult(cands, m, P, K, M, rng, n=200)
        return out

    if np.ptp(mbar) < 1e-9:                       # 后验完全平坦（证据支持纯随机）：随机选号
        import secrets
        sr = secrets.SystemRandom()
        tickets = []
        while len(tickets) < n:
            t = sorted(([hot] if hot else []) + sr.sample([x for x in range(1, K + 1) if x != hot], M - (1 if hot else 0)))
            if t not in tickets:
                tickets.append(t)
        flat = {"bma_best_cell": dict(zip(("tau", "tau_g", "q", "q_g"), best)), "bma_best_weight": round(bwmax, 3),
                "top": [], "ticket_mult": [1.0] * n, "portfolio_mult": 1.0, "portfolio_mult_without_hot": 1.0, "flat": True}
        return tickets, flat
    free = [list(c) for c in itertools.combinations(sorted(order[:M + 3]), M)]
    fm = mults(free)
    if hot is not None:
        h = hot - 1
        rest = [i for i in order if i != h][:M + 2]
        cands = [sorted([h, *c]) for c in itertools.combinations(rest, M - 1)]
        cm = mults(cands)
    else:
        cands, cm = free, fm
    idx = np.argsort(-cm)[:n]
    tickets = [[x + 1 for x in cands[i]] for i in idx]
    hotp = lambda i: 0.5 * (1 + erf(mbar[i] / (sd[i] * sqrt(2) + 1e-12)))
    info = {
        "bma_best_cell": dict(zip(("tau", "tau_g", "q", "q_g"), best)), "bma_best_weight": round(bwmax, 3),
        "top": [{"n": i + 1, "mult": round(float(np.exp(mbar[i])), 4), "p_hot": round(hotp(i), 3)} for i in order[:6]],
        "ticket_mult": [round(float(cm[i]), 4) for i in idx],
        "portfolio_mult": round(float(cm[idx].mean()), 4),
        "portfolio_mult_without_hot": round(float(np.sort(fm)[::-1][:n].mean()), 4),
    }
    return tickets, info


def describe(info, hot):
    if info.get("flat"):
        return "贝叶斯后验完全平坦（证据支持纯随机），随机选号；5 注合计头奖概率 ×1.000"
    top = "，".join(f"{t['n']:02d}（×{t['mult']:.3f}，偏热概率 {t['p_hot']:.2f}）" for t in info["top"][:4])
    s = f"贝叶斯后验最热：{top}\n5 注合计头奖概率 ×{info['portfolio_mult']:.3f}"
    if hot is not None:
        s += f"（不强制含热球 {hot:02d} 时为 ×{info['portfolio_mult_without_hot']:.3f}）"
    return s


if __name__ == "__main__":
    g = sys.argv[1] if len(sys.argv) > 1 else "ミニロト"
    hot = int(sys.argv[2]) if len(sys.argv) > 2 else None
    t, info = bayes_tickets(g, hot)
    print(describe(info, hot))
    for x in t:
        print(" ".join(f"{v:02d}" for v in x))
