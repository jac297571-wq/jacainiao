"""动态贝叶斯 Plackett-Luce：把"每个球被抓到的难易度"当成随时间漂移的隐变量，逐期滤波，直接算组合（头奖）alpha。

模型（每个彩种）：
  θ_t ∈ R^K 是各号码的对数权重（越大越容易被抓）。开奖 = 按权重无放回抽 M 个（Plackett-Luce）。
  先验：θ_0 ~ N(0, τ²I + τg²G)；漂移：θ_t = θ_{t-1} + η，η ~ N(0, qI + qg·G)。
  G 是"号码组"结构（相邻号码一组：大乐透按 7 个一色，其余每 10 个一段）——
  世界语料告诉我们真实偏差是"整组、多年"，组项专门表达这一点；τ/q 表达单号码的固定差异与短期热度。
  有出球顺序的用顺序（每期多 log2(M!) 比特信息，比只看号码集合学得更快），没有的对全部顺序求和。
  滤波：拉普拉斯近似的卡尔曼滤波（每期一次牛顿更新），K ≤ 43，逐期精确。

输出：
  1. 超参数网格上的"前瞻比特增益"（每期先预测、再看结果、再更新，天然样本外），及其贝叶斯模型平均（BMA）权重。
     零假设校准：对同样长度的纯随机数据跑同一网格，看"最好格子的增益"有多大。
  2. 组合 alpha 的实盘回放：每期用"截至上期最好的格子"选出一注（权重最高的 M 个号码），记录它实际命中几个，
     与随机的期望命中数比较。头奖概率倍数 ≈ 多命中的程度在 M 个号码上的复合。
  3. 当前后验（BMA）下：每个号码的倍数与"确实偏热"的后验概率；最优单注、最优 5 注组合的头奖概率倍数
     （对后验做蒙特卡洛积分，包含参数不确定性）。
阳性对照：大乐透 2007–2013、德州 Lotto Texas 6/54（世界语料里发现的大号偏冷）。
用法：python3 deep/dynpl.py → report_dynpl.md
"""
import csv
import itertools
import sys
import time
from math import comb, log
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import radar  # noqa: E402

GRID = [(t, tg, q, qg) for t in (0.0, 0.05, 0.15) for tg in (0.0, 0.15) for q in (0.0, 1e-4, 1e-3) for qg in (0.0, 1e-4)]
NULL_SIMS = 6
out = []


def p(s=""):
    out.append(s)
    print(s, flush=True)


# ---------------- 数据 ----------------
def series():
    S = []
    D = list(csv.DictReader((ROOT / "data" / "dlt_history.csv").open(encoding="utf-8")))
    O = {r["issue"]: [int(r[f"o{i}"]) for i in range(1, 6)] for r in csv.DictReader((ROOT / "data" / "dlt_order.csv").open(encoding="utf-8"))}
    rows = [([int(d[f"r{i}"]) for i in range(1, 6)], O.get(d["issue"]), d["issue"]) for d in D]
    cut = next(i for i, r in enumerate(rows) if r[2] >= "14001")
    S.append(("大乐透 2007–2013（阳性对照）", 35, 5, 7, rows[:cut], False))
    S.append(("大乐透 2014–今", 35, 5, 7, rows[cut:], True))
    ss = []
    for line in (ROOT / "data" / "other" / "ssq.txt").open(encoding="utf-8", errors="ignore"):
        f = line.split()
        try:
            red = [int(x) for x in f[2:8]]
            o = [int(x) for x in f[9:15]]
        except (ValueError, IndexError):
            continue
        ss.append((red, o if sorted(o) == sorted(red) else None, f[0]))
    S.append(("双色球", 33, 6, 10, ss, True))
    for g, title, K, M in (("loto6", "ロト6", 43, 6), ("loto7", "ロト7", 37, 7), ("miniloto", "ミニロト", 31, 5)):
        rows = []
        for d in csv.DictReader((ROOT / "japan" / "data" / f"{g}.csv").open(encoding="utf-8")):
            n = [int(d[f"n{i}"]) for i in range(1, M + 1)]
            o = [int(d[f"o{i}"]) for i in range(1, M + 1)] if "o1" in d else None
            rows.append((n, o if o and sorted(o) == sorted(n) else None, d["id"]))
        S.append((title, K, M, 10, rows, True))
    tx = [([int(x) for x in r["numbers"].split()], None, r["date"])
          for r in csv.DictReader((ROOT / "data" / "world" / "tx_lotto_6of54.csv").open(encoding="utf-8"))]
    S.insert(1, ("德州 Lotto Texas 6/54（阳性对照）", 54, 6, 10, tx, False))
    return S


def group_matrix(K, size):
    B = np.zeros((K, (K + size - 1) // size))
    B[np.arange(K), np.arange(K) // size] = 1
    return B @ B.T


# ---------------- Plackett-Luce 工具 ----------------
PERMS = {M: np.array(list(itertools.permutations(range(M)))) for M in range(1, 8)}


def set_logp(theta, s):
    """无序集合的精确对数概率（对所有出球顺序求和）"""
    th = theta - theta.max()
    w = np.exp(th)
    Z = w.sum()
    ws = w[s][PERMS[len(s)]]
    den = Z - np.cumsum(ws, 1) + ws
    lp = (np.log(ws) - np.log(den)).sum(1)
    m = lp.max()
    return m + np.log(np.exp(lp - m).sum())


def grad_hess(theta, order):
    """有序观测的对数似然梯度与（负）Hessian"""
    w = np.exp(theta - theta.max())
    K = len(theta)
    g = np.zeros(K)
    H = np.zeros((K, K))
    avail = np.ones(K, bool)
    for o in order:
        pk = np.where(avail, w, 0)
        pk /= pk.sum()
        g -= pk
        g[o] += 1
        H += np.diag(pk) - np.outer(pk, pk)
        avail[o] = False
    return g, H


def grad_hess_set(theta, s, rng):
    """无序观测：梯度按各顺序的后验权重平均；Hessian 用一个随机顺序近似（Fisher 信息量级相同）"""
    P = PERMS[len(s)]
    w = np.exp(theta - theta.max())
    Z = w.sum()
    ws = w[s][P]
    den = Z - np.cumsum(ws, 1) + ws
    lp = (np.log(ws) - np.log(den)).sum(1)
    sp = np.exp(lp - lp.max())
    sp /= sp.sum()
    inv = 1 / den                                  # (P, M)
    C = np.cumsum(inv, 1)                          # 第 k 步之前（含）所有步的 1/Z 之和
    g = -w * (sp @ C[:, -1])                        # 集合外号码：每一步都在候选里
    pos = np.argsort(P, 1)                         # pos[π, i] = 集合第 i 个号码在顺序 π 中的位置
    Ci = np.take_along_axis(C, pos, 1)             # 集合内号码只在被抽中之前（含）的步骤里是候选
    g[s] = 1 - w[s] * (sp @ Ci)
    order = np.asarray(s)[rng.permutation(len(s))]
    _, H = grad_hess(theta, order)
    return g, H


# ---------------- 滤波 ----------------
def run(rows, K, M, gsize, cell, rng, keep=False):
    t, tg, q, qg = cell
    G = group_matrix(K, gsize)
    I = np.eye(K)
    m = np.zeros(K)
    P = t ** 2 * I + tg ** 2 * G
    Q = q * I + qg * G
    base = -log(comb(K, M))
    gains = np.zeros(len(rows))
    hits = np.zeros(len(rows))
    for i, (s, o, _) in enumerate(rows):
        P = P + Q
        s0 = np.array(s) - 1
        gains[i] = (set_logp(m, s0) - base) / log(2)
        hits[i] = np.isin(np.argsort(-m)[:M], s0).sum() if np.ptp(m) > 0 else np.nan
        if not P.any():
            continue
        if o is not None:
            g, H = grad_hess(m, np.array(o) - 1)
        else:
            g, H = grad_hess_set(m, s0, rng)
        Pn = P @ np.linalg.inv(I + H @ P)
        m = m + Pn @ g
        m -= m.mean()
        P = (Pn + Pn.T) / 2
    return (gains, hits, m, P) if keep else (gains, hits)


def simulate_rows(K, M, n, rng, ordered):
    rows = []
    for _ in range(n):
        o = list(rng.permutation(K)[:M] + 1)
        rows.append((sorted(o), o if ordered else None, ""))
    return rows


# ---------------- 组合 alpha ----------------
def ticket_mult(sets, m, P, K, M, rng, n=300):
    """后验下各注的头奖概率倍数（对 θ ~ N(m,P) 做蒙特卡洛积分）"""
    L = np.linalg.cholesky(P + 1e-12 * np.eye(K))
    th = m + rng.standard_normal((n, K)) @ L.T
    base = 1 / comb(K, M)
    return np.array([np.mean([np.exp(set_logp(t, np.array(s))) for t in th]) / base for s in sets])


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    p("# 动态贝叶斯 Plackett-Luce：逐期滤波与组合 alpha\n")
    p("网格：τ（单号码固定差异）∈{0,0.05,0.15}，τg（整组固定差异）∈{0,0.15}，q（单号码每期漂移方差）∈{0,1e-4,1e-3}，"
      "qg（整组漂移）∈{0,1e-4}，共 36 格。全 0 的格子就是“纯随机”，增益恒为 0。\n")
    summary = []
    for name, K, M, gsize, rows, target in series():
        res = {c: run(rows, K, M, gsize, c, rng) for c in GRID}
        tot = {c: res[c][0].sum() for c in GRID}
        best = max(GRID, key=tot.get)
        # BMA 权重 ∝ 2^{总比特}
        lw = np.array([tot[c] for c in GRID]) * log(2)
        bw = np.exp(lw - lw.max())
        bw /= bw.sum()
        # 零假设校准
        ordered = sum(o is not None for _, o, _ in rows) > len(rows) / 2
        null = []
        for _ in range(NULL_SIMS):
            sim = simulate_rows(K, M, len(rows), rng, ordered)
            null.append(max(run(sim, K, M, gsize, c, rng)[0].sum() for c in GRID))
        null = np.array(null)
        # 实盘回放：每期用截至上期最好的格子
        cum = np.cumsum(np.array([res[c][0] for c in GRID]), 1)
        H = np.array([res[c][1] for c in GRID])
        choose = np.argmax(np.hstack([np.zeros((len(GRID), 1)), cum[:, :-1]]), 0)
        h = H[choose, np.arange(len(rows))]
        h = h[~np.isnan(h)]
        eh = M * M / K
        vh = M * (M / K) * (1 - M / K) * (K - M) / (K - 1)
        zh = (h.mean() - eh) / np.sqrt(vh / len(h)) if len(h) else float("nan")
        p(f"## {name}（{len(rows)} 期，K={K}，M={M}，组={gsize} 个一组）\n")
        p(f"- 最好的格子 τ={best[0]}、τg={best[1]}、q={best[2]}、qg={best[3]}：前瞻总增益 **{tot[best]:+.2f} 比特**"
          f"（平均每期头奖概率 ×{2 ** (tot[best] / len(rows)):.4f}）")
        p(f"- 纯随机数据上同一网格的“最好格子增益”：{', '.join(f'{x:+.1f}' for x in sorted(null))}"
          f" → 实际值超过其中 {np.mean(null < tot[best]) * 100:.0f}%")
        top3 = np.argsort(-bw)[:3]
        p("- BMA 权重最高的格子：" + "；".join(f"(τ={GRID[i][0]},τg={GRID[i][1]},q={GRID[i][2]},qg={GRID[i][3]}) {bw[i]:.2f}" for i in top3))
        p(f"- 实盘回放（每期选权重最高的 {M} 个号码作一注）：平均命中 {h.mean():.4f} 个，随机期望 {eh:.4f}，z={zh:+.2f}（{len(h)} 期）")
        summary.append((name, tot[best], np.mean(null < tot[best]), h.mean() / eh if len(h) else float("nan"), zh))
        if target:
            # 当前后验：用 BMA 权重最高的格子（权重 >0.05 的格子做混合）
            sets_all, mix = [], []
            for i in np.argsort(-bw):
                if bw[i] < 0.05:
                    break
                _, _, m, P = run(rows, K, M, gsize, GRID[i], rng, keep=True)
                mix.append((bw[i], m, P))
            wsum = sum(w for w, _, _ in mix)
            mbar = sum(w * m for w, m, _ in mix) / wsum
            sd = np.sqrt(sum(w * np.diag(P) for w, _, P in mix) / wsum)
            order = np.argsort(-mbar)
            p(f"- 当前后验（BMA）：权重最高的号码 " + "，".join(
                f"{i + 1:02d}（×{np.exp(mbar[i]):.3f}，偏热概率 {0.5 * (1 + __import__('math').erf(mbar[i] / (sd[i] * np.sqrt(2) + 1e-12))):.2f}）"
                for i in order[:6]))
            cand = [list(c) for c in itertools.combinations(sorted(order[:M + 3]), M)]
            mult = np.zeros(len(cand))
            for w, m, P in mix:
                mult += w / wsum * ticket_mult(cand, m, P, K, M, rng)
            idx = np.argsort(-mult)
            best5 = idx[:5]
            p(f"- 组合 alpha：最优单注 {' '.join(f'{x + 1:02d}' for x in cand[idx[0]])} 头奖概率 ×{mult[idx[0]]:.3f}；"
              f"最优 5 注合计 ×{mult[best5].mean():.3f}（相对随机 5 注）")
            p("  - 5 注：" + "；".join(" ".join(f"{x + 1:02d}" for x in cand[i]) for i in best5))
        p("")
    p("## 汇总\n")
    p("| 序列 | 前瞻总增益（比特） | 超过纯随机的比例 | 实盘命中 / 随机期望 | z |")
    p("|---|---|---|---|---|")
    for n, g, pct, r, z in summary:
        p(f"| {n} | {g:+.2f} | {pct * 100:.0f}% | ×{r:.4f} | {z:+.2f} |")
    p(f"\n用时 {time.time() - t0:.0f} 秒。")
    (ROOT / "report_dynpl.md").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
