"""日本ロト“正在发生的热球”搜寻（与双色球红球 24 同一方法），并按セット球分组。输出 japan/report_hot.md。

对每个号码：在全历史上做 CUSUM 变点，取“变点之后至今”的出现率与 z 值（只看变点之后仍在持续、且偏热的情形）。
校准：在历史上所有同长度窗口里统计“最热号码的 z”，看当前 z 处于哪个分位（已包含“从全部号码里挑最热的”带来的多重比较）。
分套：对每一套セット球，只用使用该套的回做同样的分析（每套约 1/10 的回）。
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GAMES = {"loto6": ("ロト6", 43, 6), "loto7": ("ロト7", 37, 7), "miniloto": ("ミニロト", 31, 5)}
out = ["# 日本ロト：正在发生的热球\n",
       "方法同双色球红球 24：每个号码做 CUSUM 变点，取“变点至今”的偏热程度；用历史上所有同长度窗口的“最热号码 z”分布做校准。\n",
       "“历史分位”= 历史同长度窗口中，最热号码达到这一 z 值的比例；越小越罕见。< 1% 视为强线索。\n"]


def zscores(X, n_draw, K, M):
    p0 = M / K
    c = X.sum(0)
    return (c - n_draw * p0) / np.sqrt(n_draw * p0 * (1 - p0) * (K - M) / (K - 1))


def cusum_onset(x, min_tail=40):
    n = len(x)
    c = np.cumsum(x - x.mean())
    k = np.arange(1, n)
    st = np.abs(c[:-1]) / np.sqrt(k * (n - k) / n)
    lo = 30
    j = int(np.argmax(st[lo:n - min_tail])) + lo
    return j + 1


def hist_maxz(X, n, K, M, step=10):
    cs = np.vstack([np.zeros(K), np.cumsum(X, 0)])
    zs = []
    for s in range(0, len(X) - n, step):
        zs.append(zscores_from_counts(cs[s + n] - cs[s], n, K, M).max())
    return np.array(zs)


def zscores_from_counts(c, n, K, M):
    p0 = M / K
    return (c - n * p0) / np.sqrt(n * p0 * (1 - p0) * (K - M) / (K - 1))


def scan(X, ids, K, M, label, min_tail=40, top=3):
    rows = []
    for i in range(K):
        j = cusum_onset(X[:, i], min_tail)
        tail = X[j:]
        n = len(tail)
        z = zscores_from_counts(tail[:, i].sum(), n, K, M)
        if z <= 0:
            continue
        rows.append((z, i + 1, j, n, tail[:, i].mean()))
    rows.sort(reverse=True)
    res = []
    for z, num, j, n, rate in rows[:top]:
        hz = hist_maxz(X, n, K, M)
        pct = float(np.mean(hz >= z)) if len(hz) else float("nan")
        res.append((label, num, ids[j], n, rate, M / K, z, pct))
    return res


cands = []
for game, (title, K, M) in GAMES.items():
    D = list(csv.DictReader((ROOT / "data" / f"{game}.csv").open(encoding="utf-8")))
    R = np.array([[int(d[f"n{i}"]) for i in range(1, M + 1)] for d in D])
    ids = [int(d["id"]) for d in D]
    dates = [d["date"] for d in D]
    S = np.array([d["setball"] for d in D])
    X = np.zeros((len(R), K))
    X[np.arange(len(R))[:, None], R - 1] = 1
    out.append(f"## {title}（{len(R)} 回，最新 第{ids[-1]}回 {dates[-1]}）\n")
    out.append("| 范围 | 号码 | 变点起始回 | 此后回数 | 此后出现率 | 理论 | z | 历史分位 |")
    out.append("|---|---|---|---|---|---|---|---|")
    res = scan(X, ids, K, M, "全部回")
    for s in "ABCDEFGHIJ":
        m = S == s
        if m.sum() < 120:
            continue
        sub_ids = [i for i, k in zip(ids, m) if k]
        res += scan(X[m], sub_ids, K, M, f"{s} セット", min_tail=25, top=1)
    res.sort(key=lambda r: r[7])
    for label, num, start, n, rate, p0, z, pct in res[:12]:
        mark = " ←" if pct < 0.01 else ""
        out.append(f"| {label} | {num:02d} | 第{start}回 | {n} | {rate:.3f} | {p0:.3f} | {z:+.2f} | {pct:.3f}{mark} |")
        if pct < 0.01:
            cands.append((title, label, num, start, n, rate, p0, z, pct))
    out.append("")

out.append("## 小结\n")
if cands:
    out.append("历史分位 < 1% 的强线索：\n")
    for c in cands:
        out.append(f"- {c[0]} {c[1]}：号码 {c[2]:02d} 自第{c[3]}回起 {c[4]} 回出现率 {c[5]:.3f}（理论 {c[6]:.3f}），z={c[7]:+.2f}，历史分位 {c[8]:.3f}")
else:
    out.append("没有历史分位 < 1% 的强线索。")
out.append("\n注意：分套扫描对 10 套 × 全部号码做了搜索，单套结果的“历史分位”只校准了套内的号码多重比较，跨 10 套还需再乘约 10 倍。")
text = "\n".join(out) + "\n"
(ROOT / "report_hot.md").write_text(text, encoding="utf-8")
print(text)
