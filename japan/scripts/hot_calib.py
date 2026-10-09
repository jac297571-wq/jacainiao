"""热球扫描的正确校准：对纯随机数据执行“完全相同的扫描流程”（每个号码 CUSUM 选变点 → 取变点后 z → 取全部号码最大值），
得到最大 z 的零分布；分套扫描再对 10 套取最大。同时复核双色球红球 24。输出 japan/report_hot_calib.md。"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(99)
SIMS = 400


def scan_max(X, K, M, min_tail):
    """X: (n, K) 0/1。对每列做 CUSUM 选变点，返回 (最大 z, 号码, 变点下标, 尾长)"""
    n = len(X)
    mu = X.mean(0)
    c = np.cumsum(X - mu, 0)[:-1]
    k = np.arange(1, n)[:, None]
    st = np.abs(c) / np.sqrt(k * (n - k) / n)
    lo = 30
    st[:lo] = -1
    st[n - min_tail:] = -1
    j = st.argmax(0) + 1  # 每列的变点
    p0 = M / K
    best = (-9, -1, -1, -1)
    for i in range(K):
        tail = X[j[i]:, i]
        m = len(tail)
        z = (tail.sum() - m * p0) / np.sqrt(m * p0 * (1 - p0) * (K - M) / (K - 1))
        if z > best[0]:
            best = (z, i + 1, j[i], m)
    return best


def rand_X(n, K, M):
    X = np.zeros((n, K))
    idx = np.argsort(rng.random((n, K)), 1)[:, :M]
    X[np.arange(n)[:, None], idx] = 1
    return X


def null_dist(n, K, M, min_tail, sims=SIMS):
    return np.array([scan_max(rand_X(n, K, M), K, M, min_tail)[0] for _ in range(sims)])


out = ["# 热球扫描的正确校准（对随机数据执行相同流程）\n",
       "校正 p = 纯随机数据经过同样扫描，最大 z 达到观测值的比例。分套扫描再按 10 套取最大做校正。\n",
       "| 彩种 | 范围 | 最热号码 | 变点起始 | 此后回数 | z | 校正 p |", "|---|---|---|---|---|---|---|"]

# 双色球复核
rows = []
for line in (ROOT.parent / "data" / "other" / "ssq.txt").open(encoding="utf-8", errors="ignore"):
    f = line.split()
    try:
        rows.append((f[0], [int(x) for x in f[2:8]]))
    except (ValueError, IndexError):
        pass
X = np.zeros((len(rows), 33))
for i, (_, r) in enumerate(rows):
    X[i, np.array(r) - 1] = 1
z, num, j, m = scan_max(X, 33, 6, 40)
nd = null_dist(len(X), 33, 6, 40)
out.append(f"| 双色球 | 全部期 | 红球 {num:02d} | {rows[j][0]} | {m} | {z:+.2f} | {(np.sum(nd >= z) + 1) / (len(nd) + 1):.3f} |")
# 只看“变点在最近 3 年内”的号码（当前活跃的热球）——与实际使用场景一致
print("ssq done", flush=True)

GAMES = {"loto6": ("ロト6", 43, 6), "loto7": ("ロト7", 37, 7), "miniloto": ("ミニロト", 31, 5)}
for game, (title, K, M) in GAMES.items():
    D = list(csv.DictReader((ROOT / "data" / f"{game}.csv").open(encoding="utf-8")))
    R = np.array([[int(d[f"n{i}"]) for i in range(1, M + 1)] for d in D])
    ids = [int(d["id"]) for d in D]
    S = np.array([d["setball"] for d in D])
    X = np.zeros((len(R), K))
    X[np.arange(len(R))[:, None], R - 1] = 1
    z, num, j, m = scan_max(X, K, M, 40)
    nd = null_dist(len(X), K, M, 40)
    out.append(f"| {title} | 全部回 | {num:02d} | 第{ids[j]}回 | {m} | {z:+.2f} | {(np.sum(nd >= z) + 1) / (len(nd) + 1):.3f} |")
    # 分套：取 10 套中的最大 z，与“10 套各自随机后取最大”的零分布比较
    per = []
    for s in "ABCDEFGHIJ":
        msk = S == s
        sub_ids = [i for i, k in zip(ids, msk) if k]
        zz, nn, jj, mm = scan_max(X[msk], K, M, 25)
        per.append((zz, s, nn, sub_ids[jj], mm, msk.sum()))
    per.sort(reverse=True)
    sizes = [x[5] for x in per]
    nd10 = np.array([max(scan_max(rand_X(n, K, M), K, M, 25)[0] for n in sizes) for _ in range(SIMS // 4)])
    zz, s, nn, st, mm, _ = per[0]
    out.append(f"| {title} | 分套（10 套取最大） | {s} セット {nn:02d} | 第{st}回 | {mm} | {zz:+.2f} | {(np.sum(nd10 >= zz) + 1) / (len(nd10) + 1):.3f} |")
    print(game, "done", flush=True)

text = "\n".join(out) + "\n"
(ROOT / "report_hot_calib.md").write_text(text, encoding="utf-8")
print(text)
