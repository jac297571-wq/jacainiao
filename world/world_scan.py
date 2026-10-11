"""世界彩票语料库上的向量化"热球"研究。输出 world/report_world.md。

问题：真实的摇奖机偏差长什么样？在一台机器上学到的偏差模式，能否迁移到另一台机器（大乐透、双色球、日本ロト）？

1. 逐条序列体检：全历史均匀性 χ²；雷达回测（证明档报警后 30 期的出现率倍数）。
2. 两个热球探测网络（每个号码一个"近况向量"，网络对号码共享权重，不认识号码本身）：
   S：只在模拟数据上训练（随机注入热球）。
   R：在真实世界语料上训练，严格"留一族"：给某一族（纽约 / 德州 / 中国数字型 / 中国乐透 / 日本）打分时，
      训练数据完全不含这一族。这样学到的是"真实摇奖机偏差的普遍形态"，而不是背某台机器的历史。
   打分：逐期前瞻，实际开出组合在模型下的概率 / 均匀随机概率 → 头奖概率倍数。
3. 用全部语料训练的 R 网络，给出大乐透、双色球、日本三种的当前号码权重。
"""
import csv
import itertools
import json
import sys
import time
from math import comb, log
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import radar  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(0)
rng = np.random.default_rng(0)
HL = np.array([3, 6, 12, 25, 50, 100, 160.0])
DEC = 0.5 ** (1 / HL)
F = len(HL) + 2
TH_FILE = ROOT / "data" / "world" / "thresholds.json"
out = []


def p(s=""):
    out.append(s)
    print(s, flush=True)


# ---------------- 数据 ----------------
def load_series():
    S = []
    for f in sorted((ROOT / "data" / "world").glob("*.csv")):
        name = f.stem
        M, K = map(int, name.rsplit("_", 1)[1].split("of"))
        rows = [[int(x) for x in r["numbers"].split()] for r in csv.DictReader(f.open(encoding="utf-8"))]
        fam = "中国数字型" if name.startswith("cn_") else ("纽约" if name.startswith("ny_") else "德州")
        S.append({"name": name, "fam": fam, "K": K, "M": M, "R": rows})
    dlt = radar.load("dlt")
    cut = next(i for i, (iss, _) in enumerate(dlt) if iss >= "14001")
    S.append({"name": "大乐透 2007–2013", "fam": "中国乐透", "K": 35, "M": 5, "R": [r for _, r in dlt[:cut]]})
    S.append({"name": "大乐透 2014–今", "fam": "中国乐透", "K": 35, "M": 5, "R": [r for _, r in dlt[cut:]], "target": "大乐透"})
    S.append({"name": "双色球", "fam": "中国乐透", "K": 33, "M": 6, "R": [r for _, r in radar.load("ssq")], "target": "双色球"})
    for ld, title, K, M in (("loto6", "ロト6", 43, 6), ("loto7", "ロト7", 37, 7), ("miniloto", "ミニロト", 31, 5)):
        S.append({"name": title, "fam": "日本", "K": K, "M": M, "R": [r for _, r in radar.load(ld)], "target": title})
    for s in S:
        R = np.asarray(s["R"])
        s["X"] = np.zeros((len(R), s["K"]), np.float32)
        s["X"][np.arange(len(R))[:, None], R - 1] = 1
        s["S"] = R - 1
    return S


def features(X, K, M):
    p0 = M / K
    E = np.zeros(X.shape[:-1][:-1] + (K, len(HL)))
    cum = np.zeros(X.shape[:-2] + (K,))
    f = np.zeros(X.shape + (F,), np.float32)
    norm = np.sqrt(p0 * (1 - p0) / (1 - DEC ** 2))
    for t in range(X.shape[-2]):
        x = X[..., t, :]
        E = E * DEC + (x[..., None] - p0)
        cum += x - p0
        f[..., t, :, :len(HL)] = E / norm
        f[..., t, :, len(HL)] = cum / np.sqrt((t + 1) * p0 * (1 - p0))
        f[..., t, :, len(HL) + 1] = np.log(p0)
    return f


def pl_set_logp(logw, sets, perms):
    w = torch.exp(logw - logw.max(-1, keepdim=True).values)
    tot = w.sum(-1)
    ws = torch.gather(w, -1, sets)[..., perms]
    den = tot[..., None, None] - torch.cumsum(ws, -1) + ws
    return torch.logsumexp((torch.log(ws) - torch.log(den)).sum(-1), -1)


PERMS = {}


def perms(M):
    if M not in PERMS:
        PERMS[M] = torch.tensor(list(itertools.permutations(range(M))))
    return PERMS[M]


# ---------------- 1. 体检 ----------------
def chi2(s):
    K, M, X = s["K"], s["M"], s["X"]
    n = len(X)
    c = X.sum(0)
    e = n * M / K
    x = ((c - e) ** 2 / e).sum() * (K - 1) / (K - M) if M > 1 else ((c - e) ** 2 / e).sum()
    return stats.chi2.sf(x, K - 1)


def thresholds(K, M):
    th = json.loads(TH_FILE.read_text()) if TH_FILE.exists() else {}
    key = f"{K}_{M}"
    if key not in th:
        th[key] = radar.calibrate(K, M, rng, 300)
        TH_FILE.write_text(json.dumps(th))
    return th[key]


def backtest(s, L=30):
    K, M, X = s["K"], s["M"], s["X"]
    H = thresholds(K, M)
    up, down = radar.increments(K, M)
    S = np.zeros(K)
    cool = np.zeros(K, int)
    hits = n = ev = 0
    for t in range(len(X) - 1):
        S = np.maximum(0.0, S + np.where(X[t] > 0, up, down))
        cool -= 1
        for i in np.flatnonzero((S >= H) & (cool <= 0)):
            fut = X[t + 1:t + 1 + L, i]
            hits += fut.sum()
            n += len(fut)
            ev += 1
            cool[i] = L
            S[i] = 0
    p0 = M / K
    if n == 0:
        return ev, float("nan"), float("nan")
    return ev, hits / n / p0, (hits - n * p0) / np.sqrt(n * p0 * (1 - p0))


# ---------------- 2. 探测网络 ----------------
class Det(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(F, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)

    def forward(self, f):
        return self.f(f).squeeze(-1)


def simulate(K, M, B, T):
    logw = np.zeros((B, T, K))
    for b in range(B):
        if rng.random() < 0.6:
            for _ in range(rng.integers(1, 3)):
                i, L = rng.integers(K), rng.integers(30, 300)
                st = rng.integers(0, T - 30)
                logw[b, st:st + L, i] += np.log(rng.uniform(1.3, 2.5))
    S = np.argsort(-(logw + rng.gumbel(size=logw.shape)), -1)[..., :M]
    X = np.zeros((B, T, K), np.float32)
    np.put_along_axis(X, S, 1, -1)
    return X, S


def train(pool, steps=3000):
    """pool: [(feat (T,K,F) tensor, sets (T,M) tensor, M)]，feat[t] 预测 sets[t]。每步从一条序列里抽一批"""
    net = Det()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for it in range(steps):
        f, s, M = pool[rng.integers(len(pool))]
        t = torch.from_numpy(rng.integers(0, len(f), 256))
        loss = -pl_set_logp(net(f[t]), s[t], perms(M)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def real_pool(series):
    pool = []
    for s in series:
        f = torch.from_numpy(s["F"][99:-1])
        pool.append((f, torch.from_numpy(s["S"][100:]).long(), s["M"]))
    return pool


def score(net, s):
    with torch.no_grad():
        lp = pl_set_logp(net(torch.from_numpy(s["F"][99:-1])), torch.from_numpy(s["S"][100:]).long(), perms(s["M"])).numpy()
    return (lp + log(comb(s["K"], s["M"]))) / log(2)


def fmt_gain(g):
    if len(g) < 2:
        return "–", "–"
    z = g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))
    return f"×{2 ** g.mean():.4f}", f"{z:+.2f}"


def main():
    t0 = time.time()
    S = load_series()
    for s in S:
        s["F"] = features(s["X"], s["K"], s["M"])
    p("# 世界彩票语料库：真实摇奖机偏差与迁移学习\n")
    p(f"语料：{len(S)} 条序列，共 {sum(len(s['R']) for s in S)} 期（纽约、德州、中国数字型每位一台机器、大乐透、双色球、日本ロト）。\n")

    # 训练 S 网络（模拟）
    simpool = []
    for K, M in sorted({(s["K"], s["M"]) for s in S}):
        X, Ss = simulate(K, M, 48, 500)
        f = torch.from_numpy(features(X, K, M)[:, 99:-1].reshape(-1, K, F))
        simpool.append((f, torch.from_numpy(Ss[:, 100:].reshape(-1, M)).long(), M))
    netS = train(simpool)
    print(f"S 网络训练完成 {time.time() - t0:.0f}s", flush=True)

    # 训练 R 网络（留一族）
    fams = sorted({s["fam"] for s in S})
    netR = {}
    for fam in fams:
        netR[fam] = train(real_pool([s for s in S if s["fam"] != fam]))
        print(f"R 网络（不含{fam}）完成 {time.time() - t0:.0f}s", flush=True)

    p("## 逐条序列体检 + 两种网络的前瞻打分\n")
    p("“雷达”：证明档报警后 30 期该号码的出现率倍数。“S/R 网络”：逐期前瞻的头奖概率倍数（几何平均）及 z。R 网络对每一族都是留一族训练的。\n")
    p("| 序列 | 族 | 期数 | 均匀性 p | 雷达报警 | 雷达倍数 | z | S 网络 | z | R 网络（留一族） | z |")
    p("|---|---|---|---|---|---|---|---|---|---|---|")
    agg = {"S": [], "R": []}
    for s in S:
        ev, mult, zr = backtest(s)
        gS, gR = score(netS, s), score(netR[s["fam"]], s)
        agg["S"].append((s["fam"], gS))
        agg["R"].append((s["fam"], gR))
        a, b = fmt_gain(gS)
        c, d = fmt_gain(gR)
        p(f"| {s['name']} | {s['fam']} | {len(s['R'])} | {chi2(s):.3f} | {ev} | ×{mult:.3f} | {zr:+.2f} | {a} | {b} | {c} | {d} |")

    p("\n## 按族汇总（R 网络留一族）\n")
    p("| 族 | S 网络倍数 | z | R 网络倍数 | z |")
    p("|---|---|---|---|---|")
    for fam in fams:
        gs = np.concatenate([g for f_, g in agg["S"] if f_ == fam])
        gr = np.concatenate([g for f_, g in agg["R"] if f_ == fam])
        a, b = fmt_gain(gs)
        c, d = fmt_gain(gr)
        p(f"| {fam} | {a} | {b} | {c} | {d} |")

    # 3. 全语料 R 网络的当前权重（目标彩种自身不参与训练）
    p("\n## 当前号码权重（R 网络：用除目标所在族以外的全部真实语料训练）\n")
    p("| 彩种 | 权重最高的号码（倍数 = 下一期出现概率 / 平均） |")
    p("|---|---|")
    for s in S:
        if "target" not in s:
            continue
        with torch.no_grad():
            w = torch.softmax(netR[s["fam"]](torch.from_numpy(s["F"][-1])), -1).numpy() * s["K"]
        top = np.argsort(-w)[:4]
        p(f"| {s['target']} | {'，'.join(f'{i + 1:02d}（×{w[i]:.2f}）' for i in top)} |")
    p(f"\n用时 {time.time() - t0:.0f} 秒（CPU）。")
    (ROOT / "world" / "report_world.md").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
