"""向量化第二轮：两种新思路。输出 report_vec2.md。

A. 模拟训练的“热球探测网络”（不碰真实数据训练，所以不可能对真实数据过拟合）
   每个号码在每一期有一个"近况向量"：6 个不同半衰期（3–160 期）的指数加权出现次数减去期望，再加全历史偏差。
   同一个小网络作用于每个号码（号码之间共享权重，网络不认识号码本身，只认"近况"），输出下一期权重。
   训练数据全部是模拟的：纯随机开奖里随机注入热球（时间、长度 30–300 期、强度 ×1.3–2.5 随机），
   网络由此学到"什么样的近况意味着真的变热了"，相当于学出一个比 CUSUM 更优的贝叶斯探测器。
   然后原封不动地拿到真实历史上逐期前瞻打分。
B. 号码对向量（共现结构）：每个号码 i 有一个 r 维向量 u_i，组合概率 ∝ exp(Σ a_i + Σ_{i<j} u_i·u_j)，
   对全部 C(K,M) 个组合精确归一化。它能表达"某些号码爱一起出"——即使每个号码单独看都均匀，
   这种结构也会直接改变头奖组合的概率。严格滚动前瞻，验证集早停 + 温度校准。
   阳性对照：人工数据里注入 3 对"爱一起出"的号码。
"""
import itertools
import sys
import time
from math import comb, log
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import radar  # noqa: E402

torch.set_num_threads(4)
HL = np.array([3, 6, 12, 25, 50, 100, 160.0])
DEC = 0.5 ** (1 / HL)
F = len(HL) + 1
GAMES = {"大乐透": ("dlt", 35, 5), "双色球": ("ssq", 33, 6), "ロト6": ("loto6", 43, 6),
         "ロト7": ("loto7", 37, 7), "ミニロト": ("miniloto", 31, 5)}
out = ["# 向量化第二轮\n"]


def p(s=""):
    out.append(s)
    print(s, flush=True)


def perms_of(M):
    return torch.tensor(list(itertools.permutations(range(M))))


def pl_set_logp(logw, sets, perms):
    w = torch.exp(logw - logw.max(-1, keepdim=True).values)
    tot = w.sum(-1)
    ws = torch.gather(w, -1, sets)[..., perms]
    den = tot[..., None, None] - torch.cumsum(ws, -1) + ws
    return torch.logsumexp((torch.log(ws) - torch.log(den)).sum(-1), -1)


# ---------------- A ----------------
def features(X, K, M):
    """X: (..., T, K) 0/1。返回 (..., T, K, F)：第 t 期的特征只用到第 t 期及以前（用于预测 t+1）"""
    p0 = M / K
    T = X.shape[-2]
    E = np.zeros(X.shape[:-2] + (K, len(HL)))
    cum = np.zeros(X.shape[:-2] + (K,))
    feats = np.zeros(X.shape + (F,), np.float32)
    norm = np.sqrt(p0 * (1 - p0) / (1 - DEC ** 2))   # 指数加权和的平稳标准差
    for t in range(T):
        x = X[..., t, :]
        E = E * DEC + (x[..., None] - p0)
        cum += x - p0
        feats[..., t, :, :-1] = E / norm
        feats[..., t, :, -1] = cum / np.sqrt((t + 1) * p0 * (1 - p0))
    return feats


class Det(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(F, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)                     # 初始 = 均匀

    def forward(self, f):
        return self.f(f).squeeze(-1)


def simulate(K, M, B, T, rng):
    """B 条长 T 的模拟开奖；每条以 60% 概率含 1–2 段热球。返回 0/1 矩阵与组合"""
    logw = np.zeros((B, T, K))
    for b in range(B):
        if rng.random() < 0.6:
            for _ in range(rng.integers(1, 3)):
                i, L = rng.integers(K), rng.integers(30, 300)
                s = rng.integers(0, T - 30)
                logw[b, s:s + L, i] += np.log(rng.uniform(1.3, 2.5))
    g = logw + rng.gumbel(size=logw.shape)
    S = np.argsort(-g, -1)[..., :M]
    X = np.zeros((B, T, K), np.float32)
    np.put_along_axis(X, S, 1, -1)
    return X, S


def train_det(K, M, rng, steps=1500):
    net = Det()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    perms = perms_of(M)
    T = 500
    pool = []
    for _ in range(6):
        X, S = simulate(K, M, 128, T, rng)
        pool.append((torch.from_numpy(features(X, K, M)[:, 100:-1]), torch.from_numpy(S[:, 101:]).long()))
    for it in range(steps):
        f, s = pool[it % len(pool)]
        b = torch.randint(0, f.shape[0], (32,))
        t = torch.randint(0, f.shape[1], (32, 16))
        ff = f[b[:, None], t]
        ss = s[b[:, None], t]
        loss = -pl_set_logp(net(ff), ss, perms).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def score_det(net, X, S, K, M, start):
    perms = perms_of(M)
    f = torch.from_numpy(features(X, K, M))
    with torch.no_grad():
        lp = pl_set_logp(net(f[start - 1:-1]), torch.from_numpy(S[start:]).long(), perms).numpy()
    return (lp + log(comb(K, M))) / log(2)


def row(name, g):
    z = g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))
    p(f"| {name} | {len(g)} | {g.sum():+.2f} | ×{2 ** g.mean():.4f} | {z:+.2f} |")


def onehot(R, K):
    R = np.asarray(R)
    X = np.zeros((len(R), K), np.float32)
    X[np.arange(len(R))[:, None], R - 1] = 1
    return X, R - 1


def part_a(rng):
    p("## A. 模拟训练的热球探测网络（从未见过真实数据）\n")
    p("| 序列 | 打分期数 | 总增益（比特） | 头奖概率倍数 | z |")
    p("|---|---|---|---|---|")
    for name, (ld, K, M) in GAMES.items():
        net = train_det(K, M, rng)
        # 阳性对照：新的模拟数据（必有热球）与纯随机
        Xs, Ss = simulate(K, M, 6, 800, np.random.default_rng(99))
        g = np.concatenate([score_det(net, Xs[b], Ss[b], K, M, 100) for b in range(6)])
        row(f"{name}｜模拟·含热球（阳性对照）", g)
        rows = [r for _, r in radar.load(ld)]
        X, S = onehot(rows, K)
        if ld == "dlt":
            cut = next(i for i, (iss, _) in enumerate(radar.load("dlt")) if iss >= "14001")
            g = score_det(net, X, S, K, M, 100)
            row("大乐透 2007–2013（真实偏差，阳性对照）", g[:cut - 100])
            row("大乐透 2014–今", g[cut - 100:])
            row("大乐透 最近 300 期", g[-300:])
        else:
            g = score_det(net, X, S, K, M, 100)
            row(f"{name} 全历史", g)
            row(f"{name} 最近 300 期", g[-300:])
        # 当前探测结果
        with torch.no_grad():
            w = torch.softmax(net(torch.from_numpy(features(X, K, M)[-1])), -1).numpy() * K
        top = np.argsort(-w)[:3]
        p(f"|   当前权重最高：{'，'.join(f'{i + 1:02d}（×{w[i]:.2f}）' for i in top)} | | | | |")


# ---------------- B ----------------
class Pair(nn.Module):
    def __init__(self, K, r=3):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(K))
        self.u = nn.Parameter(0.01 * torch.randn(K, r))

    def energy(self, C, tau=1.0):
        J = self.u @ self.u.T
        e = self.a[C].sum(1)
        M = C.shape[1]
        for i in range(M):
            for j in range(i + 1, M):
                e = e + J[C[:, i], C[:, j]]
        return tau * e


def pair_logp(model, sets, allC, tau=1.0):
    return model.energy(sets, tau) - torch.logsumexp(model.energy(allC, tau), 0)


def fit_pair(K, sets, allC, lam=0.02):
    nv = max(60, len(sets) // 5)
    tr, va = sets[:-nv], sets[-nv:]
    m = Pair(K)
    opt = torch.optim.Adam(m.parameters(), 0.02)
    best, state, bad = -1e18, None, 0
    for it in range(300):
        loss = -pair_logp(m, tr, allC).mean() + lam * (m.a.pow(2).sum() + m.u.pow(2).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 10 == 9:
            with torch.no_grad():
                v = pair_logp(m, va, allC).mean().item()
            if v > best:
                best, state, bad = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 3:
                    break
    m.load_state_dict(state)
    with torch.no_grad():
        tau = max(np.linspace(0, 1, 11), key=lambda t: pair_logp(m, va, allC, float(t)).mean().item())
    return m, float(tau)


def walk_pair(R, K, M, start, block=300):
    allC = torch.tensor(list(itertools.combinations(range(K), M)), dtype=torch.long)
    S = torch.from_numpy(np.sort(np.asarray(R) - 1, 1)).long()
    base = -log(comb(K, M))
    g = []
    for s in range(start, len(R), block):
        m, tau = fit_pair(K, S[:s], allC)
        with torch.no_grad():
            g.extend(((pair_logp(m, S[s:s + block], allC, tau).numpy() - base) / log(2)).tolist())
    return np.array(g)


def synth_pairs(K=35, M=5, n=1500, seed=3):
    rng = np.random.default_rng(seed)
    allC = np.array(list(itertools.combinations(range(K), M)))
    e = np.zeros(len(allC))
    for a, b in ((2, 7), (11, 20), (25, 31)):            # 3 对“爱一起出”的号码
        e += 0.8 * ((allC == a).any(1) & (allC == b).any(1))
    pr = np.exp(e - e.max())
    pr /= pr.sum()
    return (allC[rng.choice(len(allC), n, p=pr)] + 1).tolist()


def part_b():
    p("\n## B. 号码对向量（共现结构，精确归一化）严格滚动前瞻\n")
    p("| 序列 | 打分期数 | 总增益（比特） | 头奖概率倍数 | z |")
    p("|---|---|---|---|---|")
    row("人工数据：3 对号码同出 ×2.2（阳性对照）", walk_pair(synth_pairs(), 35, 5, 500))
    dlt = radar.load("dlt")
    cut = next(i for i, (iss, _) in enumerate(dlt) if iss >= "14001")
    R = [r for _, r in dlt]
    row("大乐透 2007–2013（真实偏差，阳性对照）", walk_pair(R[:cut], 35, 5, 300))
    row("大乐透 2014–今", walk_pair(R[cut:], 35, 5, 300))
    for name in ("双色球", "ミニロト", "ロト6", "ロト7"):
        ld, K, M = GAMES[name]
        rows = [r for _, r in radar.load(ld)]
        row(name, walk_pair(rows, K, M, 300 if len(rows) > 1000 else 300, block=300 if K <= 35 else 600))


if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(0)
    part_a(np.random.default_rng(0))
    part_b()
    p(f"\n用时 {time.time() - t0:.0f} 秒（CPU）。")
    (ROOT / "report_vec2.md").write_text("\n".join(out) + "\n", encoding="utf-8")
