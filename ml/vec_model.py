"""向量化神经网络：把每个号码学成一个向量，用 Transformer 读最近 W 期开奖，预测下一期每个号码的权重。
对大乐透、双色球、ロト6、ロト7、ミニロト分别做严格的滚动前瞻（walk-forward）：只用过去训练，预测下一段，再滚动。

打分：在 Plackett-Luce 无放回抽球模型下，算出实际开出那组号码的精确概率（对全部出球顺序求和），
与均匀随机比较，得到“比特增益”。平均每期增益 g 比特 ⇔ 实际开奖组合的概率被放大了 2^g 倍，
这正是“按模型买到头奖”的概率倍数。

对照：
  阳性对照 1：大乐透 2007–2013（已知绿球物理偏差，×1.4 出现率）——模型必须能学到正增益，否则说明方法没用。
  阳性对照 2：人工数据，纯随机开奖里注入一段 150 期的“热球”（出现率翻倍），检验模型能否抓住阶段性热球。
用法：python3 ml/vec_model.py → 输出 report_vec.md
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
W = 60          # 输入窗口：最近 60 期
BLOCK = 200     # 每次预测 200 期，然后从头重新训练
D = 16
TAUS = np.linspace(0, 1, 11)  # 温度：0 = 退回均匀随机


class Net(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.emb = nn.Linear(K, D, bias=False)            # 多热编码 × 矩阵 = 该期号码向量之和
        self.pos = nn.Parameter(torch.zeros(W, D))
        layer = nn.TransformerEncoderLayer(D, 4, 64, dropout=0.3, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.out = nn.Linear(D, K)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)                     # 初始 = 均匀，模型只有学到东西才会偏离

    def forward(self, x):
        h = self.enc(self.emb(x) + self.pos)
        return self.out(h.mean(1))                         # K 个对数权重


def windows(X, idx):
    return torch.stack([torch.from_numpy(X[i - W:i]) for i in idx]).float()


def pl_set_logp(logw, sets, perms):
    """Plackett-Luce 下“开出这组号码（不计顺序）”的精确对数概率。logw: (B,K)；sets: (B,M)"""
    w = torch.exp(logw - logw.max(1, keepdim=True).values)
    tot = w.sum(1)                                         # (B,)
    ws = torch.gather(w, 1, sets)[:, perms]                # (B,P,M)
    den = tot[:, None, None] - torch.cumsum(ws, 2) + ws    # 每一步剩余权重
    lp = (torch.log(ws) - torch.log(den)).sum(2)           # (B,P)
    return torch.logsumexp(lp, 1)


def evaluate(net, X, S, perms, idx, tau=1.0):
    net.eval()
    with torch.no_grad():
        return pl_set_logp(tau * net(windows(X, idx)), torch.from_numpy(S[idx]).long(), perms).numpy()


def train(K, X, S, perms, idx, max_epochs=40, patience=4, lr=2e-3):
    """从头训练；最后 20% 作验证集做早停，再在验证集上选温度 τ（τ=0 即均匀）。返回 (net, τ)"""
    nv = max(60, len(idx) // 5)
    tr, va = idx[:-nv], idx[-nv:]
    net = Net(K)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-2)
    xs, ys = windows(X, tr), torch.from_numpy(S[tr]).long()
    best, best_state, bad = -1e18, None, 0
    for _ in range(max_epochs):
        net.train()
        for b in torch.randperm(len(tr)).split(64):
            loss = -pl_set_logp(net(xs[b]), ys[b], perms).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = evaluate(net, X, S, perms, va).mean()
        if v > best:
            best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    tau = max(TAUS, key=lambda t: evaluate(net, X, S, perms, va, t).mean())
    return net, float(tau)


def walk_forward(R, K, M, start, seed=0):
    torch.manual_seed(seed)
    R = np.asarray(R)
    X = np.zeros((len(R), K), np.float32)
    X[np.arange(len(R))[:, None], R - 1] = 1
    S = R - 1
    perms = torch.tensor(list(itertools.permutations(range(M))))
    base = -log(comb(K, M))
    gains, taus = [], []
    for s in range(start, len(R), BLOCK):
        net, tau = train(K, X, S, perms, np.arange(W, s))
        taus.append(tau)
        te = np.arange(s, min(s + BLOCK, len(R)))
        gains.extend((evaluate(net, X, S, perms, te, tau) - base) / log(2))
    print(f"   温度 τ：{taus}", flush=True)
    return np.array(gains)


def summarize(name, g, out):
    z = g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))
    line = f"| {name} | {len(g)} | {g.sum():+.2f} | {g.mean():+.4f} | ×{2 ** g.mean():.3f} | {z:+.2f} |"
    out.append(line)
    print(line, flush=True)


def synthetic(K=35, M=5, n=1200, seed=5):
    rng = np.random.default_rng(seed)
    R = []
    for t in range(n):
        w = np.ones(K)
        if 700 <= t < 850:
            w[16] = 2.2                                     # 17 号在第 700–849 期出现率约翻倍
        R.append(sorted(rng.choice(K, M, replace=False, p=w / w.sum()) + 1))
    return R


def main():
    out = ["# 向量化神经网络（号码向量 + Transformer）滚动前瞻\n",
           f"每个号码学成 {D} 维向量；模型读最近 {W} 期，输出下一期各号码权重；每 {BLOCK} 期用全部过去数据从头训练一次（最后 20% 作验证集早停，并在验证集上选温度 τ，τ=0 表示模型认为没东西可学、退回均匀随机）。"
           "“倍数”= 实际开出组合在模型下的概率 / 均匀随机概率（几何平均），即按模型选号时头奖概率的平均放大倍数。"
           "z 检验增益是否显著大于 0。\n",
           "| 序列 | 预测期数 | 总增益（比特） | 每期平均 | 头奖概率倍数 | z |",
           "|---|---|---|---|---|---|"]
    print("\n".join(out), flush=True)
    t0 = time.time()
    dlt = radar.load("dlt")
    cut = next(i for i, (iss, _) in enumerate(dlt) if iss >= "14001")
    R = [r for _, r in dlt]
    series = [
        ("人工数据：随机 + 第700–849期热球 ×2（阳性对照）", synthetic(), 35, 5, 300),
        ("大乐透 2007–2013（已知绿球偏差，阳性对照）", R[:cut], 35, 5, 300),
        ("大乐透 2014–今", R[cut - W:], 35, 5, W + 300),
        ("双色球", [r for _, r in radar.load("ssq")], 33, 6, 300),
        ("ロト6", [r for _, r in radar.load("loto6")], 43, 6, 300),
        ("ロト7", [r for _, r in radar.load("loto7")], 37, 7, 200),
        ("ミニロト", [r for _, r in radar.load("miniloto")], 31, 5, 300),
    ]
    for name, Rs, K, M, start in series:
        g = walk_forward(Rs, K, M, start)
        summarize(name, g, out)
        if name.startswith("人工"):
            hot = g[700 - start:850 - start]
            out.append(f"|   其中热球段 700–849 |  {len(hot)} | {hot.sum():+.2f} | {hot.mean():+.4f} | ×{2 ** hot.mean():.3f} | |")
    out.append(f"\n用时 {time.time() - t0:.0f} 秒（CPU）。")
    (ROOT / "report_vec.md").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
