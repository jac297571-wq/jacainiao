"""机器学习兜底：把能想到的公开信息全喂给模型，逐期滚动（只用过去）预测下一期。输出 report_ml.md。

目标：每期每个号码是否开出。打分：伯努利对数损失（比特），相对均匀（前区 5/35，后区 2/12）的增益。
滚动：从 2018 年起，每 100 期为一块；每块用该块之前的全部数据重新训练。
特征（对第 t 期、号码 i）：号码本身；近 10/30/100/300 期出现频率；遗漏期数；上期、上上期是否出现；
     上期 01-11 个数；星期；上期奖池、上期销量（对数）；距新机启用的期数；
     “先知”变体额外加入本期摇奖球套号（购彩时并不知道，只用来看套号信息有没有价值）。
"""
import csv
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
H = list(csv.DictReader((ROOT / "data" / "dlt_history.csv").open(encoding="utf-8")))
SETMAP = {r["issue"]: int(r["ballset"]) for r in csv.DictReader((ROOT / "data" / "dlt_ballset.csv").open(encoding="utf-8"))}
N = len(H)
DATE = np.array([h["date"] for h in H])
FRONT = np.array([[int(h[f"r{i}"]) for i in range(1, 6)] for h in H])
BACK = np.array([[int(h["b1"]), int(h["b2"])] for h in H])
POOL = np.array([float(h["pool"] or 0) for h in H])
SALES = np.array([float(h["sales"] or 0) for h in H])
SETS = np.array([SETMAP.get(h["issue"], 0) for h in H])
WD = np.array([np.datetime64(d).astype("datetime64[D]").astype(int) % 7 for d in DATE])  # 1970-01-01 周四
NEWIDX = int(np.argmax(DATE >= "2019-09-07"))


def build(draws, K):
    """返回 X(N,K,F)、Y(N,K)、特征名"""
    Y = np.zeros((N, K))
    Y[np.arange(N)[:, None], draws - 1] = 1
    cs = np.vstack([np.zeros(K), np.cumsum(Y, 0)])
    feats, names = [], []

    def add(name, arr):
        feats.append(arr)
        names.append(name)

    add("号码", np.tile(np.arange(1, K + 1), (N, 1)).astype(float))
    for w in (10, 30, 100, 300):
        lo = np.maximum(np.arange(N) - w, 0)
        cnt = cs[np.arange(N)] - cs[lo]
        add(f"近{w}期频率", cnt / np.maximum(np.arange(N) - lo, 1)[:, None])
    gap = np.zeros((N, K))
    last = np.full(K, -1)
    for t in range(N):
        gap[t] = np.where(last >= 0, t - last, 100)
        last[Y[t] > 0] = t
    add("遗漏期数", np.minimum(gap, 100))
    add("上期出现", np.vstack([np.zeros(K), Y[:-1]]))
    add("上上期出现", np.vstack([np.zeros((2, K)), Y[:-2]]))
    small_prev = np.concatenate([[0], (FRONT[:-1] <= 11).sum(1)])
    add("上期01-11个数", np.tile(small_prev[:, None], (1, K)).astype(float))
    add("星期", np.tile(WD[:, None], (1, K)).astype(float))
    add("上期奖池", np.tile(np.log1p(np.concatenate([[0], POOL[:-1]]))[:, None], (1, K)))
    add("上期销量", np.tile(np.log1p(np.concatenate([[0], SALES[:-1]]))[:, None], (1, K)))
    add("距新机期数", np.tile((np.arange(N) - NEWIDX)[:, None], (1, K)).astype(float))
    add("本期套号(先知)", np.tile(SETS[:, None], (1, K)).astype(float))
    return np.stack(feats, -1), Y, names


def bits(q, y, base):
    q = np.clip(q, 1e-4, 1 - 1e-4)
    return (y * np.log2(q) + (1 - y) * np.log2(1 - q)).sum(-1) - (y * np.log2(base) + (1 - y) * np.log2(1 - base)).sum(-1)


def normalize(p, m):
    return np.clip(p * m / p.sum(1, keepdims=True), 1e-4, 1 - 1e-4)


def run(zone, window=None):
    draws, K, M = (FRONT, 35, 5) if zone == "前区" else (BACK, 12, 2)
    X, Y, names = build(draws, K)
    base = M / K
    start = int(np.argmax(DATE >= "2018-01-01"))
    blocks = list(range(start, N, 100))
    nf = len(names)
    oracle_col = names.index("本期套号(先知)")
    no_oracle = [j for j in range(nf) if j != oracle_col]
    gains = {}
    importances = {}

    def model_preds(kind, lo, hi):
        tr = slice(300 if window is None else max(300, lo - window), lo)
        Xtr, Ytr = X[tr], Y[tr]
        Xte = X[lo:hi]
        if kind == "01-11台阶":
            if zone != "前区":
                return None
            w = Ytr[:, :11].mean() / Ytr[:, 11:].mean()
            p = np.where(np.arange(K) < 11, w, 1.0)[None].repeat(hi - lo, 0)
            return normalize(p, M)
        cols = no_oracle if "先知" not in kind else list(range(nf))
        if kind.startswith("逻辑回归"):
            num = Xtr[..., 0].astype(int).ravel()
            oh = np.eye(K)[num - 1]
            other = Xtr[..., cols[1:]].reshape(-1, len(cols) - 1)
            mu, sd = other.mean(0), other.std(0) + 1e-9
            clf = LogisticRegression(C=0.05, max_iter=500)
            clf.fit(np.hstack([oh, (other - mu) / sd]), Ytr.ravel())
            numte = Xte[..., 0].astype(int).ravel()
            ote = (Xte[..., cols[1:]].reshape(-1, len(cols) - 1) - mu) / sd
            p = clf.predict_proba(np.hstack([np.eye(K)[numte - 1], ote]))[:, 1].reshape(hi - lo, K)
            return normalize(p, M)
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=300, l2_regularization=1.0,
                                             categorical_features=[0], random_state=0)
        clf.fit(Xtr[..., cols].reshape(-1, len(cols)), Ytr.ravel())
        p = clf.predict_proba(Xte[..., cols].reshape(-1, len(cols)))[:, 1].reshape(hi - lo, K)
        if kind == "梯度提升树":
            # 置换重要性：打乱某个特征后增益下降多少
            yte = Y[lo:hi]
            g0 = bits(normalize(p, M), yte, base).sum()
            rng = np.random.default_rng(lo)
            for j, c in enumerate(cols):
                Xp = Xte[..., cols].copy()
                Xp[..., j] = rng.permutation(Xp[..., j].ravel()).reshape(Xp[..., j].shape)
                pp = clf.predict_proba(Xp.reshape(-1, len(cols)))[:, 1].reshape(hi - lo, K)
                importances[names[c]] = importances.get(names[c], 0) + g0 - bits(normalize(pp, M), yte, base).sum()
        return normalize(p, M)

    kinds = ["01-11台阶", "逻辑回归", "梯度提升树", "梯度提升树+先知套号"]
    for kind in kinds:
        g = []
        for lo in blocks:
            hi = min(lo + 100, N)
            q = model_preds(kind, lo, hi)
            if q is None:
                break
            g.append(bits(q, Y[lo:hi], base))
        if g:
            gains[kind] = np.concatenate(g)
        print(zone, window, kind, "done", flush=True)
    return gains, importances, start


out = ["# 机器学习兜底：全部公开信息能否预测下一期？\n",
       "逐期滚动、只用过去数据训练；评估期 2018-01 起。增益单位为比特，相对均匀随机；> 0 才代表比随机更会猜。",
       "训练窗口：全部历史（扩展窗口）或只用最近 1000 / 500 期（滚动窗口，适应机制切换）。\n"]
for zone, window in (("前区", None), ("前区", 1000), ("前区", 500), ("后区", None), ("后区", 500)):
    gains, imp, start = run(zone, window)
    newmask = np.arange(start, N)[: len(next(iter(gains.values())))] >= NEWIDX
    out.append(f"## {zone} · 训练窗口：{'全部历史' if window is None else f'最近 {window} 期'}\n")
    out.append("| 模型 | 2018 起总增益 | z | 新机时期总增益 | z | 相当于每注头奖概率 |")
    out.append("|---|---|---|---|---|---|")
    K, M = (35, 5) if zone == "前区" else (12, 2)
    for kind, g in gains.items():
        z = g.sum() / (g.std(ddof=1) * np.sqrt(len(g)))
        gn = g[newmask]
        zn = gn.sum() / (gn.std(ddof=1) * np.sqrt(len(gn)))
        out.append(f"| {kind} | {g.sum():+.2f} | {z:+.2f} | {gn.sum():+.2f} | {zn:+.2f} | ×{2 ** g.mean():.4f} |")
    if imp:
        out.append("\n梯度提升树的置换重要性（打乱该特征后损失的比特数，正 = 模型确实依赖它且有用）：\n")
        out.append("| 特征 | 重要性（比特） |")
        out.append("|---|---|")
        for k, v in sorted(imp.items(), key=lambda x: -x[1]):
            out.append(f"| {k} | {v:+.2f} |")
    out.append("")

text = "\n".join(out) + "\n"
(ROOT / "report_ml.md").write_text(text, encoding="utf-8")
print(text)
