"""ロト7 分套号码权重模型的置换检验：打乱套号标签重做 5 折 CV，看真实增益 +3.9 比特在零分布里的位置。"""
import importlib.util, sys
from pathlib import Path

import numpy as np

sys.argv = ["x", "loto7"]
spec = importlib.util.spec_from_file_location("la", Path(__file__).with_name("loto_analysis.py"))
# 只取函数，不重跑整份报告：复制必要部分
import csv
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
D = list(csv.DictReader((ROOT / "data" / "loto7.csv").open(encoding="utf-8")))
K, M = 37, 7
O = np.array([[int(d[f"o{i}"]) for i in range(1, M + 1)] for d in D]) - 1
SETS = np.array(["ABCDEFGHIJ".index(d["setball"]) for d in D])
N = len(D)


def nll(th, Ob, lab):
    W = np.exp(th.reshape(10, K))[lab]
    n = len(Ob)
    avail = np.ones((n, K))
    v = 0.0
    G = np.zeros((n, K))
    for k in range(M):
        den = (W * avail).sum(1)
        pk = Ob[:, k]
        v -= (np.log(W[np.arange(n), pk]) - np.log(den)).sum()
        G += W * avail / den[:, None]
        G[np.arange(n), pk] -= 1
        avail[np.arange(n), pk] = 0
    return v, np.vstack([G[lab == s].sum(0) for s in range(10)]).ravel()


def cv(lab, lam=100):
    ed = np.linspace(0, N, 6).astype(int)
    tot = 0.0
    for f in range(5):
        te = np.arange(ed[f], ed[f + 1])
        tr = np.setdiff1d(np.arange(N), te)

        def fun(t):
            v, g = nll(t, O[tr], lab[tr])
            return v + lam * t @ t / 2, g + lam * t
        th = minimize(fun, np.zeros(10 * K), jac=True, method="L-BFGS-B").x
        tot += (nll(np.zeros(10 * K), O[te], lab[te])[0] - nll(th, O[te], lab[te])[0]) / np.log(2)
    return tot


rng = np.random.default_rng(7)
obs = cv(SETS)
null = []
for i in range(200):
    null.append(cv(rng.permutation(SETS)))
    if (i + 1) % 20 == 0:
        print(i + 1, flush=True)
null = np.array(null)
p = (np.sum(null >= obs) + 1) / (len(null) + 1)
text = (f"# ロト7 分套模型置换检验\n\n真实套号下 5 折 CV 增益 {obs:+.2f} 比特；打乱套号 200 次的零分布：均值 {null.mean():+.2f}，"
        f"95% 分位 {np.percentile(null, 95):+.2f}，最大 {null.max():+.2f}。\n\n**置换 p = {p:.3f}**\n")
(ROOT / "report_loto7_perm.md").write_text(text, encoding="utf-8")
print(text)
