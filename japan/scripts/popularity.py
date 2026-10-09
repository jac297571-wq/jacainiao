"""ロト6 买家偏好（人気数字）反推与“冷门号码”期望回报。输出 japan/report_popularity.md。

原理：若买家随机选号，第 k 等中奖注数期望 E_k = (销售额/200) × P_k。
实际/期望 ρ_k = t_k / E_k 衡量“开出的号码组合有多受欢迎”。3 等以下注数多（几千到十几万），ρ 很精确。
模型：log ρ5 = a + Σ_{开出号码} β_i + 形态特征 → β_i 即号码 i 的人气。用 5 折 CV 检验可预测性。
1–4 等为奖池均分：同样中奖，开出号码越冷门，同中的人越少，单注奖金越高。
"""
import csv
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
K, M = 43, 6
TOT = comb(K, M)
P = {1: 1 / TOT, 2: 6 / TOT, 3: 216 / TOT, 4: 9990 / TOT, 5: 155400 / TOT}
D = [d for d in csv.DictReader((ROOT / "data" / "loto6.csv").open(encoding="utf-8")) if float(d["sales"] or 0) > 0]
rng = np.random.default_rng(1)
out = ["# ロト6：买家偏好与冷门号码的期望回报\n"]


def feats(nums):
    s = sorted(nums)
    f = np.zeros(K + 6)
    for x in s:
        f[x - 1] = 1
    f[K] = sum(x <= 31 for x in s)  # 生日号（日期 1-31）
    f[K + 1] = sum(b - a == 1 for a, b in zip(s, s[1:]))  # 连号对数
    f[K + 2] = sum(x % 10 == 7 for x in s)  # 尾数 7（“幸运 7”）
    f[K + 3] = len({x // 10 for x in s})  # 跨十位段数
    f[K + 4] = sum(x <= 12 for x in s)  # 月份 1-12
    f[K + 5] = int(any(c - b == b - a for a, b, c in combinations(s, 3)))  # 含等差三项
    return f


X, Y5, Y4, Y3, S, info = [], [], [], [], [], []
for d in D:
    nums = [int(d[f"n{i}"]) for i in range(1, 7)]
    n_tk = float(d["sales"]) / 200
    t5, t4, t3 = float(d["t5"]), float(d["t4"]), float(d["t3"])
    if t5 <= 0 or t4 <= 0:
        continue
    X.append(feats(nums))
    Y5.append(np.log(t5 / (n_tk * P[5])))
    Y4.append(np.log(t4 / (n_tk * P[4])))
    Y3.append(np.log(max(t3, 0.5) / (n_tk * P[3])))
    info.append(d)
X, Y5, Y4, Y3 = map(np.array, (X, Y5, Y4, Y3))
N = len(X)


def ridge(Xm, y, lam):
    A = np.hstack([np.ones((len(Xm), 1)), Xm])
    I = np.eye(A.shape[1])
    I[0, 0] = 0
    return np.linalg.solve(A.T @ A + lam * I, A.T @ y)


def pred(w, Xm):
    return w[0] + Xm @ w[1:]


out.append(f"有销售额且有 4、5 等中奖的 {N} 回。ρ5（3 个号码中）平均 {np.exp(Y5).mean():.3f}，ρ4 平均 {np.exp(Y4).mean():.3f}。\n")
out.append("## A. 开出号码的“人气”能否由号码本身预测？（5 折 CV 的 R²）\n")
out.append("| 目标 | 正则 λ | CV R² |")
out.append("|---|---|---|")
ed = np.linspace(0, N, 6).astype(int)
best = {}
for name, y in (("log ρ5（3 等以下：5 等）", Y5), ("log ρ4（4 等）", Y4), ("log ρ3（3 等）", Y3)):
    for lam in (1, 10, 100):
        res = np.zeros(N)
        for f in range(5):
            te = np.arange(ed[f], ed[f + 1])
            tr = np.setdiff1d(np.arange(N), te)
            res[te] = y[te] - pred(ridge(X[tr], y[tr], lam), X[te])
        r2 = 1 - res.var() / y.var()
        out.append(f"| {name} | {lam} | {r2:.3f} |")
        if name.startswith("log ρ5") and (best.get("r2", -9) < r2):
            best = {"r2": r2, "lam": lam}

w5 = ridge(X, Y5, best["lam"])
w4 = ridge(X, Y4, best["lam"])
beta = w5[1:K + 1]
beta -= beta.mean()
order = np.argsort(beta)
out.append("\n## B. 各号码人气（对 log ρ5 的贡献，越大越热门）\n")
out.append("最冷门 10 个：" + "，".join(f"{i + 1:02d}({beta[i]:+.3f})" for i in order[:10]))
out.append("\n最热门 10 个：" + "，".join(f"{i + 1:02d}({beta[i]:+.3f})" for i in order[::-1][:10]))
names = ["生日号(≤31)个数", "连号对数", "尾数7个数", "跨十位段数", "月份号(≤12)个数", "含等差三项"]
out.append("\n形态特征系数（log ρ5）：" + "；".join(f"{n} {w5[1 + K + j]:+.3f}" for j, n in enumerate(names)))

# ---------- C. 期望回报模拟 ----------
out.append("\n## C. 冷门号码策略的期望回报\n")
out.append("对每种选号策略，模拟大量“中奖情形”：给定自己的号码中了 k 个，按模型预测开出组合的 ρ，"
           "1–4 等单注奖金按 “该等奖金池 / 同中注数” 计算（同中注数 = 期望 × ρ；1–2 等用泊松分布模拟同中人数）。"
           "奖金池取近 5 年平均：各等奖金总额占销售额的比例。\n")
recent = [d for d in D if d["date"] >= "2021-10-01"]
sales = np.array([float(d["sales"]) for d in recent])
share = {k: np.mean([float(d[f"t{k}"]) * float(d[f"m{k}"]) / float(d["sales"]) for d in recent]) for k in (2, 3, 4)}
# 1 等奖池：已派出 + 滚存变化，近似为 1 等实际派奖 + キャリーオーバー净增
co = np.array([float(d["co"]) for d in recent])
paid1 = np.array([float(d["t1"]) * float(d["m1"]) for d in recent])
share[1] = (paid1.sum() + co[-1] - 0) / sales.sum()
n_tk = sales.mean() / 200
out.append("近 5 年各等奖金池占销售额：" + "，".join(f"{k} 等 {share[k]:.3f}" for k in (1, 2, 3, 4)) + f"；5 等固定 1000 日元；每回约 {n_tk / 1e6:.1f} 百万注。\n")


def strategy_ev(pick_fn, n_sims=4000, carry=0.0, n_tk=n_tk):
    """返回每 200 日元的期望回报。carry：额外滚入的头奖奖金（日元）。"""
    tot = 0.0
    for _ in range(n_sims):
        S = pick_fn()
        fS = feats(S)
        others = [x for x in range(1, K + 1) if x not in S]
        ev = 1000 * P[5]
        for k, tier in ((6, 1), (5, 4.5), (4, 4)):
            # 构造一个与 S 恰好重合 k 个的开奖
            keep = list(rng.choice(S, k, replace=False))
            draw = keep + list(rng.choice(others, M - k, replace=False))
            fx = feats(draw)
            if k == 4:
                rho = np.exp(pred(w4, fx[None])[0])
                ev += P[4] * share[4] * 200 * n_tk / (n_tk * P[4] * rho)
            elif k == 5:
                rho = np.exp(pred(w4, fx[None])[0])  # 用 ρ4 模型近似 5 码组合的人气
                ev += P[3] * share[3] * 200 * n_tk / (n_tk * P[3] * rho + 1)
                ev += P[2] * share[2] * 200 * n_tk / (n_tk * P[2] * rho + 1)
            else:
                rho = np.exp(1.5 * pred(w4, fx[None])[0])  # 6 码组合人气更集中，外推
                lam = n_tk * P[1] * rho
                W = rng.poisson(lam)
                pool = share[1] * 200 * n_tk + carry
                ev += P[1] * min(pool / (1 + W), 6e8 if carry > 0 else 2e8)
        tot += ev
    return tot / n_sims / 200


def rand_pick():
    return sorted(rng.choice(np.arange(1, K + 1), M, replace=False).tolist())


cold_pool = (order[:20] + 1).tolist()  # 最冷门 20 个号码


def cold_pick():
    while True:
        s = sorted(rng.choice(cold_pool, M, replace=False).tolist())
        f = feats(s)
        if f[K + 1] == 0 and f[K + 5] == 0:  # 无连号、无等差
            return s


hot_pool = (order[::-1][:20] + 1).tolist()


def hot_pick():
    return sorted(rng.choice(hot_pool, M, replace=False).tolist())


def cold_high_pick():
    """冷门 + 尽量多用 32-43（非生日号）"""
    pool = [x for x in cold_pool if x >= 32] + [x for x in cold_pool if x < 32]
    while True:
        s = sorted(rng.choice(pool[:14], M, replace=False).tolist())
        if feats(s)[K + 1] == 0:
            return s


out.append("| 策略 | 无滚存 | 滚存 5 亿日元 | 滚存 10 亿日元 |")
out.append("|---|---|---|---|")
for name, fn in (("随机机选", rand_pick), ("热门 20 号", hot_pick), ("冷门 20 号（无连号/等差）", cold_pick),
                 ("冷门 + 偏重 32–43", cold_high_pick)):
    row = [strategy_ev(fn, carry=c) for c in (0, 5e8, 10e8)]
    out.append(f"| {name} | {row[0]:.3f} | {row[1]:.3f} | {row[2]:.3f} |")
out.append("\n（数值为每投入 1 日元的期望回报；模型对 5、6 码组合的人气做了外推，1–2 等的数值只能当量级参考。）")
big_n = 2266e6 / 200  # 第1589回：滚入 18.85 亿日元，销售 22.66 亿日元
out.append(f"\n**历史最极端情形**（第1589回：滚入 18.85 亿日元，销售 22.66 亿日元，约 {big_n / 1e6:.1f} 百万注）：")
for name, fn in (("随机机选", rand_pick), ("冷门 + 偏重 32–43", cold_high_pick)):
    out.append(f"- {name}：{strategy_ev(fn, carry=18.85e8, n_tk=big_n):.3f}")

text = "\n".join(out) + "\n"
(ROOT / "report_popularity.md").write_text(text, encoding="utf-8")
print(text)
