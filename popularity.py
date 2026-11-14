"""撞号（热门度）模型：一组号码有多少人买？

依据：每期公布的各奖级中奖注数。某奖级的"实际中奖注数 ÷ 按销量算的随机期望注数"就是开出的那组号码的热门度。
模型：log(热门度) = 各号码的系数之和 + 几个形态特征（连号、等差、同尾、同区段、跨度），带岭惩罚。
      注数少的奖级用泊松回归，注数多的奖级对 log 比值做线性回归。
      销量口径、追加投注比例等缓慢变化用"此前 150 期的平均比值"做基线扣除（只用过去的数据）。

用法：
  python3 popularity.py backtest   前 80% 训练、后 20% 检验 + 打乱对照 + 硬性规则分组对比 → report_popularity.md
  python3 popularity.py freeze     用全部数据训练并冻结模型 → popularity_model.json（之后的开奖用于前瞻验证）
  python3 popularity.py track      对冻结之后的每期开奖，比较模型预测与实际中奖注数，累计证据 → popularity_log.csv
  python3 popularity.py checkb [彩种]   B 轨最近一期已开奖选号的命中情况
  python3 popularity.py pick [彩种]  得失对比 + B 轨（避开热门）5 注 → report_popularity_picks.md、picks_b/<彩种>/<期号>.json
  python3 popularity.py score 大乐透 01 02 03 05 07 + 04 12    某一注的预计热门度（1.0 = 平均）
"""
import csv
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "popularity_model.json"
LOG_FILE = ROOT / "popularity_log.csv"
BASE_WIN = 150       # 基线：此前多少期的平均比值
MIN_HIST = 30        # 基线至少需要的期数
NPERM = 200          # 打乱对照次数
LAMS = (1, 3, 10, 30, 100, 300, 1000)


def _h(K, M, k):
    return comb(M, k) * comb(K - M, M - k) / comb(K, M)


def tier_probs(game):
    if game == "ロト6":
        C = comb(43, 6)
        return {"1等": 1 / C, "3等": 6 * 36 / C, "4等": _h(43, 6, 4), "5等": _h(43, 6, 3)}
    if game == "ロト7":
        C = comb(37, 7)
        return {"1等": 1 / C, "3等": 7 * 28 / C, "4等": _h(37, 7, 5), "5等": _h(37, 7, 4)}
    if game == "ミニロト":
        C = comb(31, 5)
        return {"1等": 1 / C, "3等": 5 * 25 / C, "4等": _h(31, 5, 3)}
    if game == "双色球":
        C = comb(33, 6)
        r = lambda k: comb(6, k) * comb(27, 6 - k) / C
        return {"一等": r(6) / 16, "二等": r(6) * 15 / 16, "三等": r(5) / 16,
                "四等": r(5) * 15 / 16 + r(4) / 16, "五等": r(4) * 15 / 16 + r(3) / 16,
                "六等": (r(2) + r(1) + r(0)) / 16}
    if game == "大乐透":
        C = comb(35, 5)
        return {"一等": 1 / (C * 66), "二等": 20 / (C * 66)}
    raise KeyError(game)


GAMES = {"大乐透": (35, 5, 12), "双色球": (33, 6, 16), "ロト6": (43, 6, 0), "ロト7": (37, 7, 0), "ミニロト": (31, 5, 0)}
TOP = {"大乐透": "一等", "双色球": "一等", "ロト6": "1等", "ロト7": "1等", "ミニロト": "1等"}


def load(game):
    """返回 [{id, main, extra, tickets, n:{奖级:注数}, prize:头奖单注奖金}]，只含有销量数据的期"""
    out = []
    if game == "大乐透":
        for r in csv.DictReader((ROOT / "data" / "dlt_history.csv").open(encoding="utf-8")):
            if r["sales"] in ("", "0"):
                continue
            out.append({"id": r["issue"], "main": [int(r[f"r{i}"]) for i in range(1, 6)],
                        "extra": [int(r["b1"]), int(r["b2"])], "tickets": int(r["sales"]) / 2,
                        "n": {"一等": int(r["first_n"]), "二等": int(r["second_n"])}, "prize": int(r["first_prize"] or 0)})
    elif game == "双色球":
        names = ["一等", "二等", "三等", "四等", "五等", "六等"]
        for line in (ROOT / "data" / "other" / "ssq.txt").open(encoding="utf-8", errors="ignore"):
            f = line.split()
            if len(f) < 29 or f[15] == "0":
                continue
            out.append({"id": f[0], "main": [int(x) for x in f[2:8]], "extra": [int(f[8])], "tickets": int(f[15]) / 2,
                        "n": {nm: int(f[17 + 2 * i]) for i, nm in enumerate(names)}, "prize": int(f[18])})
    else:
        ld, M, price = {"ロト6": ("loto6", 6, 200), "ロト7": ("loto7", 7, 300), "ミニロト": ("miniloto", 5, 200)}[game]
        tiers = {"ロト6": {"1等": 1, "3等": 3, "4等": 4, "5等": 5}, "ロト7": {"1等": 1, "3等": 3, "4等": 4, "5等": 5},
                 "ミニロト": {"1等": 1, "3等": 3, "4等": 4}}[game]
        for r in csv.DictReader((ROOT / "japan" / "data" / f"{ld}.csv").open(encoding="utf-8")):
            if r["sales"] in ("", "0"):
                continue
            out.append({"id": r["id"], "main": [int(r[f"n{i}"]) for i in range(1, M + 1)], "extra": [],
                        "tickets": int(r["sales"]) / price, "n": {nm: int(r[f"t{k}"]) for nm, k in tiers.items()},
                        "prize": int(r["m1"] or 0)})
    return out


def pattern(nums, K):
    s = sorted(nums)
    d = np.diff(s)
    run = best = 1
    for x in d:
        run = run + 1 if x == 1 else 1
        best = max(best, run)
    return [float((d == 1).sum()), float(best >= 3), float(len(set(d.tolist())) == 1),
            float(sum(c * (c - 1) // 2 for c in Counter(x % 10 for x in s).values())),
            float(max(Counter((x - 1) // 10 for x in s).values())), (s[-1] - s[0]) / K,
            float(any(s[i + 3] - s[i] <= 6 for i in range(len(s) - 3))), float(s[0] > K // 2), float(s[-1] <= K // 2 + 1)]


def is_ap(nums):
    return len(set(np.diff(sorted(nums)).tolist())) == 1


PATTERN_NAMES = ["连号对数", "三连号以上", "等差数列", "同尾对数", "同一区段最多个数", "跨度", "4 个号挤在跨度 6 以内", "全是大号", "全是小号"]
CAP = 6.0   # 训练时把单期热门度截断在 6 倍以内：个别“图案号”（如ミニロト第621回 06 12 18 24 30，头奖 1129 注、期望 20 注）
            # 否则会把它含的几个号码的系数整体带偏。评估时不截断。


def feats(game, main, extra):
    K, M, E = GAMES[game]
    x = np.zeros(K + E)
    x[np.array(main) - 1] = 1
    for e in extra:
        x[K + e - 1] = 1
    p = pattern(main, K)
    if E and len(extra) == 2:
        p.append(float(abs(extra[0] - extra[1]) == 1))
    elif game == "大乐透":
        p.append(0.0)
    return np.concatenate([x, p])


def feat_names(game):
    K, M, E = GAMES[game]
    back = "蓝" if game == "双色球" else "后"
    return [f"{i:02d}" for i in range(1, K + 1)] + [f"{back}{i:02d}" for i in range(1, E + 1)] + PATTERN_NAMES + (
        ["后区连号"] if game == "大乐透" else [])


def offsets(n, e0):
    """基线：只用此前 BASE_WIN 期的 Σ实际/Σ期望；返回 log(期望注数)，历史不足的期为 nan"""
    off = np.full(len(n), np.nan)
    cn, ce = np.concatenate([[0], np.cumsum(n)]), np.concatenate([[0], np.cumsum(e0)])
    for t in range(MIN_HIST, len(n)):
        a = max(0, t - BASE_WIN)
        base = (cn[t] - cn[a] + 0.5) / (ce[t] - ce[a] + 1e-9)
        off[t] = np.log(e0[t] * base)
    return off


def fit(X, n, off, lam, poisson):
    """返回系数 b（b[0] 为截距）。X 已标准化。"""
    Xa = np.hstack([np.ones((len(X), 1)), X])
    R = np.eye(Xa.shape[1]) * lam
    R[0, 0] = 0
    if not poisson:
        y = np.log((n + 0.5) / np.exp(off))
        y = np.clip(y, np.median(y) - np.log(CAP), np.median(y) + np.log(CAP))
        return np.linalg.solve(Xa.T @ Xa + R, Xa.T @ y)
    n = np.minimum(n, CAP * np.exp(off) * n.sum() / np.exp(off).sum() + 3)
    b = np.zeros(Xa.shape[1])
    b[0] = np.log((n.sum() + 0.5) / np.exp(off).sum())
    for _ in range(60):
        mu = np.exp(np.clip(off + Xa @ b, -30, 30))
        step = np.linalg.solve((Xa * mu[:, None]).T @ Xa + R, Xa.T @ (n - mu) - R @ b)
        m = np.abs(step).max()
        b += step * min(1.0, 2.0 / (m + 1e-12))
        if m < 1e-7:
            break
    return b


def loss(b, X, n, off, poisson):
    eta = b[0] + X @ b[1:]
    if poisson:
        return float(-(n * (off + eta) - np.exp(off + eta)).mean())
    return float(((np.log((n + 0.5) / np.exp(off)) - eta) ** 2).mean())


# ロト6 / ロト7 的头奖每期不到 1 注，直接回归学不出来：借用 3等 模型的方向，只额外拟合一个放大系数
PROXY = {"ロト6": "3等", "ロト7": "3等"}
_cache = {}


def prepare(game, tier):
    if (game, tier) not in _cache:
        _cache[(game, tier)] = _prepare(game, tier)
    return _cache[(game, tier)]


def _prepare(game, tier):
    rows = load(game)
    p = tier_probs(game)[tier]
    n = np.array([r["n"][tier] for r in rows], float)
    e0 = np.array([r["tickets"] * p for r in rows])
    off = offsets(n, e0)
    X = np.array([feats(game, r["main"], r["extra"]) for r in rows])
    ok = ~np.isnan(off)
    return [r for r, k in zip(rows, ok) if k], X[ok], n[ok], off[ok]


def standardize(Xtr):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1
    return mu, sd


def choose_lam(X, n, off, poisson):
    cut = int(len(X) * 0.8)
    best = None
    for lam in LAMS:
        b = fit(X[:cut], n[:cut], off[:cut], lam, poisson)
        l = loss(b, X[cut:], n[cut:], off[cut:], poisson)
        if best is None or l < best[0]:
            best = (l, lam)
    return best[1]


def train(game, tier, cut=None, perm=None, lam=None):
    """在前 cut 期上训练；返回 (mu, sd, b, poisson, lam)。perm 用于打乱对照。"""
    src = PROXY.get(game) if tier == TOP[game] else None
    rows, X, n, off = prepare(game, src or tier)
    cut = cut or len(X)
    mu, sd = standardize(X[:cut])
    Z = ((X - mu) / sd)[:cut]
    poisson = bool(np.median(n) < 50)
    lam = lam or choose_lam(Z, n[:cut], off[:cut], poisson)
    if perm is not None:
        Z = Z[perm]
    b = fit(Z, n[:cut], off[:cut], lam, poisson)
    if src:
        _, _, n1, o1 = prepare(game, tier)
        bs = fit((Z @ b[1:])[:, None], n1[:cut], o1[:cut], 0, True)
        b, poisson = np.concatenate([[bs[0]], bs[1] * b[1:]]), True
    return mu, sd, b, poisson, lam


def hard_rule(game, main):
    """事先规定的硬性过滤：无三连号、非等差、（号码池大于 31 时）不全在 01–31、大乐透 01–07 不超过 3 个"""
    K = GAMES[game][0]
    if pattern(main, K)[1] or is_ap(main):
        return False
    if K > 31 and max(main) <= 31:
        return False
    if game == "大乐透" and sum(x <= 7 for x in main) > 3:
        return False
    return True


def rate_ratio(n, off, mask, rng, reps=1000):
    """两组的 Σ实际/Σ期望 之比（mask 组 ÷ 其余），按期自助法给 95% 区间"""
    e = np.exp(off)
    f = lambda idx: (n[idx][mask[idx]].sum() / e[idx][mask[idx]].sum()) / (n[idx][~mask[idx]].sum() / e[idx][~mask[idx]].sum())
    allidx = np.arange(len(n))
    bs = []
    for _ in range(reps):
        idx = rng.integers(0, len(n), len(n))
        if mask[idx].sum() > 2 and (~mask[idx]).sum() > 2 and n[idx][mask[idx]].sum() > 0 and n[idx][~mask[idx]].sum() > 0:
            bs.append(f(idx))
    return f(allidx), np.quantile(bs, 0.025), np.quantile(bs, 0.975)


def backtest():
    rng = np.random.default_rng(0)
    out = ["# 撞号（热门度）模型回测\n",
           "问题：一组号码有多少人买，能不能预测？能的话，避开热门组合，中奖后分到的钱更多（中奖概率不变）。\n",
           "方法：某奖级的“实际中奖注数 ÷ 按销量算的随机期望注数”就是开出号码的热门度。用前 80% 的期数训练，"
           f"在后 20% 上检验预测与实际的相关系数；再把训练集的号码与结果打乱重训 {NPERM} 次，得到“什么都没学到”时的相关系数分布作为及格线。\n",
           "## 1. 预测检验（后 20% 期数）\n",
           "| 彩种 | 奖级 | 检验期数 | 平均每期注数 | 相关系数 | 打乱对照 95% 分位 | p | 预测最冷门 20% 的实际热门度 | 预测最热门 20% 的实际热门度 |",
           "|---|---|---|---|---|---|---|---|---|"]
    rule_lines, coef_lines = [], []
    for game in GAMES:
        for tier in tier_probs(game):
            rows, X, n, off = prepare(game, tier)
            cut = int(len(X) * 0.8)
            mu, sd, b, poisson, lam = train(game, tier, cut)
            Z = (X - mu) / sd
            pred = Z[cut:] @ b[1:]
            act = np.log((n[cut:] + 0.5) / (np.exp(off[cut:]) + (0.5 if poisson else 0)))
            c = np.corrcoef(pred, act)[0, 1] if pred.std() > 0 else 0.0
            null = []
            for _ in range(NPERM):
                bp = train(game, tier, cut, rng.permutation(cut), lam)[2]
                pp = Z[cut:] @ bp[1:]
                null.append(np.corrcoef(pp, act)[0, 1] if pp.std() > 0 else 0.0)
            null = np.array(null)
            pv = (1 + (null >= c).sum()) / (NPERM + 1)
            order = np.argsort(pred)
            q = max(1, len(order) // 5)
            e = np.exp(off[cut:])
            lo = n[cut:][order[:q]].sum() / e[order[:q]].sum()
            hi = n[cut:][order[-q:]].sum() / e[order[-q:]].sum()
            mark = "**" if pv < 0.01 else ""
            label = tier + (f"（借用 {PROXY[game]} 模型）" if tier == TOP[game] and game in PROXY else "")
            out.append(f"| {game} | {label} | {len(pred)} | {n[cut:].mean():,.1f} | {mark}{c:+.3f}{mark} | {np.quantile(null, 0.95):+.3f} | "
                       f"{pv:.3f} | ×{lo:.2f} | ×{hi:.2f} |")
            # 硬性规则分组（全部期数；规则是事先规定的，不是拟合出来的）
            mask = np.array([hard_rule(game, r["main"]) for r in rows])
            if 10 < mask.sum() < len(mask) - 10:
                rr, a, bb = rate_ratio(n, off, mask, rng)
                rule_lines.append(f"| {game} | {tier} | {mask.sum()} / {(~mask).sum()} | ×{rr:.3f} | {a:.3f} – {bb:.3f} |")
            if tier == TOP[game]:
                mu2, sd2, b2 = train(game, tier)[:3]
                eff = b2[1:] / sd2          # 每个特征出现时 log 热门度的变化
                names = feat_names(game)
                K, M, E = GAMES[game]
                idx = np.argsort(-eff[:K])
                hot = "、".join(f"{names[i]}（×{np.exp(eff[i]):.2f}）" for i in idx[:6])
                cold = "、".join(f"{names[i]}（×{np.exp(eff[i]):.2f}）" for i in idx[::-1][:6])
                pat = "、".join(f"{names[i]} ×{np.exp(eff[i]):.2f}" for i in range(K + E, len(names)) if names[i] != "跨度")
                coef_lines.append(f"- **{game}（{tier}）** 最热门的号码：{hot}\n  - 最冷门的号码：{cold}\n  - 形态：{pat}")
            print(game, tier, f"corr={c:+.3f} null95={np.quantile(null, 0.95):+.3f} p={pv:.3f} lo={lo:.2f} hi={hi:.2f}", flush=True)
    out += ["\n相关系数加粗 = p<0.01。最后两列是直接证据：把检验期按模型预测排序，最冷门的 20% 和最热门的 20% 实际撞号有多少（×1.00 = 平均）。\n",
            "## 2. 硬性规则分组对比（全部期数）\n",
            "规则（事先规定）：无三连号、非等差数列、号码池大于 31 时不全落在 01–31、大乐透 01–07 不超过 3 个。"
            "下表是“通过规则的开奖号码”相对“通不过的”的撞号人数之比，小于 1 表示通过组撞号更少。\n",
            "| 彩种 | 奖级 | 通过 / 通不过（期数） | 撞号人数之比 | 95% 区间 |", "|---|---|---|---|---|", *rule_lines,
            "\n## 3. 头奖模型里各号码与形态的热门度（全部数据训练）\n",
            "括号里是“含这个号码时，撞号人数乘以多少”。\n", *coef_lines, ""]
    (ROOT / "report_popularity.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("→ report_popularity.md")


def freeze():
    model = {}
    for game in GAMES:
        model[game] = {"last_id": load(game)[-1]["id"], "tiers": {}}
        for tier in tier_probs(game):
            rows, X, n, off = prepare(game, tier)
            mu, sd, b, poisson, lam = train(game, tier)
            Z = (X - mu) / sd
            shrink = 1.0
            if tier == TOP[game]:
                # 校准：在后 20% 上看“预测的 log 热门度”要乘多少才与实际相符；小于 1 说明模型说得过头，按它收缩（不放大）
                cut = int(len(X) * 0.8)
                m2, s2, b2 = train(game, tier, cut)[:3]
                pr = ((X - m2) / s2)[cut:] @ b2[1:]
                shrink = float(np.clip(fit(pr[:, None], n[cut:], off[cut:], 0, True)[1], 0.0, 1.0))
                b = np.concatenate([[b[0]], shrink * b[1:]])
            resid = np.log((n + 0.5) / np.exp(off)) - (b[0] + Z @ b[1:])
            model[game]["tiers"][tier] = {"poisson": poisson, "lam": lam, "mu": mu.tolist(), "sd": sd.tolist(),
                                          "b": b.tolist(), "shrink": shrink, "sigma": float(resid.std()),
                                          "sigma0": float(np.log((n + 0.5) / np.exp(off)).std())}
    MODEL_FILE.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    print("已冻结 →", MODEL_FILE.name, {g: m["last_id"] for g, m in model.items()})
    print("头奖模型校准收缩系数：", {g: round(m["tiers"][TOP[g]]["shrink"], 2) for g, m in model.items()})


def popularity(game, main, extra=(), tier=None, model=None):
    """某一注的预计热门度（相对平均 = 1.0）"""
    model = model or json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    t = model[game]["tiers"][tier or TOP[game]]
    x = feats(game, list(main), list(extra))
    return float(np.exp(((x - np.array(t["mu"])) / np.array(t["sd"])) @ np.array(t["b"])[1:]))


def track():
    """前瞻验证：模型冻结之后的每期，比较“用模型”和“不用模型（只用基线）”对实际中奖注数的预测，累计对数似然差（比特）"""
    model = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    done = set()
    if LOG_FILE.exists():
        done = {(r["game"], r["id"], r["tier"]) for r in csv.DictReader(LOG_FILE.open(encoding="utf-8"))}
    new = []
    for game in GAMES:
        for tier, t in model[game]["tiers"].items():
            rows, X, n, off = prepare(game, tier)
            b, mu, sd = np.array(t["b"]), np.array(t["mu"]), np.array(t["sd"])
            for r, x, nn, o in zip(rows, X, n, off):
                if int(r["id"]) <= int(model[game]["last_id"]) or (game, r["id"], tier) in done:
                    continue
                eta = float(((x - mu) / sd) @ b[1:])
                if t["poisson"]:
                    # 不用模型：期望 = 基线；用模型：期望 = 基线 × 预计热门度（按训练集平均归一）
                    ll1 = nn * (o + b[0] + eta) - np.exp(o + b[0] + eta)
                    ll0 = nn * (o + b[0]) - np.exp(o + b[0])
                else:
                    y = np.log((nn + 0.5) / np.exp(o))
                    ll1 = -0.5 * ((y - b[0] - eta) / t["sigma"]) ** 2
                    ll0 = -0.5 * ((y - b[0]) / t["sigma"]) ** 2
                new.append({"game": game, "id": r["id"], "tier": tier, "actual": int(nn),
                            "expected_base": round(float(np.exp(o + b[0])), 2),
                            "expected_model": round(float(np.exp(o + b[0] + eta)), 2),
                            "bits": round(float((ll1 - ll0) / np.log(2)), 4)})
    if new:
        hdr = not LOG_FILE.exists()
        with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(new[0]))
            if hdr:
                w.writeheader()
            w.writerows(new)
    allr = list(csv.DictReader(LOG_FILE.open(encoding="utf-8"))) if LOG_FILE.exists() else []
    print(f"撞号模型前瞻验证（模型冻结后新增 {len({(r['game'], r['id']) for r in allr})} 期；本次新增 {len({(r['game'], r['id']) for r in new})} 期）")
    for game in GAMES:
        g = [r for r in allr if r["game"] == game]
        if not g:
            print(f"  {game}：冻结后尚无新开奖")
            continue
        bits = sum(float(r["bits"]) for r in g)
        top = [r for r in g if r["tier"] == TOP[game]]
        last = max((r["id"] for r in g), key=int)
        lt = [r for r in top if r["id"] == last]
        tail = f"；最新一期 {last} 头奖：实际 {lt[0]['actual']} 注，模型预计 {lt[0]['expected_model']}，不用模型预计 {lt[0]['expected_base']}" if lt else ""
        print(f"  {game}：{len({r['id'] for r in g})} 期，累计证据 {bits:+.2f} 比特（正 = 模型比不用模型预测得准）{tail}")
    tot = sum(float(r["bits"]) for r in allr)
    print(f"  合计 {tot:+.2f} 比特。达到 +10 比特可认为模型在新数据上确实有效；低于 −10 比特则模型失效。")


# ───────────── 得失对比与 B 轨选号 ─────────────
PICK_DIRS = {"大乐透": "picks", "双色球": "picks_ssq", "ロト6": "japan/picks/loto6", "ロト7": "japan/picks/loto7",
             "ミニロト": "japan/picks/miniloto"}
B_DIR = ROOT / "picks_b"
POOL = 16            # 候选号码个数（在其中穷举组合）


def prize_curve(game, recent=500):
    """头奖单注奖金 f(中奖总注数)：取最近 500 期里中奖注数相近（1.5 倍以内）的各期奖金中位数；数据不足处按 1/N 外推"""
    rows = [r for r in load(game)[-recent:] if r["n"][TOP[game]] > 0 and r["prize"] > 0]
    n = np.array([r["n"][TOP[game]] for r in rows], float)
    m = np.array([r["prize"] for r in rows], float)
    f = {}
    has = {N: np.abs(np.log(n / N)) <= np.log(1.5) + 1e-9 for N in range(1, 400)}
    first = min(N for N, k in has.items() if k.sum() >= 5)
    for N in range(first, 400):
        f[N] = float(np.median(m[has[N]])) if has[N].sum() >= 5 else f[N - 1] * (N - 1) / N
        if N > first:
            f[N] = min(f[N], f[N - 1])
    for N in range(first - 1, 0, -1):          # 注数很少的情形没有数据：按 1/N 外推，但不超过出现过的最高奖金
        f[N] = min(float(m.max()), f[first] * first / N)
    mu = float(np.mean([r["n"][TOP[game]] for r in load(game)[-recent:]]))
    return f, mu


def make_value(game):
    """V(r)：中头奖时的期望奖金，相对“热门度平均的一注”的倍数。其他中奖者人数 ~ 泊松(mu·r)"""
    f, mu = prize_curve(game)
    ks = np.arange(0, 399)
    fv = np.array([f[k + 1] for k in ks])

    def ev(r):
        lam = mu * r
        logp = ks * np.log(lam) - lam - np.cumsum(np.concatenate([[0], np.log(np.arange(1, len(ks)))]))
        return float(np.exp(logp) @ fv)
    base = ev(1.0)
    return (lambda r: ev(max(r, 1e-6)) / base), mu, f


def latest_a(game):
    files = sorted((ROOT / PICK_DIRS[game]).glob("*.json"))
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    out = []
    for t in d["tickets"]:
        if isinstance(t, dict):
            b = t["blue"]
            out.append((t["red"], b if isinstance(b, list) else [b]))
        else:
            out.append((t, []))
    return files[-1].stem, out


def pick(only=None, save=True):
    sys.path.insert(0, str(ROOT / "deep"))
    import bayes_pick
    import dynpl
    model = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    radar = json.loads((ROOT / "radar_status.json").read_text(encoding="utf-8"))
    out = ["# 得失对比：现行规则（A 轨）与避开热门的新规则（B 轨）\n",
           "每注三个数：**概率倍数**（贝叶斯模型，中头奖的概率是随机的几倍）、**撞号**（预计同时中头奖的人数是平均的几倍）、"
           "**奖金倍数**（按最近 500 期“中奖注数→单注奖金”的实际关系算，中头奖时期望拿到的奖金是平均的几倍）。"
           "**期望值 = 概率倍数 × 奖金倍数**，是两条规则比较的唯一标准。\n",
           f"B 轨规则：在贝叶斯分数与冷门程度综合最高的 {POOL} 个号码里穷举，取期望值最高的 5 注；任意两注至多共用 M−2 个号码；"
           "雷达热球只在证明档才强制加入；大乐透后区、双色球蓝球取最冷门的。"
           "为防止模型外推，不选三连号、等差数列，也不选预计热门度低于 98% 历史开奖号码的组合。\n"]
    summary = ["## 汇总\n", "| 彩种 | A 轨概率倍数 | A 轨奖金倍数 | **A 轨期望值** | B 轨概率倍数 | B 轨奖金倍数 | **B 轨期望值** | B ÷ A |", "|---|---|---|---|---|---|---|---|"]
    detail = []
    for game, (K, M, E) in GAMES.items():
        if only and game != only:
            continue
        rng = np.random.default_rng(0)
        V, mu_w, f = make_value(game)
        t = model[game]["tiers"][TOP[game]]
        eff = np.array(t["b"])[1:] / np.array(t["sd"])
        _, _, mix, _, _ = bayes_pick.posterior(game, rng)
        mbar = sum(w * m for w, m, _ in mix)
        flat = np.ptp(mbar) < 1e-9

        def mults(cands):
            if flat:
                return np.ones(len(cands))
            o = np.zeros(len(cands))
            for w, m, P in mix:
                o += w * dynpl.ticket_mult([[x - 1 for x in c] for c in cands], m, P, K, M, rng, n=200)
            return o

        pop = lambda main, extra=(): popularity(game, main, extra, model=model)
        # A 轨现状
        issue, a = latest_a(game)
        am = mults([x[0] for x in a])
        ar = np.array([pop(m_, e_) for m_, e_ in a])
        av = np.array([V(r) for r in ar])
        # B 轨：候选号码按 贝叶斯 log 倍数 − w·log 热门度 排序
        wslope = -(np.log(V(1.1)) - np.log(V(0.9))) / (np.log(1.1) - np.log(0.9))
        hot = radar.get(game, {}).get("alarms", [])[:1]
        score = mbar - wslope * eff[:K]
        order = [int(i) + 1 for i in np.argsort(-score) if int(i) + 1 not in hot][:POOL]
        import itertools
        cands = [sorted(hot + list(c)) for c in itertools.combinations(order, M - len(hot))]
        # 后区 / 蓝球：最冷门的 5 个（对）
        if E:
            exs = [[b] for b in range(1, E + 1)] if game == "双色球" else [list(c) for c in itertools.combinations(range(1, E + 1), 2)]
            base_main = cands[0]
            exs = sorted(exs, key=lambda e_: pop(base_main, e_))[:5]
        else:
            exs = [[]] * 5
        hist = np.array([pop(r_["main"], exs[0]) for r_ in load(game)])
        floor = float(np.quantile(hist, 0.02))     # 比 98% 的历史开奖号码还冷门的组合不要：模型没见过，不可信
        cr = np.array([pop(c, exs[0]) for c in cands])
        okc = np.array([r_ >= floor and not pattern(c, K)[1] and not is_ap(c) for c, r_ in zip(cands, cr)])
        approx = np.where(okc, np.array([mbar[np.array(c) - 1].sum() + np.log(V(r_)) for c, r_ in zip(cands, cr)]), -np.inf)
        chosen = []
        for i in np.argsort(-approx):
            if all(len(set(cands[i]) & set(c)) <= M - 2 for c in chosen):
                chosen.append(cands[i])
            if len(chosen) == 5:
                break
        bm = mults(chosen)
        br = np.array([pop(c, e_) for c, e_ in zip(chosen, exs)])
        bv = np.array([V(r) for r in br])
        ea, eb = float((am * av).mean()), float((bm * bv).mean())
        summary.append(f"| {game} | ×{am.mean():.3f} | ×{av.mean():.3f} | **×{ea:.3f}** | ×{bm.mean():.3f} | ×{bv.mean():.3f} | **×{eb:.3f}** | {eb / ea:.2f} |")
        fmt = lambda m_, e_: " ".join(f"{x:02d}" for x in m_) + (("  +  " + " ".join(f"{x:02d}" for x in e_)) if e_ else "")
        detail += [f"\n## {game}（第 {issue} 期/回）\n",
                   f"头奖平均每期 {mu_w:.1f} 注；单注奖金：独中 {f[1]:,.0f}，与 {max(1, round(mu_w))} 人同中 {f[max(1, round(mu_w)) + 1]:,.0f}，"
                   f"与 {max(2, round(3 * mu_w))} 人同中 {f[max(2, round(3 * mu_w)) + 1]:,.0f}。"
                   + ("  强制含证明档热球 %02d。" % hot[0] if hot else ""),
                   "\n| 轨 | 号码 | 概率倍数 | 撞号 | 奖金倍数 | 期望值 |", "|---|---|---|---|---|---|"]
        for (m_, e_), x, r, v in zip(a, am, ar, av):
            detail.append(f"| A | {fmt(m_, e_)} | ×{x:.3f} | ×{r:.2f} | ×{v:.3f} | ×{x * v:.3f} |")
        for m_, e_, x, r, v in zip(chosen, exs, bm, br, bv):
            detail.append(f"| B | {fmt(m_, e_)} | ×{x:.3f} | ×{r:.2f} | ×{v:.3f} | ×{x * v:.3f} |")
        if save:
            d = b_dir(game)
            d.mkdir(parents=True, exist_ok=True)
            fp = d / f"{issue}.json"
            last = load(game)[-1]["id"]
            if fp.exists():
                print(f"{game}：{fp.relative_to(ROOT)} 已存在，不覆盖")
            elif int(issue) <= int(last):
                print(f"{game}：第 {issue} 期已开奖，不生成")
            else:
                fp.write_text(json.dumps({"id": issue, "hot": hot, "tickets": [{"main": m_, "extra": e_, "mult": round(float(x), 4),
                                          "popularity": round(float(r), 3), "prize_factor": round(float(v), 3)}
                                          for m_, e_, x, r, v in zip(chosen, exs, bm, br, bv)],
                                          "a_value": round(ea, 4), "b_value": round(eb, 4)}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{game} 第{issue}：A 期望值 ×{ea:.3f}（概率 ×{am.mean():.3f}，奖金 ×{av.mean():.3f}）；"
              f"B 期望值 ×{eb:.3f}（概率 ×{bm.mean():.3f}，奖金 ×{bv.mean():.3f}）", flush=True)
        for m_, e_ in zip(chosen, exs):
            print("   B:", fmt(m_, e_))
    if not only:
        (ROOT / "report_popularity_picks.md").write_text("\n".join(out + summary + detail) + "\n", encoding="utf-8")
        print("→ report_popularity_picks.md")


def b_dir(game):
    return B_DIR / PICK_DIRS[game].split("/")[-1].replace("picks_ssq", "ssq").replace("picks", "dlt")


def checkb(only=None):
    """B 轨已开奖各期的命中情况"""
    for game in GAMES:
        if only and game != only:
            continue
        draws = {r["id"]: r for r in load(game)}
        files = sorted(b_dir(game).glob("*.json")) if b_dir(game).exists() else []
        done = [f for f in files if f.stem in draws]
        if not done:
            print(f"{game} B 轨：尚无已开奖的期")
            continue
        f = done[-1]
        d, r = json.loads(f.read_text(encoding="utf-8")), draws[f.stem]
        hits = ["%d" % len(set(t["main"]) & set(r["main"])) + (("+%d" % len(set(t["extra"]) & set(r["extra"]))) if r["extra"] else "")
                for t in d["tickets"]]
        print(f"{game} B 轨第 {f.stem} 期命中：" + "、".join(hits) + f"（已开奖 {len(done)} 期）")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    if cmd == "backtest":
        backtest()
    elif cmd == "freeze":
        freeze()
    elif cmd == "track":
        track()
    elif cmd == "pick":
        pick(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "checkb":
        checkb(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "score":
        game = sys.argv[2]
        args = sys.argv[3:]
        main = [int(x) for x in (args[:args.index("+")] if "+" in args else args)]
        extra = [int(x) for x in args[args.index("+") + 1:]] if "+" in args else []
        print(f"{game} {main} + {extra}：预计撞号人数 ×{popularity(game, main, extra):.2f}（平均 = 1.00）")
