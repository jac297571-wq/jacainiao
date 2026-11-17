"""收敛选号：先把号码池收窄到前区 NF 个、后区 NB 个，再从池内出 N 注。

收窄依据：前区每个号码的分数 = 贝叶斯 log 概率倍数 − 斜率 × log 热门度（撞号模型）；后区没有概率信号，只按冷门程度取。
出注：在池内穷举，按“概率倍数 × 中奖后的奖金倍数”排序，任意两注前区至多共用 3 个号，不选三连号、等差、过于冷门（模型没见过）的组合。
用法：python3 pool_pick.py [NF=12] [NB=6] [N=10]   （大乐透；结果存 picks_pool/dlt/<期号>.json，已存在不覆盖）
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "deep"))
import bayes_pick  # noqa: E402
import dynpl  # noqa: E402
import popularity as P  # noqa: E402

GAME = "大乐透"


def main():
    NF, NB, N = (int(sys.argv[i]) if len(sys.argv) > i else d for i, d in ((1, 12), (2, 6), (3, 10)))
    K, M, E = P.GAMES[GAME]
    rng = np.random.default_rng(0)
    model = json.loads(P.MODEL_FILE.read_text(encoding="utf-8"))
    t = model[GAME]["tiers"][P.TOP[GAME]]
    eff = np.array(t["b"])[1:] / np.array(t["sd"])
    V, mu_w, _ = P.make_value(GAME)
    _, _, mix, _, _ = bayes_pick.posterior(GAME, rng)
    mbar = sum(w * m for w, m, _ in mix)
    slope = -(np.log(V(1.1)) - np.log(V(0.9))) / (np.log(1.1) - np.log(0.9))
    score = mbar - slope * eff[:K]
    front = sorted(int(i) + 1 for i in np.argsort(-score)[:NF])
    back = sorted(int(i) + 1 for i in np.argsort(eff[K:K + E])[:NB])
    pop = lambda m_, e_: P.popularity(GAME, m_, e_, model=model)
    print(f"前区 {NF} 码：" + " ".join(f"{x:02d}" for x in front))
    for x in front:
        print(f"   {x:02d}  贝叶斯 ×{np.exp(mbar[x - 1]):.3f}  热门度 ×{np.exp(eff[x - 1]):.2f}")
    print(f"后区 {NB} 码：" + " ".join(f"{x:02d}" for x in back) + "   热门度 " + " ".join(f"×{np.exp(eff[K + x - 1]):.2f}" for x in back))
    pairs = sorted((list(c) for c in itertools.combinations(back, 2)), key=lambda e_: pop(front[:M], e_))
    hist = np.array([pop(r["main"], pairs[0]) for r in P.load(GAME)])
    floor = float(np.quantile(hist, 0.02))
    cands = [list(c) for c in itertools.combinations(front, M)]
    cr = np.array([pop(c, pairs[0]) for c in cands])
    ok = np.array([r >= floor and not P.pattern(c, K)[1] and not P.is_ap(c) for c, r in zip(cands, cr)])
    J = np.where(ok, np.array([mbar[np.array(c) - 1].sum() + np.log(V(r)) for c, r in zip(cands, cr)]), -np.inf)
    chosen = []
    for i in np.argsort(-J):
        if not np.isfinite(J[i]):
            break
        if all(len(set(cands[i]) & set(c)) <= 3 for c in chosen):
            chosen.append(cands[i])
        if len(chosen) == N:
            break
    mult = np.zeros(len(chosen))
    for w, m, Pm in mix:
        mult += w * dynpl.ticket_mult([[x - 1 for x in c] for c in chosen], m, Pm, K, M, rng, n=300)
    exs = (pairs * (N // len(pairs) + 1))[:len(chosen)]
    r = np.array([pop(c, e_) for c, e_ in zip(chosen, exs)])
    v = np.array([V(x) for x in r])
    print(f"\n{len(chosen)} 注（合计：概率 ×{mult.mean():.3f}，奖金 ×{v.mean():.3f}，期望值 ×{(mult * v).mean():.3f}）")
    for k, (c, e_, x, rr, vv) in enumerate(zip(chosen, exs, mult, r, v), 1):
        print(f"  {k:2d}. " + " ".join(f"{a:02d}" for a in c) + "  +  " + " ".join(f"{a:02d}" for a in e_)
              + f"   概率 ×{x:.3f}  撞号 ×{rr:.2f}  奖金 ×{vv:.3f}  期望值 ×{x * vv:.3f}")
    used = sorted({a for c in chosen for a in c})
    print("用到的前区号码 %d/%d；每个号码出现次数：" % (len(used), NF) + " ".join(f"{a:02d}×{sum(a in c for c in chosen)}" for a in front))
    issue = P.latest_a(GAME)[0]
    d = ROOT / "picks_pool" / "dlt"
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{issue}.json"
    if fp.exists():
        print(f"{fp.relative_to(ROOT)} 已存在，不覆盖")
    elif int(issue) <= int(P.load(GAME)[-1]["id"]):
        print(f"第 {issue} 期已开奖，不保存")
    else:
        fp.write_text(json.dumps({"id": issue, "front_pool": front, "back_pool": back,
                                  "tickets": [{"main": c, "extra": e_, "mult": round(float(x), 4), "popularity": round(float(rr), 3)}
                                              for c, e_, x, rr in zip(chosen, exs, mult, r)]}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("已保存", fp.relative_to(ROOT))


if __name__ == "__main__":
    main()
