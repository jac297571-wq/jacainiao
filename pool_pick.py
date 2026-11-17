"""收敛选号（大乐透）：先把号码池收窄到前区 NF 个、后区 NB 个，再从池内出 N 注。

收窄依据：前区每个号码的分数 = 贝叶斯 log 概率倍数 − 斜率 × log 热门度（撞号模型）；后区没有概率信号，只按冷门程度取。
出注：在池内穷举，按“概率倍数 × 中奖后的奖金倍数”排序，任意两注前区至多共用 3 个号，不选三连号、等差、过于冷门（模型没见过）的组合；
      --cap 时每个号码出现次数不超过平均值的 1.5 倍（让各注在池内摊开）。
用法：
  python3 pool_pick.py pick [NF] [NB] [N] [--cap]   生成下一期 → picks_pool/dlt/<期号>.json（已存在或已开奖不覆盖）；默认值见 DEFAULT
  python3 pool_pick.py compare                      几种池子大小的对比（不保存）
  python3 pool_pick.py check                        最近一期已开奖池内选号的命中
"""
import itertools
import json
import sys
from math import ceil, comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "deep"))
import bayes_pick  # noqa: E402
import dynpl  # noqa: E402
import popularity as P  # noqa: E402

GAME = "大乐透"
DIR = ROOT / "picks_pool" / "dlt"
DEFAULT = (15, 7, 10, True)      # 前区池、后区池、注数、是否均衡（每个号码出现次数不超过平均的 1.5 倍）


class Ctx:
    def __init__(self):
        self.K, self.M, self.E = P.GAMES[GAME]
        self.rng = np.random.default_rng(0)
        self.model = json.loads(P.MODEL_FILE.read_text(encoding="utf-8"))
        t = self.model[GAME]["tiers"][P.TOP[GAME]]
        self.eff = np.array(t["b"])[1:] / np.array(t["sd"])
        self.V = P.make_value(GAME)[0]
        self.mix = bayes_pick.posterior(GAME, self.rng)[2]
        self.mbar = sum(w * m for w, m, _ in self.mix)
        self.slope = -(np.log(self.V(1.1)) - np.log(self.V(0.9))) / (np.log(1.1) - np.log(0.9))
        self.pop = lambda m_, e_: P.popularity(GAME, m_, e_, model=self.model)
        self.hist = None

    def build(self, NF, NB, N, cap):
        K, M, E = self.K, self.M, self.E
        score = self.mbar - self.slope * self.eff[:K]
        front = sorted(int(i) + 1 for i in np.argsort(-score)[:NF])
        back = sorted(int(i) + 1 for i in np.argsort(self.eff[K:K + E])[:NB])
        pairs = sorted((list(c) for c in itertools.combinations(back, 2)), key=lambda e_: self.pop(front[:M], e_))
        if self.hist is None:
            self.hist = np.array([self.pop(r["main"], pairs[0]) for r in P.load(GAME)])
        floor = float(np.quantile(self.hist, 0.02))
        cands = [list(c) for c in itertools.combinations(front, M)]
        cr = np.array([self.pop(c, pairs[0]) for c in cands])
        ok = np.array([r >= floor and not P.pattern(c, K)[1] and not P.is_ap(c) for c, r in zip(cands, cr)])
        J = np.where(ok, np.array([self.mbar[np.array(c) - 1].sum() + np.log(self.V(r)) for c, r in zip(cands, cr)]), -np.inf)
        limit = ceil(1.5 * N * M / NF) if cap else N
        chosen, cnt = [], {x: 0 for x in front}
        for i in np.argsort(-J):
            if not np.isfinite(J[i]):
                break
            c = cands[i]
            if all(len(set(c) & set(o)) <= 3 for o in chosen) and all(cnt[x] < limit for x in c):
                chosen.append(c)
                for x in c:
                    cnt[x] += 1
            if len(chosen) == N:
                break
        mult = np.zeros(len(chosen))
        for w, m, Pm in self.mix:
            mult += w * dynpl.ticket_mult([[x - 1 for x in c] for c in chosen], m, Pm, K, M, self.rng, n=300)
        exs = (pairs * (N // len(pairs) + 1))[:len(chosen)]
        r = np.array([self.pop(c, e_) for c, e_ in zip(chosen, exs)])
        v = np.array([self.V(x) for x in r])
        return {"front": front, "back": back, "tickets": chosen, "extras": exs, "mult": mult, "pop": r, "value": v, "count": cnt,
                "p_pool": comb(NF, M) / comb(K, M) * comb(NB, 2) / comb(E, 2)}


fmt = lambda xs: " ".join(f"{a:02d}" for a in xs)


def show(res):
    print("前区池：" + fmt(res["front"]) + "    后区池：" + fmt(res["back"]))
    print(f"{len(res['tickets'])} 注（合计：概率 ×{res['mult'].mean():.3f}，奖金 ×{res['value'].mean():.3f}，期望值 ×{(res['mult'] * res['value']).mean():.3f}）")
    for k, (c, e_, x, rr, vv) in enumerate(zip(res["tickets"], res["extras"], res["mult"], res["pop"], res["value"]), 1):
        print(f"  {k:2d}. {fmt(c)}  +  {fmt(e_)}   概率 ×{x:.3f}  撞号 ×{rr:.2f}  期望值 ×{x * vv:.3f}")
    print("各号码出现次数：" + " ".join(f"{a:02d}×{n}" for a, n in res["count"].items()))


def pick(args):
    cap = "--cap" in args or DEFAULT[3]
    nums = [int(a) for a in args if a.isdigit()]
    NF, NB, N = (nums[i] if len(nums) > i else DEFAULT[i] for i in range(3))
    res = Ctx().build(NF, NB, N, cap)
    show(res)
    issue = P.latest_a(GAME)[0]
    DIR.mkdir(parents=True, exist_ok=True)
    fp = DIR / f"{issue}.json"
    if fp.exists():
        print(f"{fp.relative_to(ROOT)} 已存在，不覆盖")
    elif int(issue) <= int(P.load(GAME)[-1]["id"]):
        print(f"第 {issue} 期已开奖，不保存")
    else:
        fp.write_text(json.dumps({"id": issue, "front_pool": res["front"], "back_pool": res["back"], "cap": cap,
                                  "tickets": [{"main": c, "extra": e_, "mult": round(float(x), 4), "popularity": round(float(rr), 3)}
                                              for c, e_, x, rr in zip(res["tickets"], res["extras"], res["mult"], res["pop"])]},
                                 ensure_ascii=False, indent=1), encoding="utf-8")
        print("已保存", fp.relative_to(ROOT))


def compare():
    ctx = Ctx()
    print("| 前区池 / 后区池 | 均衡 | 概率 | 奖金 | 期望值 | 用到的前区号码 | 单个号码最多出现 | 开奖号码全落在池内的概率 |\n|---|---|---|---|---|---|---|---|")
    keep = {}
    for NF, NB, cap in ((12, 6, False), (12, 6, True), (15, 7, True), (18, 8, True), (21, 9, True)):
        r = ctx.build(NF, NB, 10, cap)
        keep[(NF, NB, cap)] = r
        used = sum(n > 0 for n in r["count"].values())
        print(f"| {NF} / {NB} | {'是' if cap else '否'} | ×{r['mult'].mean():.3f} | ×{r['value'].mean():.3f} | ×{(r['mult'] * r['value']).mean():.3f} | "
              f"{used}/{NF} | {max(r['count'].values())} 次 | {r['p_pool'] * 100:.3f}% |")
    return keep


def prize(f, b):
    """大乐透奖级（2019 年起的规则）"""
    table = {(5, 2): "一等", (5, 1): "二等", (5, 0): "三等", (4, 2): "四等", (4, 1): "五等", (3, 2): "六等", (4, 0): "七等",
             (3, 1): "八等", (2, 2): "八等", (3, 0): "九等", (1, 2): "九等", (2, 1): "九等", (0, 2): "九等"}
    return table.get((f, b), "")


def check():
    draws = {r["id"]: r for r in P.load(GAME)}
    done = [f for f in sorted(DIR.glob("*.json")) if f.stem in draws] if DIR.exists() else []
    if not done:
        print("收敛选号：尚无已开奖的期")
        return
    f = done[-1]
    d, r = json.loads(f.read_text(encoding="utf-8")), draws[f.stem]
    inf, inb = len(set(d["front_pool"]) & set(r["main"])), len(set(d["back_pool"]) & set(r["extra"]))
    print(f"收敛选号第 {f.stem} 期：开奖 {fmt(r['main'])} + {fmt(r['extra'])}；前区池（{len(d['front_pool'])} 码）含 {inf}/5 个，后区池（{len(d['back_pool'])} 码）含 {inb}/2 个")
    for k, t in enumerate(d["tickets"], 1):
        a, b = len(set(t["main"]) & set(r["main"])), len(set(t["extra"]) & set(r["extra"]))
        print(f"  {k:2d}. {fmt(t['main'])}  +  {fmt(t['extra'])}   前{a}+后{b}  {prize(a, b)}")
    print(f"（已开奖 {len(done)} 期）")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pick"
    if cmd == "compare":
        compare()
    elif cmd == "check":
        check()
    else:
        pick(sys.argv[2:] if cmd == "pick" else sys.argv[1:])
