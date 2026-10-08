"""cifulucky：每期 5 注单式选号器。

思路：
  1. 台阶信号——新摇奖机（2019-09-07 起）下 01-11 号被抽中的倾向偏高——每次用最新出球顺序
     重新拟合 Plackett-Luce 台阶权重并做似然比检验。显著（z>2）时前区 5 个号全部从 01-11 中选：
     开奖若真随机，所有组合概率相同，集中在 01-11 没有任何损失；信号若真实，这里是概率最高的区域。
     信号消失则自动退回 35 选 5 均匀随机。
  1b. 连号信号——同期出现连号略多于理论（疑似混合不充分）——显著时每注至少含 1 对连号。
  2. 可选（--anti-split）：过滤“好看”的热门组合，不影响中奖率，只影响中了后被平分的程度。
  3. 5 注之间前区最多重 2 个号、后区互不相同，覆盖面最大。
  4. 选号写入 picks/，开奖后用 check 对奖，长期记录真实表现。

用法：
  python3 cifulucky.py              # 生成下一期 5 注
  python3 cifulucky.py --update     # 先抓取最新开奖再生成
  python3 cifulucky.py check        # 核对所有历史选号
  python3 cifulucky.py sprt         # 01-11 信号的序贯检验：累积证据，到阈值自动判真/判假
"""
import argparse
import csv
import json
import secrets
from datetime import date
from itertools import combinations, permutations
from math import comb, exp, log, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "dlt_history.csv"
ORDER_DATA = ROOT / "data" / "dlt_ballset.csv"
STEP = 11  # 台阶：01-11
PICKS = ROOT / "picks"
TICKETS = 5
MACHINE_SINCE = "19104"  # 新“维纳斯”摇奖机 2019-09-07 启用后的首期
SHRINK = 0.5  # 对数权重向 0 收缩一半，防止过度相信
rng = secrets.SystemRandom()


def load():
    rows = list(csv.DictReader(DATA.open(encoding="utf-8")))
    return [(r["issue"], [int(r[f"r{i}"]) for i in range(1, 6)], [int(r["b1"]), int(r["b2"])]) for r in rows]


def load_orders():
    """新摇奖机时期每期前区出球顺序"""
    rows = csv.DictReader(ORDER_DATA.open(encoding="utf-8"))
    return [list(map(int, r["order"].split()))[:5] for r in rows if r["issue"] >= MACHINE_SINCE and r["order"]]


def pl_loglik(lw, orders):
    """Plackett-Luce 对数似然：01-STEP 号权重 e^lw，其余为 1"""
    w = [exp(lw) if x <= STEP else 1.0 for x in range(36)]
    total0 = sum(w[1:])
    ll = 0.0
    for o in orders:
        tot = total0
        for x in o:
            ll += log(w[x] / tot)
            tot -= w[x]
    return ll


def step_signal():
    """拟合台阶权重（黄金分割搜索），似然比检验。返回 (原始权重, 收缩后权重, z, 期数)"""
    orders = load_orders()
    a, b = -0.5, 0.5
    g = (sqrt(5) - 1) / 2
    for _ in range(40):
        c, d = b - g * (b - a), a + g * (b - a)
        if pl_loglik(c, orders) > pl_loglik(d, orders):
            b = d
        else:
            a = c
    lw = (a + b) / 2
    lr = 2 * (pl_loglik(lw, orders) - pl_loglik(0.0, orders))
    z = sqrt(max(lr, 0)) * (1 if lw > 0 else -1)
    return exp(lw), exp(lw * SHRINK), z, len(orders)


def ticket_prob_ratio(r, w):
    """台阶权重 w 下，这一注前区被开出的概率 / 均匀随机下的概率（对 5! 种出球顺序精确求和）"""
    weights = [w if x <= STEP else 1.0 for x in range(36)]
    total0 = sum(weights[1:])
    p = 0.0
    for perm in permutations(r):
        q, tot = 1.0, total0
        for x in perm:
            q *= weights[x] / tot
            tot -= weights[x]
        p += q
    return p * comb(35, 5)


def consec_signal(draws):
    """全部历史中“同期有连号”的比例 vs 理论（混合不充分假设）。返回 (是否启用, z)"""
    from math import comb
    th = 1 - comb(31, 5) / comb(35, 5)
    k = sum(any(b - a == 1 for a, b in zip(sorted(r), sorted(r)[1:])) for _, r, _ in draws)
    z = (k / len(draws) - th) / sqrt(th * (1 - th) / len(draws))
    return z > 2, z


def max_run(r):
    best = cur = 1
    for a, b in zip(r, r[1:]):
        cur = cur + 1 if b - a == 1 else 1
        best = max(best, cur)
    return best


def max_ap(r):
    s, best = set(r), 2
    for a, b in combinations(r, 2):
        n, x = 2, 2 * b - a
        while x in s:
            n, x = n + 1, x + (b - a)
        best = max(best, n)
    return best


def popular(r, last):
    """热门（易被平分）的组合"""
    return (
        max_run(r) >= 3
        or sum(b - a == 1 for a, b in zip(r, r[1:])) >= 2
        or max_ap(r) >= 4
        or len({x % 2 for x in r}) == 1
        or len(set(r) & set(last)) >= 3
        or all(x % 5 == 0 or x % 7 == 0 for x in r)
    )


def generate(draws, anti_split=False):
    w_raw, w, z, n = step_signal()
    use_step = z > 2
    pool = list(range(1, STEP + 1)) if use_step else list(range(1, 36))
    need_consec, cz = consec_signal(draws)
    last = draws[-1][1]
    fronts, backs = [], []
    max_overlap, tries = 2, 0
    while len(fronts) < TICKETS:
        tries += 1
        if tries % 20000 == 0:  # 号码池小时放宽注间重叠限制
            max_overlap += 1
        r = sorted(rng.sample(pool, 5))
        if need_consec and max_run(r) < 2:
            continue
        if (anti_split and popular(r, last)) or r in fronts or any(len(set(r) & set(f)) > max_overlap for f in fronts):
            continue
        fronts.append(r)
    while len(backs) < TICKETS:
        b = sorted(rng.sample(range(1, 13), 2))
        if b not in backs and not (anti_split and b[1] - b[0] == 1):  # 后区连号也是热门
            backs.append(b)
    info = {"step_w_raw": w_raw, "step_w": w, "step_z": z, "step_n": n, "use_step": use_step,
            "consec": need_consec, "consec_z": cz, "max_overlap": max_overlap}
    return list(zip(fronts, backs)), info


def next_issue(issue):
    y, n = int(issue[:2]), int(issue[2:])
    if int(f"20{y:02d}") < date.today().year:
        return f"{date.today().year % 100:02d}001"
    return f"{y:02d}{n + 1:03d}"


def fmt(r, b):
    return " ".join(f"{x:02d}" for x in r) + "  +  " + " ".join(f"{x:02d}" for x in b)


def cmd_pick(args):
    if args.update:
        import importlib.util
        for name in ("fetch", "fetch_ballset"):
            spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
    draws = load()
    issue = next_issue(draws[-1][0])
    tickets, info = generate(draws, args.anti_split)
    print(f"第 {issue} 期（数据截至 {draws[-1][0]}）")
    print(f"台阶信号（01-{STEP:02d}，新机 {info['step_n']} 期出球顺序）：权重 ×{info['step_w_raw']:.3f}，z={info['step_z']:+.2f}，"
          + (f"启用，收缩后 ×{info['step_w']:.3f}，前区全部取自 01-{STEP:02d}" if info["use_step"] else "不显著，35 选 5 均匀随机"))
    print(f"连号信号：z={info['consec_z']:+.2f}，" + ("每注至少含 1 对连号" if info["consec"] else "不显著，不限制"))
    print()
    for i, (r, b) in enumerate(tickets, 1):
        up = ticket_prob_ratio(r, info["step_w"])
        up_raw = ticket_prob_ratio(r, info["step_w_raw"])
        print(f"  {i}. {fmt(r, b)}    模型下头奖概率 ×{up:.2f}（收缩）/ ×{up_raw:.2f}（不收缩）")
    PICKS.mkdir(exist_ok=True)
    f = PICKS / f"{issue}.json"
    if f.exists() and not args.force:
        print(f"\n{f.name} 已存在，未覆盖（--force 可覆盖）")
        return
    f.write_text(json.dumps({"issue": issue, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()},
                             "tickets": [{"red": r, "blue": b} for r, b in tickets]}, ensure_ascii=False, indent=1))
    print(f"\n已保存 picks/{f.name}")


def cmd_check(_):
    draws = {i: (r, b) for i, r, b in load()}
    total = {}
    for f in sorted(PICKS.glob("*.json")):
        p = json.loads(f.read_text())
        if p["issue"] not in draws:
            print(f"{p['issue']}: 未开奖")
            continue
        R, B = draws[p["issue"]]
        print(f"{p['issue']}: 开奖 {fmt(R, B)}")
        for t in p["tickets"]:
            hr, hb = len(set(t["red"]) & set(R)), len(set(t["blue"]) & set(B))
            total[(hr, hb)] = total.get((hr, hb), 0) + 1
            mark = "  ★" if (hr, hb) == (5, 2) else ""
            print(f"    {fmt(t['red'], t['blue'])}   前{hr}+后{hb}{mark}")
    if total:
        print("\n累计命中分布（前区+后区）：")
        for k in sorted(total, reverse=True):
            print(f"  {k[0]}+{k[1]}: {total[k]} 注")


SPRT_START = "26115"  # 前瞻起点：此前数据已用于提出假设，不计入裁决
SPRT_W1 = 1.10  # 备择假设：01-11 权重 ×1.10；零假设：×1.00
SPRT_A, SPRT_B = log(19), -log(19)  # 两类错误率各约 5%


def cmd_sprt(_):
    """序贯概率比检验：逐期累积对数似然比，越过上界判“信号为真”，越过下界判“信号为假”"""
    rows = [r for r in csv.DictReader(ORDER_DATA.open(encoding="utf-8")) if r["issue"] >= SPRT_START and r["order"]]
    lw1 = log(SPRT_W1)
    L = 0.0
    print(f"SPRT：H0 01-{STEP:02d} 权重 ×1.00  vs  H1 ×{SPRT_W1:.2f}；起点 {SPRT_START}；上界 {SPRT_A:+.2f}，下界 {SPRT_B:+.2f}")
    print("（按模拟，平均需要约 400–650 期、即约 3–4 年才能下结论）\n")
    for r in rows:
        o = list(map(int, r["order"].split()))[:5]
        step = pl_loglik(lw1, [o]) - pl_loglik(0.0, [o])
        L += step
        print(f"  {r['issue']}  前区出球 {' '.join(f'{x:02d}' for x in o)}  本期 {step:+.3f}  累计 {L:+.3f}")
        if L >= SPRT_A or L <= SPRT_B:
            break
    if not rows:
        print("  尚无前瞻数据（先 --update）")
    verdict = "✅ 判定：信号为真" if L >= SPRT_A else ("✗ 判定：信号为假（只是巧合）" if L <= SPRT_B else "… 尚未下结论，继续观察")
    print(f"\n累计 {L:+.3f}，已观察 {len(rows)} 期。{verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="cifulucky：每期 5 注选号器")
    ap.add_argument("cmd", nargs="?", default="pick", choices=["pick", "check", "sprt"])
    ap.add_argument("--update", action="store_true", help="先抓取最新开奖数据")
    ap.add_argument("--anti-split", action="store_true", help="过滤热门组合，减少中奖后被平分")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的本期选号")
    a = ap.parse_args()
    {"pick": cmd_pick, "check": cmd_check, "sprt": cmd_sprt}[a.cmd](a)
