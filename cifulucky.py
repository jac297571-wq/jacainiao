"""cifulucky：每期 5 注单式选号器。

思路：
  1. 唯一的统计信号——近年前区 1-12 号偏多——每次用最新数据重新检验；
     显著（z>2）才按收缩后的比例加权，信号消失自动退回均匀随机。
  2. 可选（--anti-split）：过滤“好看”的热门组合，不影响中奖率，只影响中了后被平分的程度。
  3. 5 注之间前区最多重 2 个号、后区互不相同，覆盖面最大。
  4. 选号写入 picks/，开奖后用 check 对奖，长期记录真实表现。

用法：
  python3 cifulucky.py              # 生成下一期 5 注
  python3 cifulucky.py --update     # 先抓取最新开奖再生成
  python3 cifulucky.py check        # 核对所有历史选号
"""
import argparse
import csv
import json
import secrets
from datetime import date
from itertools import combinations
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "dlt_history.csv"
PICKS = ROOT / "picks"
TICKETS = 5
WINDOW = 500  # 用最近多少期估计小号偏移
SHRINK = 0.5  # 向均匀收缩一半，防止过度押注
rng = secrets.SystemRandom()


def load():
    rows = list(csv.DictReader(DATA.open(encoding="utf-8")))
    return [(r["issue"], [int(r[f"r{i}"]) for i in range(1, 6)], [int(r["b1"]), int(r["b2"])]) for r in rows]


def small_weight(draws):
    """返回 (1-12 号的权重倍数, z 值)。z<=2 时权重为 1（均匀）"""
    recent = draws[-WINDOW:]
    s = [sum(x <= 12 for x in r) for _, r, _ in recent]
    mean, th = sum(s) / len(s), 5 * 12 / 35
    var = 5 * (12 / 35) * (23 / 35) * (30 / 34)
    z = (mean - th) / sqrt(var / len(s))
    if z <= 2:
        return 1.0, z
    return 1 + (mean / th - 1) * SHRINK, z


def weighted_sample(w_small, k=5):
    pool = list(range(1, 36))
    out = []
    for _ in range(k):
        weights = [w_small if x <= 12 else 1.0 for x in pool]
        x = rng.choices(pool, weights)[0]
        out.append(x)
        pool.remove(x)
    return sorted(out)


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
    w, z = small_weight(draws)
    last = draws[-1][1]
    fronts, backs = [], []
    while len(fronts) < TICKETS:
        r = weighted_sample(w)
        if (anti_split and popular(r, last)) or any(len(set(r) & set(f)) > 2 for f in fronts):
            continue
        fronts.append(r)
    while len(backs) < TICKETS:
        b = sorted(rng.sample(range(1, 13), 2))
        if b not in backs and not (anti_split and b[1] - b[0] == 1):  # 后区连号也是热门
            backs.append(b)
    return list(zip(fronts, backs)), w, z


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
        spec = importlib.util.spec_from_file_location("fetch", ROOT / "scripts" / "fetch.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
    draws = load()
    issue = next_issue(draws[-1][0])
    tickets, w, z = generate(draws, args.anti_split)
    print(f"第 {issue} 期（数据截至 {draws[-1][0]}）")
    sig = f"z={z:+.2f}，1-12 号权重 ×{w:.3f}" if w > 1 else f"z={z:+.2f}，信号不显著，均匀随机"
    print(f"小号信号：{sig}\n")
    for i, (r, b) in enumerate(tickets, 1):
        print(f"  {i}. {fmt(r, b)}")
    PICKS.mkdir(exist_ok=True)
    f = PICKS / f"{issue}.json"
    if f.exists() and not args.force:
        print(f"\n{f.name} 已存在，未覆盖（--force 可覆盖）")
        return
    f.write_text(json.dumps({"issue": issue, "small_z": round(z, 3), "small_w": round(w, 4),
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="cifulucky：每期 5 注选号器")
    ap.add_argument("cmd", nargs="?", default="pick", choices=["pick", "check"])
    ap.add_argument("--update", action="store_true", help="先抓取最新开奖数据")
    ap.add_argument("--anti-split", action="store_true", help="过滤热门组合，减少中奖后被平分")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的本期选号")
    a = ap.parse_args()
    (cmd_pick if a.cmd == "pick" else cmd_check)(a)
