"""日本ロト热球选号器 + 前瞻序贯检验。

依据（japan/scripts/hot_hunt.py、japan/scripts/hot_calib.py → japan/report_hot_calib.md）：
用“对随机数据执行同样扫描流程”的正确校准，两个当前活跃的热球通过检验：
  ロト7   22：自第 658 回起 41 回出现 46.3%（理论 18.9%），校正 p≈0.010
  ミニロト 31：自第 1363 回起 45 回出现 42.2%（理论 16.1%），校正 p≈0.008
（4 个彩种扫描的多重比较校正后约 0.03–0.04。）
每注都包含热球：若开奖真随机，没有任何代价；若偏差为真，则有优势。由前瞻 SPRT 裁决。

用法：
  python3 japan/jp_lucky.py [loto7|miniloto] [--update] [--force]   生成下一回 5 注
  python3 japan/jp_lucky.py [loto7|miniloto] check                  对奖
  python3 japan/jp_lucky.py [loto7|miniloto] sprt                   热球前瞻检验
"""
import argparse
import csv
import importlib.util
import json
import secrets
from math import log
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = {
    "loto7": {"title": "ロト7", "K": 37, "M": 7, "hot": 22, "since": 658, "p1": 0.30, "start": 699},
    "miniloto": {"title": "ミニロト", "K": 31, "M": 5, "hot": 31, "since": 1363, "p1": 0.28, "start": 1408},
}
rng = secrets.SystemRandom()


def load(game):
    M = CFG[game]["M"]
    return [(int(d["id"]), [int(d[f"n{i}"]) for i in range(1, M + 1)], d["date"])
            for d in csv.DictReader((ROOT / "data" / f"{game}.csv").open(encoding="utf-8"))]


def cmd_pick(game, a):
    c = CFG[game]
    if a.update:
        spec = importlib.util.spec_from_file_location("fl", ROOT / "scripts" / "fetch_loto.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.fetch(game)
    rows = load(game)
    K, M, hot = c["K"], c["M"], c["hot"]
    after = [r for r in rows if r[0] >= c["since"]]
    k = sum(hot in r[1] for r in after)
    issue = rows[-1][0] + 1
    print(f"{c['title']} 第{issue}回（数据截至 第{rows[-1][0]}回 {rows[-1][2]}）")
    print(f"热球 {hot:02d}：自第{c['since']}回起 {k}/{len(after)} 回 = {k / len(after):.3f}（理论 {M / K:.3f}）\n")
    others = [x for x in range(1, K + 1) if x != hot]
    tickets = []
    while len(tickets) < 5:
        t = sorted([hot] + rng.sample(others, M - 1))
        if t not in tickets and all(len(set(t) & set(u)) <= M // 2 + 1 for u in tickets):
            tickets.append(t)
    for i, t in enumerate(tickets, 1):
        print(f"  {i}. " + " ".join(f"{x:02d}" for x in t))
    d = ROOT / "picks" / game
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{issue}.json"
    if f.exists() and not a.force:
        print(f"\n{f.name} 已存在，未覆盖（--force 可覆盖）")
        return
    f.write_text(json.dumps({"id": issue, "hot": hot, "tickets": tickets}, ensure_ascii=False, indent=1))
    print(f"\n已保存 japan/picks/{game}/{f.name}")


def cmd_check(game, _):
    draws = {i: r for i, r, _ in load(game)}
    for f in sorted((ROOT / "picks" / game).glob("*.json")):
        pk = json.loads(f.read_text())
        if pk["id"] not in draws:
            print(f"第{pk['id']}回：未开奖")
            continue
        R = set(draws[pk["id"]])
        print(f"第{pk['id']}回：开奖 " + " ".join(f"{x:02d}" for x in sorted(R)))
        for t in pk["tickets"]:
            h = len(set(t) & R)
            print("    " + " ".join(f"{x:02d}" for x in t) + f"   中 {h} 个{'  ★' if h == len(t) else ''}")


def cmd_sprt(game, _):
    c = CFG[game]
    p0, p1 = c["M"] / c["K"], c["p1"]
    A = log(19)
    L = 0.0
    rows = [r for r in load(game) if r[0] >= c["start"]]
    print(f"{c['title']} 热球 {c['hot']:02d} SPRT：H0 {p0:.3f} vs H1 {p1:.3f}；起点 第{c['start']}回；界 ±{A:.2f}\n")
    for i, r, _ in rows:
        x = c["hot"] in r
        L += log(p1 / p0) if x else log((1 - p1) / (1 - p0))
        print(f"  第{i}回 {'含' if x else '不含'} {c['hot']:02d}  累计 {L:+.3f}")
        if abs(L) >= A:
            break
    v = "✅ 判定：热球为真" if L >= A else ("✗ 判定：只是巧合" if L <= -A else "… 尚未下结论")
    print(f"\n已观察 {len(rows)} 回，累计 {L:+.3f}。{v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("game", choices=list(CFG))
    ap.add_argument("cmd", nargs="?", default="pick", choices=["pick", "check", "sprt"])
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"pick": cmd_pick, "check": cmd_check, "sprt": cmd_sprt}[a.cmd](a.game, a)
