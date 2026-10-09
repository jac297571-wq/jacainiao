"""双色球 5 注选号器 + “热球”前瞻检验。

依据（scripts/ssq_hunt.py / report_ssq.md）：红球 24 自 2025084 期起出现率 28.4%（理论 18.2%，z=+3.91），
按正确校准（对随机数据执行同样的扫描流程），校正 p=0.224，**不显著**，属于弱线索。
策略：每注包含该号码。若开奖真随机，这样做没有任何代价；若偏差为真，则有优势。最终由前瞻 SPRT 裁决。

用法：
  python3 ssq_lucky.py --update   # 更新数据并生成下一期 5 注（picks_ssq/<期号>.json）
  python3 ssq_lucky.py check      # 对奖
  python3 ssq_lucky.py sprt       # 热球前瞻序贯检验（自 2026116 起）
"""
import argparse
import json
import secrets
import urllib.request
from math import log, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "other" / "ssq.txt"
PICKS = ROOT / "picks_ssq"
HOT = 24
HOT_SINCE = "2025084"
P0 = 6 / 33
P1 = 0.25  # 备择假设：热球出现率 25%（观测 28.4% 向理论收缩）
SPRT_START = "2026116"
rng = secrets.SystemRandom()


def update():
    req = urllib.request.Request("https://data.17500.cn/ssq_asc.txt", headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"})
    DATA.write_bytes(urllib.request.urlopen(req, timeout=60).read())


def load():
    rows = []
    for line in DATA.open(encoding="utf-8", errors="ignore"):
        f = line.split()
        try:
            rows.append((f[0], [int(x) for x in f[2:8]], int(f[8])))
        except (ValueError, IndexError):
            pass
    return rows


def hot_signal(rows):
    after = [r for r in rows if r[0] >= HOT_SINCE]
    n = len(after)
    k = sum(HOT in r[1] for r in after)
    z = (k / n - P0) / sqrt(P0 * (1 - P0) * 27 / 32 / n)
    return n, k, z


def next_issue(issue):
    return str(int(issue) + 1)


def fmt(r, b):
    return " ".join(f"{x:02d}" for x in sorted(r)) + "  +  " + f"{b:02d}"


def cmd_pick(a):
    if a.update:
        update()
    rows = load()
    n, k, z = hot_signal(rows)
    use = z > 2
    issue = next_issue(rows[-1][0])
    print(f"双色球 第 {issue} 期（数据截至 {rows[-1][0]}）")
    print(f"热球 {HOT:02d}：自 {HOT_SINCE} 起 {k}/{n} 期 = {k / n:.3f}（理论 {P0:.3f}），z={z:+.2f}，"
          + ("信号成立，每注都含 24" if use else "信号消失，均匀随机"))
    tickets, blues = [], []
    others = [x for x in range(1, 34) if x != HOT]
    while len(tickets) < 5:
        r = sorted(([HOT] if use else []) + rng.sample(others if use else list(range(1, 34)), 5 if use else 6))
        if r in tickets or any(len(set(r) & set(t)) > (3 if use else 2) for t in tickets):
            continue
        tickets.append(r)
    while len(blues) < 5:
        b = rng.randint(1, 16)
        if b not in blues:
            blues.append(b)
    print()
    for i, (r, b) in enumerate(zip(tickets, blues), 1):
        print(f"  {i}. {fmt(r, b)}")
    PICKS.mkdir(exist_ok=True)
    f = PICKS / f"{issue}.json"
    if f.exists() and not a.force:
        print(f"\n{f.name} 已存在，未覆盖（--force 可覆盖）")
        return
    f.write_text(json.dumps({"issue": issue, "hot": HOT, "hot_z": round(z, 3), "use_hot": use,
                             "tickets": [{"red": r, "blue": b} for r, b in zip(tickets, blues)]}, ensure_ascii=False, indent=1))
    print(f"\n已保存 picks_ssq/{f.name}")


def cmd_check(_):
    draws = {i: (r, b) for i, r, b in load()}
    for f in sorted(PICKS.glob("*.json")):
        pk = json.loads(f.read_text())
        if pk["issue"] not in draws:
            print(f"{pk['issue']}: 未开奖")
            continue
        R, B = draws[pk["issue"]]
        print(f"{pk['issue']}: 开奖 {fmt(R, B)}")
        for t in pk["tickets"]:
            hr, hb = len(set(t["red"]) & set(R)), int(t["blue"] == B)
            print(f"    {fmt(t['red'], t['blue'])}   红{hr}+蓝{hb}{'  ★' if (hr, hb) == (6, 1) else ''}")


def cmd_sprt(_):
    rows = [r for r in load() if r[0] >= SPRT_START]
    A, Bd = log(19), -log(19)
    L = 0.0
    print(f"热球 {HOT:02d} 前瞻 SPRT：H0 出现率 {P0:.3f} vs H1 {P1:.3f}；起点 {SPRT_START}；上界 {A:+.2f}，下界 {Bd:+.2f}\n")
    for i, r, _ in rows:
        x = HOT in r
        L += log(P1 / P0) if x else log((1 - P1) / (1 - P0))
        print(f"  {i}  {'含' if x else '不含'} {HOT:02d}  累计 {L:+.3f}")
        if L >= A or L <= Bd:
            break
    v = "✅ 判定：热球为真" if L >= A else ("✗ 判定：只是巧合" if L <= Bd else "… 尚未下结论")
    print(f"\n已观察 {len(rows)} 期，累计 {L:+.3f}。{v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="pick", choices=["pick", "check", "sprt"])
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"pick": cmd_pick, "check": cmd_check, "sprt": cmd_sprt}[a.cmd](a)
