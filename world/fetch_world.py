"""世界彩票语料库：抓取各国/各地公开的完整开奖历史，统一成 data/world/<名称>.csv（date,numbers）。
号码矩阵改过的彩票（例如 Powerball 59→69）自动按“号码范围”切段，每段一个文件，K 记在文件名里。

来源：纽约州开放数据（Powerball、NY Lotto、Take 5、Cash4Life）、德州彩票（Lotto Texas、Cash Five、Texas Two Step）、
17500.cn（福彩 3D、七星彩；排列 5 已在 data/other/pl5.txt）。数字型彩票每一位是一台独立的 10 球摇奖机，按"机器"拆成单独序列。
"""
import csv
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "world"
RAW = OUT / "raw"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"
SRC = {
    "ny_powerball": "https://data.ny.gov/api/views/d6yy-54nr/rows.csv",
    "ny_lotto": "https://data.ny.gov/api/views/6nbc-h7bj/rows.csv",
    "ny_take5": "https://data.ny.gov/api/views/dg63-4siq/rows.csv",
    "ny_cash4life": "https://data.ny.gov/api/views/kwxv-fwze/rows.csv",
    "tx_lotto": "https://www.texaslottery.com/export/sites/lottery/Games/Lotto_Texas/Winning_Numbers/lottotexas.csv",
    "tx_cash5": "https://www.texaslottery.com/export/sites/lottery/Games/Cash_Five/Winning_Numbers/cashfive.csv",
    "tx_twostep": "https://www.texaslottery.com/export/sites/lottery/Games/Texas_Two_Step/Winning_Numbers/texastwostep.csv",
    "cn_3d": "https://data.17500.cn/3d_asc.txt",
    "cn_7xc": "https://data.17500.cn/7xc_asc.txt",
}


def get(name, url):
    f = RAW / name
    for _ in range(4):
        r = subprocess.run(["curl", "-sS", "-m", "120", "-A", UA, "-o", str(f), url])
        if r.returncode == 0 and f.stat().st_size > 1000:
            break
    return f


def ny(f, cols, n):
    """纽约：日期 MM/DD/YYYY，号码以空格分隔；cols 为号码所在列名，取前 n 个"""
    out = []
    for r in csv.DictReader(f.open(encoding="utf-8")):
        d = datetime.strptime(r["Draw Date"], "%m/%d/%Y").date().isoformat()
        for c, tag in cols:
            v = (r.get(c) or "").split()
            if len(v) >= n:
                out.append((d + tag, [int(x) for x in v[:n]]))
    return out


def tx(f, n):
    out = []
    for line in f.open(encoding="utf-8", errors="ignore"):
        v = line.strip().split(",")
        if len(v) < 4 + n:
            continue
        d = f"{int(v[3]):04d}-{int(v[1]):02d}-{int(v[2]):02d}"
        out.append((d, [int(x) for x in v[4:4 + n]]))
    return out


def digits(f, pos):
    out = []
    for line in f.open(encoding="utf-8", errors="ignore"):
        v = line.split()
        try:
            out.append((v[1], [int(v[2 + p]) for p in pos]))
        except (ValueError, IndexError):
            pass
    return out


def segment(rows, min_len=400):
    """按号码上限切段：先按 300 期分块取最大号码得到粗分段，再把边界精确到"第一次出现新上限"/"最后一次出现旧上限"。"""
    rows = sorted(rows)
    mx = np.array([max(r) for _, r in rows])
    n = len(rows)
    blk = [int(mx[i:i + 300].max()) for i in range(0, n, 300)]
    # 合并相同上限的相邻块
    segs = []
    for i, K in enumerate(blk):
        if segs and segs[-1][0] == K:
            segs[-1][2] = min(n, (i + 1) * 300)
        else:
            segs.append([K, i * 300, min(n, (i + 1) * 300)])
    for j in range(1, len(segs)):
        K1, s1, _ = segs[j - 1]
        K2 = segs[j][0]
        lo, hi = max(0, segs[j][1] - 300), min(n, segs[j][1] + 300)
        if K2 > K1:
            cut = lo + int(np.argmax(mx[lo:hi] > K1))
        else:
            idx = np.flatnonzero(mx[lo:hi] > K2)
            cut = lo + (int(idx[-1]) + 1 if len(idx) else 0)
        segs[j - 1][2] = cut
        segs[j][1] = cut
    return [(int(mx[s:e].max()), rows[s:e]) for K, s, e in segs if e - s >= min_len]


def save(name, K, M, rows):
    f = OUT / f"{name}_{M}of{K}.csv"
    with f.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "numbers"])
        for d, r in rows:
            w.writerow([d, " ".join(map(str, r))])
    print(f"{f.name}: {len(rows)} 期 {rows[0][0]} ~ {rows[-1][0]}")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.csv"):
        old.unlink()
    raw = {k: get(k, u) for k, u in SRC.items()}
    games = {
        "ny_powerball": (ny(raw["ny_powerball"], [("Winning Numbers", "")], 5), 5),
        "ny_lotto": (ny(raw["ny_lotto"], [("Winning Numbers", "")], 6), 6),
        "ny_take5_eve": (ny(raw["ny_take5"], [("Evening Winning Numbers", "")], 5), 5),
        "ny_take5_mid": (ny(raw["ny_take5"], [("Midday Winning Numbers", "")], 5), 5),
        "ny_cash4life": (ny(raw["ny_cash4life"], [("Winning Numbers", "")], 5), 5),
        "tx_lotto": (tx(raw["tx_lotto"], 6), 6),
        "tx_cash5": (tx(raw["tx_cash5"], 5), 5),
        "tx_twostep": (tx(raw["tx_twostep"], 4), 4),
    }
    for name, (rows, M) in games.items():
        rows = [(d, sorted(r)) for d, r in rows if len(set(r)) == M]
        for K, seg in segment(rows):
            save(name, K, M, seg)
    # 数字型：每一位是一台独立的 0–9 摇奖机，号码 +1 变成 1–10
    for name, f, npos in (("cn_3d", raw["cn_3d"], 3), ("cn_pl5", ROOT / "data" / "other" / "pl5.txt", 5),
                          ("cn_7xc", raw["cn_7xc"], 6)):
        rows = digits(f, range(npos))
        for p in range(npos):
            save(f"{name}_pos{p + 1}", 10, 1, sorted((d, [r[p] + 1]) for d, r in rows))


if __name__ == "__main__":
    main()
