"""抓取日本ロト系列全部开奖（loto6.jp / loto7.jp / miniloto.jp 同构数据接口）。
用法：python3 japan/scripts/fetch_loto.py [loto6|loto7|miniloto ...]  → japan/data/<game>.csv"""
import csv, json, sys, time, urllib.request
from pathlib import Path

SITES = {"loto6": "loto6.jp", "loto7": "loto7.jp", "miniloto": "miniloto.jp"}
OUT = Path(__file__).resolve().parent.parent / "data"


def get(url, ref):
    hdr = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest", "Referer": ref}
    for i in range(6):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60).read())
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(url)


def fetch(game, step=100):
    host = SITES[game]
    ref = f"https://{host}/databases/resultlist"
    rows, offset = [], 1
    while True:
        d = get(f"https://{host}/databases/resultlist/data?count={step}&order=id&order_rule=asc&offset={offset}", ref)
        rows += d["data"]
        if len(rows) >= d["all"] or not d["data"]:
            break
        offset += step
        time.sleep(1)
    keys = [k for k in rows[0] if not k.endswith(("_g6", "_g11")) and k not in ("rownum",)]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{game}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: int(r["id"])))
    print(f"{game}: {len(rows)} 回 -> {OUT / (game + '.csv')}", flush=True)


if __name__ == "__main__":
    for g in sys.argv[1:] or SITES:
        fetch(g)
