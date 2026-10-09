"""抓取ロト6全部开奖（loto6.jp 数据接口）：本数字、出球顺序、ボーナス、セット球、会场、奖金、キャリーオーバー。
输出 japan/data/loto6.csv"""
import csv, json, time, urllib.request
from pathlib import Path

URL = "https://loto6.jp/databases/resultlist/data?count={}&order=id&order_rule=asc&offset={}"
HDR = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest", "Referer": "https://loto6.jp/databases/resultlist"}
OUT = Path(__file__).resolve().parent.parent / "data" / "loto6.csv"
KEYS = ["id", "date", "n1", "n2", "n3", "n4", "n5", "n6", "nbo", "o1", "o2", "o3", "o4", "o5", "o6",
        "setball", "stage_id", "t1", "m1", "t2", "m2", "t3", "m3", "t4", "m4", "t5", "m5", "co", "sales"]


def get(url):
    for i in range(6):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=60).read())
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(url)


def main(step=100):
    rows, offset = [], 1
    while True:
        d = get(URL.format(step, offset))
        rows += d["data"]
        print(f"{len(rows)}/{d['all']}", flush=True)
        if len(rows) >= d["all"] or not d["data"]:
            break
        offset += step
        time.sleep(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: int(r["id"])))
    print(f"{len(rows)} 回 -> {OUT}")


if __name__ == "__main__":
    main()
