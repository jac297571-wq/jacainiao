"""抓取每期使用的摇奖球套号（福建体彩网 data_api，字段 YaoJiangQiu），输出 data/dlt_ballset.csv"""
import csv, json, time, urllib.request
from pathlib import Path

API = "https://www.fjtc.com.cn/data_api/lottery?type=dlt&resType=1&page={}&limit={}"
OUT = Path(__file__).resolve().parent.parent / "data" / "dlt_ballset.csv"


def get(page, limit):
    req = urllib.request.Request(API.format(page, limit), headers={"User-Agent": "Mozilla/5.0"})
    for i in range(4):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=60).read())["data"] or []
        except Exception:
            time.sleep(2 ** i)
    raise RuntimeError(f"page {page} failed")


def main(limit=100):
    rows, page = {}, 1
    while True:
        data = get(page, limit)
        if not data:
            break
        for d in data:
            rows[d["QiHao"]] = {"issue": d["QiHao"], "date": d["KaiJiangRiQi"], "ballset": d.get("YaoJiangQiu", ""),
                                "order": d.get("ChuQiuShuZi", ""), "sorted": d.get("CaiGuoShuZi", "")}
        print(f"page {page}: {len(data)} 条，最早 {data[-1]['QiHao']}", flush=True)
        if len(data) < limit:
            break
        page += 1
        time.sleep(1)
    out = sorted(rows.values(), key=lambda r: r["issue"])
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} 期 -> {OUT}")


if __name__ == "__main__":
    main()
