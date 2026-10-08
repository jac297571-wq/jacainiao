"""抓取大乐透全部历史开奖（数据源：500.com 走势图历史页），输出 data/dlt_history.csv"""
import csv, re, sys, urllib.request
from pathlib import Path

URL = "https://datachart.500.com/dlt/history/newinc/history.php?start=07001&end=99999"
OUT = Path(__file__).resolve().parent.parent / "data" / "dlt_history.csv"


def num(s):
    s = s.replace(",", "").strip()
    return int(s) if s.isdigit() else ""


def main(src=None):
    if src:
        raw = Path(src).read_bytes()
    else:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=60).read()
    html = raw.decode("gb2312", errors="ignore")
    body = html.split('id="tdata"', 1)[1].split("</tbody>", 1)[0]
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S):
        tr = re.sub(r"<!--.*?-->", "", tr)
        c = [re.sub(r"<[^>]+>", "", x).strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(c) < 15:
            continue
        rows.append({
            "issue": c[0], "date": c[14],
            "r1": c[1], "r2": c[2], "r3": c[3], "r4": c[4], "r5": c[5],
            "b1": c[6], "b2": c[7],
            "pool": num(c[8]), "first_n": num(c[9]), "first_prize": num(c[10]),
            "second_n": num(c[11]), "second_prize": num(c[12]), "sales": num(c[13]),
        })
    rows.sort(key=lambda r: r["issue"])
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} 期 -> {OUT}  ({rows[0]['issue']} ~ {rows[-1]['issue']})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
