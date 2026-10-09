"""抓取ロト6リハーサル数字（速报ナビ hpfree.com，三个页面覆盖第1回至今）。
リハーサル固定使用 A セット球。输出 japan/data/loto6_rehearsal.csv：id, r1..r6（出球顺序）, rb（ボーナス）"""
import csv, re, time, urllib.request
from pathlib import Path

PAGES = ["https://www.hpfree.com/takarakuji/loto6/rehearsal.html",
         "https://www.hpfree.com/takarakuji/loto6/rehearsal-1.html",
         "https://www.hpfree.com/sml6/loto6/rehearsal.html"]
OUT = Path(__file__).resolve().parent.parent / "data" / "loto6_rehearsal.csv"


def get(url):
    for i in range(5):
        try:
            b = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
            for enc in ("utf-8", "shift_jis", "euc-jp"):
                try:
                    return b.decode(enc)
                except UnicodeDecodeError:
                    pass
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(url)


def parse(html):
    txt = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S)
    s = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", txt))
    rows = {}
    # 第N回 后跟 7 个两位数字（可夹颜色字）
    for m in re.finditer(r"第(\d+)回((?:\s+\d{1,2}(?:\s+[^\s\d第]{1,2})?){7})", s):
        nums = re.findall(r"\b(\d{1,2})\b", m.group(2))
        if len(nums) == 7 and all(1 <= int(x) <= 43 for x in nums) and len(set(nums)) == 7:
            rows[int(m.group(1))] = [int(x) for x in nums]
    return rows


def main():
    allrows = {}
    for u in PAGES:
        r = parse(get(u))
        print(u, len(r), (min(r), max(r)) if r else None, flush=True)
        for k, v in r.items():
            allrows.setdefault(k, v)
        time.sleep(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "r1", "r2", "r3", "r4", "r5", "r6", "rb"])
        for k in sorted(allrows):
            w.writerow([k] + allrows[k])
    ids = sorted(allrows)
    print(f"{len(ids)} 回 -> {OUT}  ({ids[0]}–{ids[-1]})，缺失 {ids[-1] - ids[0] + 1 - len(ids)} 回")


if __name__ == "__main__":
    main()
