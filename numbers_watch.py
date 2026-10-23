"""ナンバーズ3 / ナンバーズ4 正收益哨兵。

ナンバーズ回报率约 45%：要转正，中奖概率需要约 ×2.2（每位数字偏热约 30%/22%）。
平时不下注；本脚本每天开奖后检查：在最近 100 / 300 / 1000 回里，按每一位分别估计各数字的出现概率
（带强收缩先验的 Dirichlet 后验），取"每位最热数字"组成的号码，计算它的期望回报 > 1 的后验概率。
因为是在 1000（或 10000）个号码、3 个窗口里挑最好的，假报警风险很大，所以告警阈值
用纯随机数据跑同样流程来校准：随机情形下约每 1000 回（约 4 年）才误报一次。
另设"风车异常"预警：任一位置的数字分布显著偏离均匀（校准到约每 250 回、约一年误报一次），
它比正收益告警灵敏得多，出现时先观察，等正收益告警再下注。
注意：期望回报只略高于 1 的边缘偏差，统计上需要很多回才能确认；本哨兵主要抓"大偏差"。
用法：python3 numbers_watch.py [--update]   → 输出状态并写 numbers_status.json
"""
import argparse
import csv
import io
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "japan" / "data"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"
GAMES = {"numbers3": {"title": "ナンバーズ3", "D": 3, "mini": True},
         "numbers4": {"title": "ナンバーズ4", "D": 4, "mini": False}}
WINDOWS = (100, 300, 1000)
PRICE = 200
MC = 4000
PRIOR = 8.0  # 每个数字的 Dirichlet 伪计数：先验认为偏差很小，需要足够多数据才会承认偏差（抵消“挑最热”带来的赢家诅咒）
CAL_FILE = DATA / "numbers_thresholds.json"


def update(game):
    raw = subprocess.run(["curl", "-sS", "-m", "120", "-A", UA, f"https://loto-life.net/csv/{game}"],
                         capture_output=True).stdout
    rows = list(csv.reader(io.StringIO(raw.decode("cp932", "ignore"))))
    if len(rows) < 1000:
        print(f"{game}: 下载失败，保留旧数据")
        return
    with (DATA / f"{game}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "date", "number", "straight", "mini"])
        for r in rows[1:]:
            n = r[2].replace("=", "").replace('"', "")
            if n.isdigit():
                w.writerow([r[0], r[1], n, r[4], r[12] if len(r) > 12 else ""])


def load(game):
    return list(csv.DictReader((DATA / f"{game}.csv").open(encoding="utf-8")))


def stat(digits, payout, rng, mc=MC):
    """digits: (W, D) 最近 W 回的各位数字。返回 (最热号码, 期望回报后验均值, P(期望回报>1))"""
    W, D = digits.shape
    counts = np.stack([np.bincount(digits[:, p], minlength=10) for p in range(D)])  # (D,10)
    best = counts.argmax(1)
    alpha = counts + PRIOR
    # Dirichlet 后验下，最热号码的中奖概率 = Π_p θ_{p,best_p}；θ 的边缘是 Beta
    a = alpha[np.arange(D), best]
    b = alpha.sum(1) - a
    th = rng.beta(a, b, size=(mc, D)).prod(1)
    ev = th * payout / PRICE
    return "".join(map(str, best)), float(ev.mean()), float((ev > 1).mean())


def scan(digits, payout, rng, mc=MC):
    out = []
    for Wn in WINDOWS:
        if len(digits) >= Wn:
            out.append((Wn, *stat(digits[-Wn:], payout, rng, mc)))
    return out


def wheel_stat(digits):
    """风车异常：各窗口、各位置的卡方统计量（df=9）的最大值"""
    best = 0.0
    for Wn in WINDOWS:
        if len(digits) >= Wn:
            for p in range(digits.shape[1]):
                c = np.bincount(digits[-Wn:, p], minlength=10)
                e = Wn / 10
                best = max(best, float(((c - e) ** 2 / e).sum()))
    return best


def calibrate_wheel(D, rng, sims=1000, target=250):
    """预警档：随机情形下约每 250 回（约一年）误报一次"""
    vals = [wheel_stat(rng.integers(0, 10, size=(max(WINDOWS), D))) for _ in range(sims)]
    return float(np.quantile(vals, 1 - 1 / target))


def calibrate(D, payout, rng, sims=1000, target=1000):
    """随机数据上，每回的统计量 max_W P(EV>1) 的分布；取使误报率 = 1/target 的阈值"""
    vals = []
    for _ in range(sims):
        dg = rng.integers(0, 10, size=(max(WINDOWS), D))
        vals.append(max(s[3] for s in scan(dg, payout, rng, mc=800)))
    return float(np.quantile(vals, 1 - 1 / target)) if sims >= target else float(max(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    cal = json.loads(CAL_FILE.read_text()) if CAL_FILE.exists() else {}
    status = {}
    for game, g in GAMES.items():
        if a.update:
            update(game)
        rows = load(game)
        D = g["D"]
        digits = np.array([[int(c) for c in r["number"]] for r in rows if len(r["number"]) == D])
        pays = [float(r["straight"]) for r in rows[-100:] if r["straight"] not in ("", "0")]
        payout = float(np.median(pays))
        if game not in cal:
            cal[game] = calibrate(D, payout, rng)
            CAL_FILE.write_text(json.dumps(cal))
        if game + "_wheel" not in cal:
            cal[game + "_wheel"] = calibrate_wheel(D, rng)
            CAL_FILE.write_text(json.dumps(cal))
        ws = wheel_stat(digits)
        res = scan(digits, payout, rng)
        top = max(res, key=lambda s: s[3])
        alarm = top[3] > cal[game]
        st = {"latest": rows[-1]["id"], "date": rows[-1]["date"], "payout_median": payout,
              "threshold": round(cal[game], 3), "alarm": alarm,
              "wheel_chi2": round(ws, 1), "wheel_threshold": round(cal[game + "_wheel"], 1),
              "wheel_alarm": ws > cal[game + "_wheel"],
              "windows": [{"W": w, "best": n, "ev": round(e, 3), "p_ev_gt_1": round(pp, 3)} for w, n, e, pp in res]}
        if g["mini"]:
            mpays = [float(r["mini"]) for r in rows[-100:] if r.get("mini") not in (None, "", "0")]
            mpay = float(np.median(mpays))
            if game + "_mini" not in cal:
                cal[game + "_mini"] = calibrate(2, mpay, rng)
                CAL_FILE.write_text(json.dumps(cal))
            mini = scan(digits[:, 1:], mpay, rng)
            mt = max(mini, key=lambda s: s[3])
            st["mini"] = [{"W": w, "best": n, "ev": round(e, 3), "p_ev_gt_1": round(pp, 3)} for w, n, e, pp in mini]
            st["mini_threshold"] = round(cal[game + "_mini"], 3)
            st["mini_alarm"] = mt[3] > cal[game + "_mini"]
        status[g["title"]] = st
        flag = "🔴 正收益告警！" if alarm or st.get("mini_alarm") else ("🟡 风车异常预警" if st["wheel_alarm"] else "无告警")
        print(f"{g['title']}（截至第{st['latest']}回 {st['date']}，直选奖金中位 {payout:,.0f} 日元）：{flag}")
        print(f"   风车异常：最大卡方 {ws:.1f}（预警阈值 {cal[game + '_wheel']:.1f}）")
        for w in st["windows"]:
            print(f"   近 {w['W']} 回最热 {w['best']}：期望回报 {w['ev']:.2f}，P(>1)={w['p_ev_gt_1']:.3f}（告警阈值 {cal[game]:.3f}）")
        for w in st.get("mini", []):
            print(f"   ミニ 近 {w['W']} 回最热 {w['best']}：期望回报 {w['ev']:.2f}，P(>1)={w['p_ev_gt_1']:.3f}（告警阈值 {st['mini_threshold']:.3f}）")
    (ROOT / "numbers_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
