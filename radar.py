"""热球雷达：每期开奖后对 5 种彩票的每个号码做 CUSUM 监控，号码一开始变热就报警。

方法：Page CUSUM，对每个号码每期是否开出做伯努利对数似然比累积（H0 出现率 p0，H1 = 2×p0），负值归零。
两档阈值（纯随机模拟标定）：
  下注档 bet：每个彩种约每 30 期误报一次。按热球下注在随机情形下没有任何代价，所以宁可多报，尽早上车。
  证明档 proof：每个彩种约每 300 期误报一次，用来判断热球是否可信。
用法：python3 radar.py [--update]   → 输出各彩种当前状态与告警（同时写 radar_status.json）
"""
import argparse
import csv
import json
import subprocess
import sys
from math import log
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GAMES = {
    "大乐透": {"K": 35, "M": 5, "loader": "dlt"},
    "双色球": {"K": 33, "M": 6, "loader": "ssq"},
    "ロト6": {"K": 43, "M": 6, "loader": "loto6"},
    "ロト7": {"K": 37, "M": 7, "loader": "loto7"},
    "ミニロト": {"K": 31, "M": 5, "loader": "miniloto"},
}
TARGETS = {"bet": 30, "proof": 300}  # 每个彩种平均多少期误报一次
THRESH_CACHE = ROOT / "data" / "radar_thresholds.json"


def load(loader):
    if loader == "dlt":
        rows = list(csv.DictReader((ROOT / "data" / "dlt_history.csv").open(encoding="utf-8")))
        return [(r["issue"], [int(r[f"r{i}"]) for i in range(1, 6)]) for r in rows]
    if loader == "ssq":
        out = []
        for line in (ROOT / "data" / "other" / "ssq.txt").open(encoding="utf-8", errors="ignore"):
            f = line.split()
            try:
                out.append((f[0], [int(x) for x in f[2:8]]))
            except (ValueError, IndexError):
                pass
        return out
    M = {"loto6": 6, "loto7": 7, "miniloto": 5}[loader]
    rows = csv.DictReader((ROOT / "japan" / "data" / f"{loader}.csv").open(encoding="utf-8"))
    return [(r["id"], [int(r[f"n{i}"]) for i in range(1, M + 1)]) for r in rows]


def increments(K, M):
    p0 = M / K
    p1 = min(2 * p0, 0.9)
    return log(p1 / p0), log((1 - p1) / (1 - p0))


def run_cusum(X, up, down):
    S = np.zeros(X.shape[1])
    hist = np.zeros_like(X, dtype=float)
    for t in range(len(X)):
        S = np.maximum(0.0, S + np.where(X[t] > 0, up, down))
        hist[t] = S
    return hist


def calibrate(K, M, rng, target, n=6000):
    """模拟 n 期纯随机开奖，找使全部号码合计误报间隔 ≈ target 的阈值"""
    up, down = increments(K, M)
    X = np.zeros((n, K))
    X[np.arange(n)[:, None], np.argsort(rng.random((n, K)), 1)[:, :M]] = 1
    for H in np.arange(0.5, 12.0, 0.1):
        S = np.zeros(K)
        alarms = 0
        for t in range(n):
            S = np.maximum(0.0, S + np.where(X[t] > 0, up, down))
            hit = S >= H
            alarms += hit.sum()
            S[hit] = 0
        if alarms == 0 or n / alarms >= target:
            return round(float(H), 2)
    return 12.0


def delay(K, M, H, rng, sims=300):
    """某号码出现率真实翻倍时，告警延迟（期）的中位数"""
    up, down = increments(K, M)
    p1 = min(2 * M / K, 0.9)
    d = []
    for _ in range(sims):
        s, t = 0.0, 0
        while s < H and t < 2000:
            s = max(0.0, s + (up if rng.random() < p1 else down))
            t += 1
        d.append(t)
    return int(np.median(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="先更新大乐透/双色球/日本三种彩票的数据")
    a = ap.parse_args()
    if a.update:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch.py")], check=False)
        subprocess.run([sys.executable, str(ROOT / "japan" / "scripts" / "fetch_loto.py")], check=False)
        subprocess.run(["curl", "-sS", "-m", "60", "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0",
                        "-o", str(ROOT / "data" / "other" / "ssq.txt"), "https://data.17500.cn/ssq_asc.txt"], check=False)
    rng = np.random.default_rng(2026)
    th = json.loads(THRESH_CACHE.read_text()) if THRESH_CACHE.exists() else {}
    status = {}
    print("热球雷达（CUSUM，H1 = 出现率翻倍）。下注档：约每 30 期误报一次；证明档：约每 300 期误报一次\n")
    for name, g in GAMES.items():
        K, M = g["K"], g["M"]
        key = f"{K}_{M}"
        if key not in th or "bet" not in th[key]:
            th[key] = {}
            for mode, tgt in TARGETS.items():
                H_ = calibrate(K, M, rng, tgt)
                th[key][mode] = {"H": H_, "delay": delay(K, M, H_, rng)}
        H = th[key]["proof"]["H"]
        Hb = th[key]["bet"]["H"]
        rows = load(g["loader"])
        X = np.zeros((len(rows), K))
        for i, (_, r) in enumerate(rows):
            X[i, np.array(r) - 1] = 1
        up, down = increments(K, M)
        S = run_cusum(X, up, down)[-1]
        order = np.argsort(-S)
        alarms = [int(i + 1) for i in order if S[i] >= H]
        bet = [int(i + 1) for i in order if S[i] >= Hb]
        top = [(int(i + 1), round(float(S[i]), 2)) for i in order[:3]]
        status[name] = {"latest": rows[-1][0], "H": H, "H_bet": Hb, "alarms": alarms, "bet": bet, "top": top}
        flag = ("🔴 证明档告警：" + "、".join(f"{x:02d}" for x in alarms) + "；") if alarms else ""
        flag += ("🟡 下注档热球：" + "、".join(f"{x:02d}" for x in bet)) if bet else "下注档无热球"
        print(f"{name}（截至 {rows[-1][0]}；下注档阈值 {Hb}、约 {th[key]['bet']['delay']} 期抓到翻倍热球；"
              f"证明档阈值 {H}、约 {th[key]['proof']['delay']} 期）：{flag}")
        print("   累计值前三：" + "，".join(f"{n:02d}={v}" for n, v in top) + "\n")
    THRESH_CACHE.write_text(json.dumps(th, indent=1))
    (ROOT / "radar_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()


def hot_for(game_name, k=1):
    """供选号器调用：返回该彩种下注档最热的 k 个号码（按 CUSUM 值降序）"""
    f = ROOT / "radar_status.json"
    if not f.exists():
        return []
    st = json.loads(f.read_text()).get(game_name, {})
    return st.get("bet", [])[:k]
