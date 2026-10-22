"""用本地 ollama 大模型当"预测者"，按与贝叶斯模型相同的尺子做严格前瞻评测。

每一期：把截至上一期的最近 W 期开奖发给模型，让它给出下一期每个号码的权重（JSON），
然后按实际开奖计算"开出这组号码的概率 / 纯随机概率"（Plackett-Luce 精确集合概率），即头奖概率倍数。
同时给出两个对照：纯随机（×1.000）和"最近 W 期出现频率"这一朴素基线。
模型回答会缓存到 llm_cache/，中断后重跑会自动续上。

用法（在本地有 ollama 的电脑上，仓库根目录运行）：
  python3 llm_eval.py --model qwen2.5:14b-instruct-q4_K_M --game 大乐透 --n 100
  python3 llm_eval.py --model huihui_ai/qwen2.5-abliterate:14b --game ミニロト --n 100
可选：--window 100（喂给模型的历史期数）--host http://127.0.0.1:11434
彩种：大乐透（只评 2014 年后）、双色球、ロト6、ロト7、ミニロト
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from math import comb, log
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deep"))
import radar  # noqa: E402
from dynpl import set_logp  # noqa: E402

GAMES = {"大乐透": ("dlt", 35, 5), "双色球": ("ssq", 33, 6), "ロト6": ("loto6", 43, 6),
         "ロト7": ("loto7", 37, 7), "ミニロト": ("miniloto", 31, 5)}
SMOOTH = 0.05  # 与 5% 均匀混合，防止模型给某号码 0 权重导致 -∞


def ask(host, model, prompt):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json",
                       "options": {"temperature": 0, "num_ctx": 8192}}).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["response"]


def parse(text, K):
    w = np.zeros(K)
    try:
        obj = json.loads(text)
        items = obj.get("weights", obj) if isinstance(obj, dict) else {}
        for k, v in items.items():
            n = int(re.sub(r"\D", "", str(k)) or 0)
            if 1 <= n <= K:
                w[n - 1] = max(float(v), 0.0)
    except Exception:
        pass
    return w if w.sum() > 0 else None


def prompt_for(game, K, M, hist):
    lines = "\n".join(" ".join(f"{x:02d}" for x in sorted(r)) for r in hist)
    return (f"下面是{game}最近 {len(hist)} 期的开奖号码（每行一期，从旧到新；每期从 1–{K} 中开出 {M} 个不重复号码）：\n"
            f"{lines}\n\n请预测下一期。对 1 到 {K} 的每个号码给出一个非负权重，权重越大表示你认为它越可能开出。"
            f'只输出 JSON，格式为 {{"weights": {{"1": 0.03, "2": 0.02, ...}}}}，必须包含全部 {K} 个号码。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--game", default="大乐透", choices=list(GAMES))
    ap.add_argument("--n", type=int, default=100, help="评测最近多少期")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    a = ap.parse_args()
    ld, K, M = GAMES[a.game]
    rows = [r for i, r in radar.load(ld)]
    if ld == "dlt":
        ids = [i for i, _ in radar.load("dlt")]
        rows = rows[next(j for j, i in enumerate(ids) if i >= "14001"):]
    cache_dir = ROOT / "llm_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_f = cache_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', a.model)}_{ld}_w{a.window}.jsonl"
    cache = {}
    if cache_f.exists():
        for line in cache_f.open(encoding="utf-8"):
            d = json.loads(line)
            cache[d["t"]] = d["resp"]
    base = -log(comb(K, M))
    g_llm, g_freq, hits = [], [], []
    start = len(rows) - a.n
    for t in range(start, len(rows)):
        hist = rows[t - a.window:t]
        if t not in cache:
            t0 = time.time()
            resp = ask(a.host, a.model, prompt_for(a.game, K, M, hist))
            cache[t] = resp
            with cache_f.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": t, "resp": resp}, ensure_ascii=False) + "\n")
            print(f"  第 {t - start + 1}/{a.n} 期，用时 {time.time() - t0:.0f}s", flush=True)
        w = parse(cache[t], K)
        if w is None:
            w = np.ones(K)  # 回答无法解析：按均匀处理（不给模型加分也不扣分）
        p = (1 - SMOOTH) * w / w.sum() + SMOOTH / K
        s = np.array(rows[t]) - 1
        g_llm.append((set_logp(np.log(p), s) - base) / log(2))
        c = np.zeros(K)
        for r in hist:
            c[np.array(r) - 1] += 1
        q = (1 - SMOOTH) * c / c.sum() + SMOOTH / K
        g_freq.append((set_logp(np.log(q), s) - base) / log(2))
        hits.append(np.isin(np.argsort(-p)[:M], s).sum())

    def line(name, g):
        g = np.array(g)
        z = g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))
        return f"| {name} | {g.sum():+.2f} | ×{2 ** g.mean():.4f} | {z:+.2f} |"

    out = [f"# {a.model} · {a.game} · 最近 {a.n} 期前瞻评测（窗口 {a.window} 期）\n",
           "| 预测者 | 总增益（比特） | 头奖概率倍数 | z |", "|---|---|---|---|",
           "| 纯随机 | +0.00 | ×1.0000 | — |", line("近期频率（朴素基线）", g_freq), line(a.model, g_llm),
           f"\n模型每期最看好的 {M} 个号码平均命中 {np.mean(hits):.3f} 个（随机期望 {M * M / K:.3f}）。",
           "倍数 < 1 表示比随机选号更差。参考：贝叶斯模型当前各彩种约 ×1.00–1.22（见 report_dynpl.md）。"]
    text = "\n".join(out)
    print("\n" + text)
    (cache_dir / f"report_{cache_f.stem}.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
