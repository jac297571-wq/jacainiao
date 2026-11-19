"""“凭直觉”实验：把每个号码的历史写成一串 0/1，不告诉模型这是彩票，只问下一位是 1 的概率。

每一期、每个号码（大乐透前区 35 个、后区 12 个）各问一次：给出这个号码最近 W 期的 0/1 序列（1 = 那一期开出），
让模型给出下一位是 1 的概率。留出最后 N 期（默认 50）评测，三把尺子：
  1. 前 12 / 后 6 命中：按模型概率排出前区前 12、后区前 6，开奖号码平均落进几个（随机期望 1.714 / 1.000）；
     同时把模型的概率在号码之间随机打乱 2000 次，得到“没有信息”时的分布，算 p 值。
  2. 对数损失（比特）：相对“每个号码都用基础概率 5/35、2/12”的预测，模型多给了多少信息（正 = 比基础概率好）。
  3. 与“近 W 期频率”这一朴素基线对比。
注意：50 期 × 前区 5 个号，统计力有限，只能看出很大的效果（前 12 命中要到 2.0 左右才显著）。
用法（Mac 上，仓库根目录，ollama 已启动）：
  python3 llm/binary_seq.py --model huihui_ai/qwen2.5-abliterate:14b
  可选：--n 50（留出期数）--window 100（序列长度）--pause 0.3（每次调用后暂停秒数，降低 GPU 占用）
回答缓存在 llm_cache/binary_*.jsonl，中断后重跑自动续上；报告写到 llm/report_binary_<模型>.md
"""
import argparse
import csv
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
START = "14001"


def load():
    rows = [r for r in csv.DictReader((ROOT / "data" / "dlt_history.csv").open(encoding="utf-8")) if r["issue"] >= START]
    F = np.zeros((len(rows), 35), int)
    B = np.zeros((len(rows), 12), int)
    for i, r in enumerate(rows):
        F[i, [int(r[f"r{k}"]) - 1 for k in range(1, 6)]] = 1
        B[i, [int(r["b1"]) - 1, int(r["b2"]) - 1]] = 1
    return rows, F, B


def prompt(seq):
    s = "".join(map(str, seq))
    return ("Here is a sequence of 0s and 1s, oldest first:\n" + s +
            "\n\nWhat is the probability that the next symbol is 1? Go with your intuition. "
            'Reply with JSON only, like {"p": 0.25}.')


def ask(host, model, text):
    body = json.dumps({"model": model, "prompt": text, "stream": False, "format": "json",
                       "options": {"temperature": 0, "num_ctx": 2048}}).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["response"]


def parse(text):
    try:
        v = json.loads(text).get("p")
    except Exception:
        m = re.search(r"0?\.\d+|[01](?:\.0+)?", text or "")
        v = m.group(0) if m else None
    try:
        v = float(v)
        if v > 1 and v <= 100:
            v /= 100
        return min(max(v, 0.001), 0.999) if 0 <= v <= 1 else None
    except (TypeError, ValueError):
        return None


_jit = np.random.default_rng(1)


def topk_hits(P, X, k):
    """并列时随机打破（模型常给很多相同的概率，不能让号码小的总排在前面）"""
    P = P + _jit.random(P.shape) * 1e-9
    return np.array([X[i, np.argsort(-P[i])[:k]].sum() for i in range(len(P))])


def bits(P, X, base):
    P = np.clip(P, 1e-3, 1 - 1e-3)
    ll = X * np.log2(P) + (1 - X) * np.log2(1 - P)
    l0 = X * np.log2(base) + (1 - X) * np.log2(1 - base)
    return float((ll - l0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--pause", type=float, default=0.0)
    a = ap.parse_args()
    rows, F, B = load()
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", a.model)
    cdir = ROOT / "llm_cache"
    cdir.mkdir(exist_ok=True)
    cf = cdir / f"binary_{tag}_w{a.window}.jsonl"
    cache = {}
    if cf.exists():
        for ln in cf.open(encoding="utf-8"):
            d = json.loads(ln)
            cache[(d["t"], d["zone"], d["num"])] = d["resp"]
    T = range(len(rows) - a.n, len(rows))
    out = {}
    total, done, t0 = len(T) * 47, 0, time.time()
    for zone, X in (("front", F), ("back", B)):
        P = np.zeros((len(T), X.shape[1]))
        for i, t in enumerate(T):
            for j in range(X.shape[1]):
                key = (t, zone, j + 1)
                if key not in cache:
                    cache[key] = ask(a.host, a.model, prompt(X[t - a.window:t, j]))
                    with cf.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"t": t, "zone": zone, "num": j + 1, "resp": cache[key]}, ensure_ascii=False) + "\n")
                    if a.pause:
                        time.sleep(a.pause)
                done += 1
                v = parse(cache[key])
                P[i, j] = np.nan if v is None else v
            el = time.time() - t0
            print(f"  {zone} 第 {i + 1}/{len(T)} 期（{rows[t]['issue']}），进度 {done}/{total}，已用 {el / 60:.0f} 分钟", flush=True)
        out[zone] = (P, X[list(T)])
    rng = np.random.default_rng(0)
    lines = [f"# {a.model} · 0/1 序列直觉预测 · 大乐透最后 {a.n} 期（序列长度 {a.window}）\n",
             "| 区 | 有效回答 | 模型前 k 平均命中 | 随机期望 | 打乱对照 p | 近期频率基线命中 | 相对基础概率的信息（比特） | 频率基线（比特） |",
             "|---|---|---|---|---|---|---|---|"]
    for zone, k, base in (("front", 12, 5 / 35), ("back", 6, 2 / 12)):
        P, X = out[zone]
        valid = np.isfinite(P).mean()
        P = np.where(np.isfinite(P), P, base)
        h = topk_hits(P, X, k)
        null = np.array([topk_hits(np.array([rng.permutation(r) for r in P]), X, k).mean() for _ in range(2000)])
        pv = (1 + (null >= h.mean()).sum()) / 2001
        t_idx = list(range(len(rows) - a.n, len(rows)))
        src = F if zone == "front" else B
        Fq = np.array([(src[t - a.window:t].mean(0) * a.window + 1) / (a.window + 1 / base) for t in t_idx])
        hf = topk_hits(Fq + rng.random(Fq.shape) * 1e-6, X, k)
        exp_ = X.sum(1).mean() * k / X.shape[1]
        lines.append(f"| {'前区' if zone == 'front' else '后区'}（前 {k}） | {valid:.0%} | {h.mean():.3f} | {exp_:.3f} | {pv:.3f} | {hf.mean():.3f} | "
                     f"{bits(P, X, base):+.1f} | {bits(Fq, X, base):+.1f} |")
        uniq = len(np.unique(np.round(P, 3)))
        lines.append(f"|  | 模型给出的不同概率值：{uniq} 种；平均 {P.mean():.3f}（基础概率 {base:.3f}） | | | | | | |")
    lines += ["\n解读：命中高于随机期望且打乱对照 p<0.01、同时信息比特为正，才说明模型的“直觉”真有东西。",
              "信息比特为负表示模型给的概率比“每个号码都用基础概率”还差（通常是过度自信）。"]
    text = "\n".join(lines)
    print("\n" + text)
    (ROOT / "llm" / f"report_binary_{tag}.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
