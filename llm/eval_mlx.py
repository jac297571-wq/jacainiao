"""评测 MLX 模型（可带 LoRA 适配器）在留出的最后 N 期上的预测能力，尺子与 llm_eval.py / 贝叶斯模型相同。

每期把最近 30 期发给模型，按温度 1 采样 S 次"下一期"，统计每个号码被采到的次数作为权重
（加 α 个伪计数做收缩，避免采样噪声被当成信号），再算实际开奖组合的头奖概率倍数。
用法：
  python llm/eval_mlx.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --game 大乐透                         # 微调前
  python llm/eval_mlx.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --adapter llm/adapters_dlt --game 大乐透  # 微调后
  加 --pause 0.5 可降低 GPU 占用（每次生成后暂停 0.5 秒），电脑仍可流畅使用，只是更慢
"""
import argparse
import re
import sys
import time
from math import comb, log
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deep"))
sys.path.insert(0, str(ROOT / "llm"))
from dynpl import set_logp  # noqa: E402
from make_data import GAMES, W, rows_for, user_msg  # noqa: E402

ALPHA = 20.0  # 每个号码的伪计数
NULL_REPS = 20  # 同采样次数的随机采样器重复次数（公平及格线）


def sampler_fn(model_id, adapter):
    from mlx_lm import generate, load
    model, tok = load(model_id, adapter_path=adapter) if adapter else load(model_id)
    try:
        from mlx_lm.sample_utils import make_sampler
        kw = {"sampler": make_sampler(temp=1.0)}
    except ImportError:
        kw = {"temp": 1.0}

    def sample(text):
        prompt = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
        return generate(model, tok, prompt=prompt, max_tokens=24, verbose=False, **kw)
    return sample


def parse(text, K, M):
    nums = []
    for s in re.findall(r"\d+", text):
        n = int(s)
        if 1 <= n <= K and n not in nums:
            nums.append(n)
    return nums[:M]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--game", default="大乐透", choices=list(GAMES))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--pause", type=float, default=0.0, help="每次生成后暂停的秒数，给 GPU 留出空闲（例如 0.5）")
    a = ap.parse_args()
    rows, ld, K, M = rows_for(a.game)
    sample = sampler_fn(a.model, a.adapter)
    base = -log(comb(K, M))
    gains, nulls, hits, valid = [], [], [], 0
    rng = np.random.default_rng(0)
    for i, t in enumerate(range(len(rows) - a.n, len(rows))):
        c = np.full(K, ALPHA)
        msg = user_msg(a.game, K, M, rows[t - W:t])
        for _ in range(a.samples):
            nums = parse(sample(msg), K, M)
            if a.pause:
                time.sleep(a.pause)
            valid += len(nums) == M
            for n in nums:
                c[n - 1] += 1
        p = c / c.sum()
        s = np.array(rows[t]) - 1
        gains.append((set_logp(np.log(p), s) - base) / log(2))
        ng = []
        for _ in range(NULL_REPS):  # 一个什么都没学到、但采样次数相同的"模型"
            cn = np.full(K, ALPHA)
            for _ in range(a.samples):
                cn[rng.choice(K, M, replace=False)] += 1
            ng.append((set_logp(np.log(cn / cn.sum()), s) - base) / log(2))
        nulls.append(np.mean(ng))
        hits.append(np.isin(np.argsort(-p)[:M], s).sum())
        print(f"  {i + 1}/{a.n}  本期 {2 ** gains[-1]:.3f}  累计 ×{2 ** np.mean(gains):.4f}", flush=True)
    g = np.array(gains)
    z = g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))
    d = g - np.array(nulls)
    zd = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    name = a.model + (f" + {a.adapter}" if a.adapter else "")
    text = (f"# {name} · {a.game} · 留出最后 {a.n} 期\n\n"
            f"| 预测者 | 总增益（比特） | 头奖概率倍数 | z |\n|---|---|---|---|\n"
            f"| 纯随机（理论） | +0.00 | ×1.0000 | — |\n"
            f"| 随机采样器（同样 {a.samples} 次采样，公平及格线） | {np.sum(nulls):+.2f} | ×{2 ** np.mean(nulls):.4f} | — |\n"
            f"| {name} | {g.sum():+.2f} | ×{2 ** g.mean():.4f} | {z:+.2f} |\n\n"
            f"**相对公平及格线：×{2 ** d.mean():.4f}，z={zd:+.2f}**（z>2 才算模型真的学到了东西）\n\n"
            f"格式正确的回答比例 {valid / (a.n * a.samples):.0%}；最看好的 {M} 个号码平均命中 {np.mean(hits):.3f}（随机 {M * M / K:.3f}）。")
    print("\n" + text)
    tag = "lora" if a.adapter else "base"
    (ROOT / "llm" / f"report_{ld}_{tag}.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
