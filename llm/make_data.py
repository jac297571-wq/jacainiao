"""生成 MLX LoRA 微调数据：每条样本 = 最近 W 期开奖（用户消息）→ 下一期开奖（助手回答）。

严格前瞻：最后 N 期（默认 100）不进入训练，留给 eval_mlx.py 评测；其余按时间切出最后 10% 作验证集。
大乐透只用 2014 年之后（与当前机器一致）。
用法：python llm/make_data.py --game 大乐透   → llm/data_dlt/{train,valid}.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import radar  # noqa: E402

GAMES = {"大乐透": ("dlt", 35, 5), "双色球": ("ssq", 33, 6), "ロト6": ("loto6", 43, 6),
         "ロト7": ("loto7", 37, 7), "ミニロト": ("miniloto", 31, 5)}
W = 30


def rows_for(game):
    ld, K, M = GAMES[game]
    data = radar.load(ld)
    if ld == "dlt":
        data = [d for d in data if d[0] >= "14001"]
    return [sorted(r) for _, r in data], ld, K, M


def fmt(r):
    return " ".join(f"{x:02d}" for x in r)


def user_msg(game, K, M, hist):
    return (f"{game}最近 {len(hist)} 期开奖（从旧到新，每期从 1–{K} 中开出 {M} 个号码）：\n"
            + "\n".join(fmt(r) for r in hist) + "\n下一期：")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="大乐透", choices=list(GAMES))
    ap.add_argument("--holdout", type=int, default=100)
    a = ap.parse_args()
    rows, ld, K, M = rows_for(a.game)
    end = len(rows) - a.holdout
    ex = [{"messages": [{"role": "user", "content": user_msg(a.game, K, M, rows[t - W:t])},
                        {"role": "assistant", "content": fmt(rows[t])}]} for t in range(W, end)]
    cut = int(len(ex) * 0.9)
    out = ROOT / "llm" / f"data_{ld}"
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", ex[:cut]), ("valid", ex[cut:])):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for e in part:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"{a.game}：训练 {cut} 条，验证 {len(ex) - cut} 条，留出最后 {a.holdout} 期评测 → {out}")


if __name__ == "__main__":
    main()
