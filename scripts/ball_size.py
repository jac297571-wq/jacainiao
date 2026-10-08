"""从开奖视频的出球槽特写中测量每颗球的像素直径，按颜色比较。输出 report_ballsize.md。

方法：
  1. video_timing.py 得到后区最后一球的时刻 T；在 T+5、T+9、T+13 秒各取一帧特写。
  2. 出球槽区域内霍夫圆检测，取最显著的 5 个圆，按 x 从左到右排序 = 出球顺序（与开奖数据核对）。
  3. 对每个圆，在半径 ±25% 的环带内取 Canny 边缘点做最小二乘圆拟合，得到亚像素半径。
  4. 每帧内用 5 球的中位数做归一化（消除镜头远近差异），比较各颜色的相对直径。
用法：python3 scripts/ball_size.py <视频目录> <timing.jsonl>
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
VDIR, TIMING = Path(sys.argv[1]), Path(sys.argv[2])
COLORS = ["蓝", "黑", "红", "黄", "绿"]
ORDERS = {}
for r in csv.DictReader((ROOT / "data" / "dlt_ballset.csv").open(encoding="utf-8")):
    toks = r["order"].split()
    if len(toks) >= 5 and all(t.isdigit() for t in toks[:5]):
        ORDERS[r["date"]] = (r["issue"], list(map(int, toks))[:5], r["ballset"])


def tag_to_date(tag):
    t = tag if len(tag) == 8 else "20" + tag
    return f"{t[:4]}-{t[4:6]}-{t[6:8]}"


def refine(gray, x, y, r):
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    ys, xs = np.nonzero(edges)
    d = np.hypot(xs - x, ys - y)
    m = (d > 0.75 * r) & (d < 1.25 * r)
    if m.sum() < 30:
        return None
    X, Y = xs[m].astype(float), ys[m].astype(float)
    # 代数圆拟合 + 一轮离群剔除
    for _ in range(2):
        A = np.column_stack([X, Y, np.ones_like(X)])
        b = X ** 2 + Y ** 2
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = c[0] / 2, c[1] / 2
        rr = np.sqrt(c[2] + cx ** 2 + cy ** 2)
        res = np.abs(np.hypot(X - cx, Y - cy) - rr)
        keep = res < 2.5
        X, Y = X[keep], Y[keep]
    return rr, len(X)


def measure(frame):
    roi_y0, roi_y1, roi_x0, roi_x1 = 440, 680, 80, 760
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sub = cv2.medianBlur(gray[roi_y0:roi_y1, roi_x0:roi_x1], 5)
    circles = cv2.HoughCircles(sub, cv2.HOUGH_GRADIENT, dp=1.2, minDist=55, param1=120, param2=25, minRadius=28, maxRadius=55)
    if circles is None:
        return None
    c = circles[0][:5]
    if len(c) < 5:
        return None
    c = c[np.argsort(c[:, 0])]
    xs = c[:, 0]
    if np.any(np.diff(xs) < 50) or np.ptp(c[:, 1]) > 60:  # 5 球应横向排开、高度相近
        return None
    out = []
    for x, y, r in c:
        f = refine(gray, x + roi_x0, y + roi_y0, r)
        if f is None:
            return None
        out.append((x + roi_x0, y + roi_y0, f[0], f[1]))
    return out


rows = []
for line in TIMING.read_text().splitlines():
    d = json.loads(line)
    fr, bk = d["front"], d["back"]
    if None in fr or None in bk or min(np.diff(fr)) < 2:
        continue
    tag = d["file"][:-4]
    date = tag_to_date(tag)
    if date not in ORDERS:
        continue
    issue, order, bset = ORDERS[date]
    for k, dt in enumerate((5, 7, 9, 11, 13)):
        img = VDIR / f"_m_{tag}_{k}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(bk[1] + dt), "-i", str(VDIR / d["file"]), "-frames:v", "1", str(img)])
        frame = cv2.imread(str(img))
        if frame is None:
            continue
        m = measure(frame)
        if m is None:
            continue
        rs = np.array([x[2] for x in m])
        med = np.median(rs)
        for pos, ((x, y, r, npts), num) in enumerate(zip(m, order)):
            rows.append({"issue": issue, "set": bset, "frame": k, "pos": pos, "num": num, "color": (num - 1) // 7,
                         "r": r, "rel": r / med, "x": x, "npts": npts})
        if k == 2:
            vis = frame.copy()
            for (x, y, r, _), num in zip(m, order):
                cv2.circle(vis, (int(x), int(y)), int(r), (0, 255, 255), 2)
                cv2.putText(vis, f"{num:02d}", (int(x) - 15, int(y) - int(r) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imwrite(str(VDIR / f"_annot_{issue}.jpg"), vis)

out = ["# 摇奖球像素直径测量（出球槽特写）\n"]
if not rows:
    out.append("没有成功测量的帧。")
else:
    issues = sorted({r["issue"] for r in rows})
    out.append(f"成功测量 {len(issues)} 期、{len({(r['issue'], r['frame']) for r in rows})} 帧、{len(rows)} 个球次。\n")
    rel = np.array([r["rel"] for r in rows])
    col = np.array([r["color"] for r in rows])
    pos = np.array([r["pos"] for r in rows])
    # 位置效应（槽内左右透视）：先去掉每个位置的平均，再比较颜色
    adj = rel.copy()
    for p in range(5):
        adj[pos == p] -= adj[pos == p].mean() - 1
    out.append("| 颜色 | 球次 | 相对直径（去位置效应） | 95% CI | 相对其余颜色 |")
    out.append("|---|---|---|---|---|")
    for c in range(5):
        a = adj[col == c]
        if len(a) == 0:
            continue
        # 以“期”为单位聚类的稳健标准误：同一期多帧高度相关
        per_issue = {}
        for r_, v in zip([r for r in rows if r["color"] == c], a):
            per_issue.setdefault(r_["issue"], []).append(v)
        means = np.array([np.mean(v) for v in per_issue.values()])
        se = means.std(ddof=1) / np.sqrt(len(means)) if len(means) > 1 else np.nan
        others = adj[col != c].mean()
        out.append(f"| {COLORS[c]} | {len(a)} | {a.mean():.4f} | ±{1.96 * se:.4f} | {(a.mean() / others - 1) * 100:+.2f}% |")
    # 蓝 vs 其余：以期为单位配对比较（同一帧内既有蓝又有非蓝时）
    diffs = []
    for iss in issues:
        sub = [(r, v) for r, v in zip(rows, adj) if r["issue"] == iss]
        b = [v for r, v in sub if r["color"] == 0]
        o = [v for r, v in sub if r["color"] != 0]
        if b and o:
            diffs.append(np.mean(b) - np.mean(o))
    if len(diffs) > 2:
        t = stats.ttest_1samp(diffs, 0)
        out.append(f"\n**蓝球 vs 同期其他颜色**（配对，{len(diffs)} 期）：平均相对直径差 {np.mean(diffs) * 100:+.2f}%，t={t.statistic:+.2f}，p={t.pvalue:.3f}")
    out.append(f"\n测量精度：同一颗球在同一期不同帧之间的相对直径标准差 ≈ "
               f"{np.mean([np.std([r['rel'] for r in rows if r['issue'] == i and r['pos'] == p]) for i in issues for p in range(5) if sum(1 for r in rows if r['issue'] == i and r['pos'] == p) > 1]) * 100:.2f}%")
    with (ROOT / "data" / "ball_size_measurements.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
text = "\n".join(out) + "\n"
(ROOT / "report_ballsize.md").write_text(text, encoding="utf-8")
print(text)
