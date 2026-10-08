"""从开奖视频中提取每颗球的出球时刻：检测画面下方号码气泡由空心变为实心蓝色的时刻。

用法：python3 scripts/video_timing.py <视频文件...>
输出：每个视频一行 JSON：前区 5 球、后区 2 球的出现时刻（秒）。
"""
import json
import sys

import cv2
import numpy as np

FRONT_X = [120, 197, 274, 352, 428]
BACK_X = [755, 832]
Y = 528
R = 14


def filled(frame, x):
    patch = frame[Y - R:Y + R, x - R:x + R].astype(int)
    b, g, r = patch[..., 0].mean(), patch[..., 1].mean(), patch[..., 2].mean()
    return b > 150 and b - r > 60  # 实心蓝色气泡（前区）


def filled_back(frame, x):
    patch = frame[Y - R:Y + R, x - R:x + R].astype(int)
    b, g, r = patch[..., 0].mean(), patch[..., 1].mean(), patch[..., 2].mean()
    return r > 150 and g > 120 and r - b > 40  # 后区实心为黄色


def empty(frame, x):
    patch = frame[Y - R:Y + R, x - R:x + R].astype(int)
    b, r = patch[..., 0].mean(), patch[..., 2].mean()
    return abs(b - r) < 45 and b < 200  # 空心气泡：透出背景，偏灰


def scan(path, t0=200, t1=760, step=0.2):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    times = {"front": [None] * 5, "back": [None] * 2, "start": None}
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    t = t0
    nxt = t0
    while t < t1:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if t < nxt:
            continue
        nxt = t + step
        if times["start"] is None:
            if all(empty(frame, x) for x in FRONT_X) and all(empty(frame, x) for x in BACK_X):
                times["start"] = round(t, 2)  # 摇奖画面出现、尚未出球
            continue
        for i, x in enumerate(FRONT_X):
            if times["front"][i] is None and filled(frame, x):
                # 要求后续所有前序气泡都已点亮，避免误检
                if all(times["front"][j] is not None for j in range(i)):
                    times["front"][i] = round(t, 2)
        for i, x in enumerate(BACK_X):
            if times["back"][i] is None and times["front"][4] is not None and filled_back(frame, x):
                if all(times["back"][j] is not None for j in range(i)):
                    times["back"][i] = round(t, 2)
        if times["back"][1] is not None:
            break
    return times


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps({"file": p, **scan(p)}, ensure_ascii=False), flush=True)
