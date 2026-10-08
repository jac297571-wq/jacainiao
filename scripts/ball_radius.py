"""1080p 出球槽特写中精确测量球半径：霍夫圆给初值 → 沿上半圆 60 条射线找最强边缘 → 稳健圆拟合。

用法（模块）：measure(frame_bgr) -> [(x, y, r, 残差像素)] 按 x 从左到右，最多 5 个。
"""
import cv2
import numpy as np

ROI = (600, 1050, 100, 1200)  # y0, y1, x0, x1（1920x1080 画面中的出球槽区域）


def _hough(gray):
    y0, y1, x0, x1 = ROI
    sub = cv2.medianBlur(gray[y0:y1, x0:x1], 5)
    c = cv2.HoughCircles(sub, cv2.HOUGH_GRADIENT, dp=1.0, minDist=95, param1=110, param2=28, minRadius=48, maxRadius=80)
    if c is None:
        return []
    c = c[0]
    c[:, 0] += x0
    c[:, 1] += y0
    return c


def _ray_fit(grad, x, y, r):
    pts = []
    for a in np.linspace(np.pi * 1.05, np.pi * 1.95, 60):  # 上半圆（图像坐标 y 向下，取 sin<0 的一侧）
        rs = np.arange(0.75 * r, 1.25 * r, 0.25)
        px = x + rs * np.cos(a)
        py = y + rs * np.sin(a)
        ok = (px >= 1) & (px < grad.shape[1] - 2) & (py >= 1) & (py < grad.shape[0] - 2)
        if ok.sum() < 10:
            continue
        g = cv2.remap(grad, px[ok].astype(np.float32).reshape(1, -1), py[ok].astype(np.float32).reshape(1, -1),
                      cv2.INTER_LINEAR).ravel()
        k = int(np.argmax(g))
        if g[k] < 25:
            continue
        pts.append((px[ok][k], py[ok][k]))
    if len(pts) < 20:
        return None
    P = np.array(pts)
    for _ in range(3):
        A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
        b = (P ** 2).sum(1)
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = c[0] / 2, c[1] / 2
        rr = np.sqrt(c[2] + cx ** 2 + cy ** 2)
        res = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - rr)
        thr = max(1.5, 2.5 * np.median(res))
        if (res < thr).sum() < 15:
            break
        P = P[res < thr]
    return cx, cy, rr, float(np.median(res))


def measure(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 用颜色梯度（三通道最大值）比灰度更能分开蓝球和蓝色背景
    gx = np.max([np.abs(cv2.Sobel(frame[..., k].astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)) for k in range(3)], 0)
    gy = np.max([np.abs(cv2.Sobel(frame[..., k].astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)) for k in range(3)], 0)
    grad = np.hypot(gx, gy).astype(np.float32)
    seeds = _hough(gray.astype(np.uint8))
    out = []
    for x, y, r in seeds:
        f = _ray_fit(grad, x, y, r)
        if f is not None:
            out.append(f)
    out.sort(key=lambda t: t[0])
    return out


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        im = cv2.imread(p)
        m = measure(im)
        print(p, [(round(x), round(y), round(r, 2), round(e, 2)) for x, y, r, e in m])
        vis = im.copy()
        for x, y, r, e in m:
            cv2.circle(vis, (int(x), int(y)), int(round(r)), (0, 255, 255), 2)
        cv2.imwrite(p.replace(".png", "_fit.jpg"), vis[600:1050, 100:1200])
