"""吹气式摇奖机简化物理模型：球的质量/直径偏差 → 被捕获概率的敏感度。输出 report_aero.md。

模型（刻意简化，只求量级）：
  - 球形混合仓 半径 0.25 m；底部中心向上吹气，形成“中心上升、四周下降”的回流
  - 球：4 g、直径 45 mm 的实心泡沫球；受重力 + 二次气动阻力 F = ½ρ C_d A |u−v|(u−v)
  - 湍流：每个球感受到的气流 = 平均流场 + Ornstein-Uhlenbeck 随机脉动
  - 碰壁：法向恢复系数 0.5，切向保留 0.9
  - 出球：仓顶中心出球管口（半径 3 cm），球心进入且向上运动即被捕获，然后从仓底随机位置重新放入
  - 忽略球与球之间的碰撞（35 颗球的体积占仓体积约 3%）
各球被捕获近似为相互竞争的泊松过程：P_i ∝ λ_i（捕获率）。
关键量：弹性 e = d ln λ / d ln m —— 质量差 1% 导致捕获率差 e%。
同一组随机数驱动所有质量档（公共随机数），以降低比较的噪声。
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RHO, CD, G = 1.2, 0.47, 9.81
RC = 0.25  # 混合仓半径
TUBE = 0.03  # 出球管口半径
M0, D0 = 0.004, 0.045


def terminal_velocity(m=M0, d=D0):
    A = np.pi * (d / 2) ** 2
    return np.sqrt(2 * m * G / (RHO * CD * A))


def mean_flow(pos, U0):
    """轴对称回流：中心 r<0.35Rc 上升，外圈下降；顶部向外、底部向内"""
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    r = np.sqrt(x ** 2 + y ** 2) + 1e-9
    core = 0.35 * RC
    uz = U0 * (1 - (r / core) ** 2) * np.where(r < core, 1.0, 0.35)
    uz = np.where(r < core, uz, -0.35 * U0 * np.clip((r - core) / (RC - core), 0, 1))
    ur = 0.3 * U0 * (z / RC)  # 上部外扩、下部内收
    return np.stack([ur * x / r, ur * y / r, uz], axis=-1)


def simulate(mult_m, mult_d, U0, T=120.0, reps=2000, dt=2e-3, sigma=0.35, tau=0.05, seed=0):
    """mult_m / mult_d: 各档质量、直径倍数（同长度）。返回各档捕获率 λ（次/秒/球）"""
    rng = np.random.default_rng(seed)
    K = len(mult_m)
    m = M0 * np.asarray(mult_m)[:, None]
    d = D0 * np.asarray(mult_d)[:, None]
    k = (0.5 * RHO * CD * np.pi * (d / 2) ** 2 / m)[..., None]  # 阻力系数 /m，形状 (K,1,1)
    rad = d[..., None] / 2

    def spawn(n):
        p = rng.normal(size=(n, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        p *= (RC * 0.6) * rng.random((n, 1)) ** (1 / 3)
        p[:, 2] = -abs(p[:, 2]) - 0.05
        return p

    base = spawn(reps)
    pos = np.repeat(base[None], K, axis=0)
    vel = np.zeros_like(pos)
    turb = np.zeros((reps, 3))
    caps = np.zeros(K)
    a_ou = np.exp(-dt / tau)
    b_ou = sigma * U0 * np.sqrt(1 - a_ou ** 2)
    for _ in range(int(T / dt)):
        turb = a_ou * turb + b_ou * rng.normal(size=(reps, 3))  # 公共湍流
        u = mean_flow(pos, U0) + turb[None]
        rel = u - vel
        acc = k * np.linalg.norm(rel, axis=-1, keepdims=True) * rel
        acc[..., 2] -= G
        vel += acc * dt
        pos += vel * dt
        # 碰壁
        dist = np.linalg.norm(pos, axis=-1, keepdims=True)
        lim = RC - rad
        hit = dist > lim
        if hit.any():
            n = pos / dist
            vn = (vel * n).sum(-1, keepdims=True)
            vt = vel - vn * n
            vel = np.where(hit, 0.9 * vt - 0.5 * np.maximum(vn, 0) * n + np.minimum(vn, 0) * n, vel)
            pos = np.where(hit, n * lim, pos)
        # 出球
        r_xy = np.sqrt(pos[..., 0] ** 2 + pos[..., 1] ** 2)
        cap = (pos[..., 2] > RC - rad[..., 0] - 0.01) & (r_xy < TUBE) & (vel[..., 2] > 0)
        if cap.any():
            caps += cap.sum(1)
            kk, ii = np.nonzero(cap)
            pos[kk, ii] = spawn(len(kk))
            vel[kk, ii] = 0
    return caps / (reps * T)


def elasticity(mults, lam):
    x, y = np.log(mults), np.log(lam)
    return np.polyfit(x, y, 1)[0]


def main():
    vt = terminal_velocity()
    out = ["# 吹气式摇奖机简化模型：质量/直径敏感度\n",
           f"球：{M0 * 1000:.0f} g，直径 {D0 * 1000:.0f} mm，终端速度 v_t = {vt:.2f} m/s（气流需超过它才能托起球）\n",
           "弹性 e = d ln λ / d ln m：球重 1% 时捕获率变化 e%。\n",
           "| 中心风速 U0/v_t | 平均捕获率(次/秒/球) | 质量弹性 e_m | 直径弹性 e_d |",
           "|---|---|---|---|"]
    mults = np.array([0.94, 0.97, 1.0, 1.03, 1.06])
    ratios = (1.1, 1.3, 1.6, 2.0, 2.5, 3.5)
    jobs = [(mults, np.ones_like(mults), r * vt) for r in ratios] + [(np.ones_like(mults), mults, r * vt) for r in ratios]
    from multiprocessing import Pool
    with Pool() as pool:
        res = pool.starmap(simulate, jobs)
    rows = []
    for i, ratio in enumerate(ratios):
        lam_m, lam_d = res[i], res[len(ratios) + i]
        em, ed = elasticity(mults, lam_m), elasticity(mults, lam_d)
        rows.append((ratio, lam_m[2], em, ed))
        out.append(f"| {ratio:.1f} | {lam_m[2]:.3f} | {em:+.2f} | {ed:+.2f} |")
        print(out[-1], flush=True)

    # 反推：解释观测到的小号偏移需要多大的质量差
    tilt = 0.1539 / 0.1286 - 1  # 2022 年后：1-12 号单号出现率 0.1539，13-35 号约 0.1286（总和守恒）
    out += ["\n## 反推：要造成观测到的小号偏移，1-12 号球需要轻多少？\n",
            f"2022 年后 1-12 号与 13-35 号的单号出现率之比 ≈ {1 + tilt:.3f}（相对差 {tilt * 100:.1f}%）。\n",
            "| U0/v_t | 所需质量差（1-12 号比其余轻） | 约合 |",
            "|---|---|---|"]
    for ratio, lam, em, ed in rows:
        if em < -0.05:
            dm = np.expm1(np.log1p(tilt) / em)  # (1+dm)^em = 1+tilt
            out.append(f"| {ratio:.1f} | {-dm * 100:.1f}% | {-dm * M0 * 1000:.2f} g |")
        else:
            out.append(f"| {ratio:.1f} | 质量几乎不影响（e≈0），无法由质量差解释 | — |")
    text = "\n".join(out) + "\n"
    (ROOT / "report_aero.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
