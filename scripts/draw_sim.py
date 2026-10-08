"""完整开奖仿真：35 颗球 + 球间碰撞 + 搅拌 + 逐个出球；对不确定参数做拉丁超立方采样。输出 report_drawsim.md。

每次开奖：每颗球的质量、直径在公差内随机（质量 σ=5%、直径 σ=3%），先搅拌 T_MIX 秒，再打开出球口，
逐个捕获直到 5 颗。对全部模拟开奖拟合 Plackett-Luce 模型：
    P(第 k 个出球为 i | 剩余) ∝ exp(β_m·ln m_i + β_d·ln d_i)
β_m、β_d 就是“有效权重”对质量、直径的弹性，与 scripts/pl_model.py 从真实出球顺序估出的权重同一尺度，可直接对接：
    01-11 号的权重比 ×W ⇒ 需要 Δln m = ln W / β_m（或 Δln d = ln W / β_d）。
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parent.parent
RHO, CD, G = 1.2, 0.47, 9.81
RC = 0.25
M0, D0 = 0.004, 0.045
NB, PICK = 35, 5
SIG_M, SIG_D = 0.05, 0.03
T_MIX, T_MAX, DT = 6.0, 50.0, 1e-3
VT = np.sqrt(2 * M0 * G / (RHO * CD * np.pi * (D0 / 2) ** 2))

# 不确定参数及范围
PARAMS = {
    "ratio": (1.1, 3.0),  # 中心风速 / 终端速度
    "core": (0.25, 0.5),  # 上升气柱半径 / 仓半径
    "turb": (0.2, 0.6),  # 湍流强度 / 中心风速
    "tau": (0.02, 0.15),  # 湍流相关时间 s
    "e_wall": (0.3, 0.8),
    "e_ball": (0.3, 0.9),
    "tube": (0.025, 0.04),  # 出球口半径 m
}


def mean_flow(pos, U0, core):
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    r = np.sqrt(x ** 2 + y ** 2) + 1e-9
    c = core * RC
    up = U0 * (1 - (r / c) ** 2)
    down = -0.35 * U0 * np.clip((r - c) / (RC - c), 0, 1)
    uz = np.where(r < c, up, down)
    ur = 0.3 * U0 * (z / RC)
    return np.stack([ur * x / r, ur * y / r, uz], axis=-1)


from numba import njit, prange

PKEYS = ["ratio", "core", "turb", "tau", "e_wall", "e_ball", "tube"]


@njit(cache=True)
def _draw(lm, ld, pv, seed):
    """单次开奖（numba 编译）。pv 依次为 PKEYS。返回出球顺序（-1 表示超时）"""
    np.random.seed(seed)
    ratio, core, turbk, tau, e_wall, e_ball, tube = pv[0], pv[1], pv[2], pv[3], pv[4], pv[5], pv[6]
    U0 = ratio * VT
    c = core * RC
    m = np.exp(lm)
    rad = np.exp(ld) / 2
    kd = 0.5 * RHO * CD * np.pi * rad ** 2 / m
    pos = np.empty((NB, 3))
    vel = np.zeros((NB, 3))
    tb = np.zeros((NB, 3))
    for i in range(NB):  # 按号码顺序码放在仓底
        a = 2 * np.pi * i / NB
        pos[i, 0] = 0.12 * np.cos(a) + 0.003 * np.random.normal()
        pos[i, 1] = 0.12 * np.sin(a) + 0.003 * np.random.normal()
        pos[i, 2] = -RC + 0.06 + 0.03 * (i % 3) + 0.003 * np.random.normal()
    active = np.ones(NB, np.bool_)
    order = -np.ones(PICK, np.int64)
    npick = 0
    a_ou = np.exp(-DT / tau)
    b_ou = turbk * U0 * np.sqrt(1 - a_ou ** 2)
    nsteps = int(T_MAX / DT)
    nmix = int(T_MIX / DT)
    for step in range(nsteps):
        for i in range(NB):
            if not active[i]:
                continue
            x, y, z = pos[i, 0], pos[i, 1], pos[i, 2]
            r = np.sqrt(x * x + y * y) + 1e-9
            if r < c:
                uz = U0 * (1 - (r / c) ** 2)
            else:
                uz = -0.35 * U0 * min(max((r - c) / (RC - c), 0.0), 1.0)
            ur = 0.3 * U0 * (z / RC)
            u0 = ur * x / r
            u1 = ur * y / r
            for k in range(3):
                tb[i, k] = a_ou * tb[i, k] + b_ou * np.random.normal()
            r0 = u0 + tb[i, 0] - vel[i, 0]
            r1 = u1 + tb[i, 1] - vel[i, 1]
            r2 = uz + tb[i, 2] - vel[i, 2]
            nr = np.sqrt(r0 * r0 + r1 * r1 + r2 * r2)
            vel[i, 0] += kd[i] * nr * r0 * DT
            vel[i, 1] += kd[i] * nr * r1 * DT
            vel[i, 2] += (kd[i] * nr * r2 - G) * DT
            for k in range(3):
                pos[i, k] += vel[i, k] * DT
        # 球间碰撞
        for i in range(NB):
            if not active[i]:
                continue
            for j in range(i + 1, NB):
                if not active[j]:
                    continue
                d0 = pos[i, 0] - pos[j, 0]
                d1 = pos[i, 1] - pos[j, 1]
                d2 = pos[i, 2] - pos[j, 2]
                dist = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2) + 1e-12
                rs = rad[i] + rad[j]
                if dist < rs:
                    n0, n1, n2 = d0 / dist, d1 / dist, d2 / dist
                    vn = (vel[i, 0] - vel[j, 0]) * n0 + (vel[i, 1] - vel[j, 1]) * n1 + (vel[i, 2] - vel[j, 2]) * n2
                    if vn < 0:
                        J = -(1 + e_ball) * vn / (1 / m[i] + 1 / m[j])
                        vel[i, 0] += J * n0 / m[i]
                        vel[i, 1] += J * n1 / m[i]
                        vel[i, 2] += J * n2 / m[i]
                        vel[j, 0] -= J * n0 / m[j]
                        vel[j, 1] -= J * n1 / m[j]
                        vel[j, 2] -= J * n2 / m[j]
                    push = 0.5 * (rs - dist)
                    pos[i, 0] += push * n0
                    pos[i, 1] += push * n1
                    pos[i, 2] += push * n2
                    pos[j, 0] -= push * n0
                    pos[j, 1] -= push * n1
                    pos[j, 2] -= push * n2
        # 碰壁 + 出球
        cand = -1
        ncand = 0
        for i in range(NB):
            if not active[i]:
                continue
            dist0 = np.sqrt(pos[i, 0] ** 2 + pos[i, 1] ** 2 + pos[i, 2] ** 2)
            lim = RC - rad[i]
            if dist0 > lim:
                n0, n1, n2 = pos[i, 0] / dist0, pos[i, 1] / dist0, pos[i, 2] / dist0
                vn = vel[i, 0] * n0 + vel[i, 1] * n1 + vel[i, 2] * n2
                if vn > 0:
                    t0, t1, t2 = vel[i, 0] - vn * n0, vel[i, 1] - vn * n1, vel[i, 2] - vn * n2
                    vel[i, 0] = 0.9 * t0 - e_wall * vn * n0
                    vel[i, 1] = 0.9 * t1 - e_wall * vn * n1
                    vel[i, 2] = 0.9 * t2 - e_wall * vn * n2
                pos[i, 0], pos[i, 1], pos[i, 2] = n0 * lim, n1 * lim, n2 * lim
            if step >= nmix:
                rxy = np.sqrt(pos[i, 0] ** 2 + pos[i, 1] ** 2)
                if pos[i, 2] > RC - rad[i] - 0.01 and rxy < tube and vel[i, 2] > 0:
                    ncand += 1
                    if np.random.random() < 1.0 / ncand:  # 多个同时到达时等概率选一
                        cand = i
        if cand >= 0:
            order[npick] = cand
            npick += 1
            active[cand] = False
            if npick == PICK:
                break
    return order


@njit(parallel=True, cache=True)
def _batch(lm, ld, pv, seed0):
    R = lm.shape[0]
    out = np.empty((R, PICK), np.int64)
    for r in prange(R):
        out[r] = _draw(lm[r], ld[r], pv, seed0 + r)
    return out


def run_batch(par, R, seed):
    """R 次独立开奖。返回 (ln m, ln d, 出球顺序[R,5]，-1 表示超时未出)"""
    rng = np.random.default_rng(seed)
    lm = np.log(M0) + SIG_M * rng.normal(size=(R, NB))
    ld = np.log(D0) + SIG_D * rng.normal(size=(R, NB))
    pv = np.array([par[k] for k in PKEYS])
    return lm, ld, _batch(lm, ld, pv, seed * 100003)


def fit_pl(lm, ld, order):
    """条件 logit / Plackett-Luce：协变量为中心化的 ln m、ln d。返回 β 与标准误"""
    X = np.stack([lm - lm.mean(1, keepdims=True), ld - ld.mean(1, keepdims=True)], -1)  # (R,NB,2)
    ok = (order >= 0).all(1)
    X, order = X[ok], order[ok]
    R = len(order)

    def nll(beta):
        s = X @ beta  # (R,NB)
        avail = np.ones((R, NB), bool)
        v, g = 0.0, np.zeros(2)
        H = np.zeros((2, 2))
        for k in range(PICK):
            e = np.where(avail, np.exp(s - s.max(1, keepdims=True)), 0)
            pr = e / e.sum(1, keepdims=True)
            pk = order[:, k]
            v -= (s[np.arange(R), pk] - s.max(1) - np.log(e.sum(1))).sum()
            mx = (pr[..., None] * X).sum(1)
            g -= (X[np.arange(R), pk] - mx).sum(0)
            dx = X - mx[:, None, :]
            H += np.einsum("rn,rni,rnj->ij", pr, dx, dx)
            avail[np.arange(R), pk] = False
        return v, g, H

    r = minimize(lambda b: nll(b)[:2], np.zeros(2), jac=True, method="BFGS")
    _, _, H = nll(r.x)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return r.x, se, R


def run_config(args):
    idx, par, R, batches = args
    res = [run_batch(par, R, seed=1000 * idx + b) for b in range(batches)]
    lm = np.concatenate([x[0] for x in res])
    ld = np.concatenate([x[1] for x in res])
    order = np.concatenate([x[2] for x in res])
    beta, se, n = fit_pl(lm, ld, order)
    timeout = float((order < 0).any(1).mean())
    picks = order[(order >= 0).all(1)].ravel()
    mem = float((picks < 11).mean() / (11 / NB))  # 码放位置记忆：01-11（码放在前的球）被抽中比例 / 理论
    print(f"config {idx}: {par} -> β_m={beta[0]:+.2f}±{se[0]:.2f} β_d={beta[1]:+.2f}±{se[1]:.2f} n={n} 超时={timeout:.2f} 码放记忆={mem:.3f}", flush=True)
    return idx, par, beta, se, n, timeout, mem


def main():
    n_cfg = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 192
    batches = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    lhs = qmc.LatinHypercube(d=len(PARAMS), seed=42).random(n_cfg)
    keys = list(PARAMS)
    cfgs = []
    for i, u in enumerate(lhs):
        par = {k: PARAMS[k][0] + u[j] * (PARAMS[k][1] - PARAMS[k][0]) for j, k in enumerate(keys)}
        cfgs.append((i, par, R, batches))
    results = [run_config(c) for c in cfgs]

    # 对接真实数据：01-11 号相对权重（pl_model.py 全样本 M1）
    W, W_lo, W_hi = 1.112, 1.044, 1.180
    out = ["# 完整开奖仿真：35 球 + 碰撞 + 不确定参数采样\n",
           f"每个参数组合模拟 {R * batches} 次开奖；每次开奖内球的质量 σ={SIG_M:.0%}、直径 σ={SIG_D:.0%} 随机。",
           f"β = 有效权重对质量/直径的弹性（Plackett-Luce 拟合）。终端速度 v_t={VT:.2f} m/s。\n",
           "| # | 风速/v_t | 气柱 | 湍流 | τ | 壁恢复 | 球恢复 | 出球口 | β_m | β_d | 超时率 | 码放记忆 | 01-11 需轻 | 或需大 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    bm_all, bd_all = [], []
    mems = []
    for idx, par, beta, se, n, to, mem in results:
        mems.append(mem)
        bm, bd = beta
        need_m = -np.expm1(np.log(W) / bm) if bm < -0.1 else np.nan
        need_d = np.expm1(np.log(W) / bd) if bd > 0.1 else np.nan
        bm_all.append(bm)
        bd_all.append(bd)
        out.append(f"| {idx} | {par['ratio']:.2f} | {par['core']:.2f} | {par['turb']:.2f} | {par['tau']:.3f} | "
                   f"{par['e_wall']:.2f} | {par['e_ball']:.2f} | {par['tube'] * 100:.1f}cm | {bm:+.2f}±{se[0]:.2f} | "
                   f"{bd:+.2f}±{se[1]:.2f} | {to:.2f} | ×{mem:.3f} | "
                   + (f"{need_m:.1%} ({need_m * M0 * 1000:.2f} g)" if np.isfinite(need_m) else "—") + " | "
                   + (f"{need_d:.1%} ({need_d * D0 * 1000:.1f} mm)" if np.isfinite(need_d) else "—") + " |")
    bm_all, bd_all = np.array(bm_all), np.array(bd_all)
    # 不确定性合成：参数采样 × 真实数据权重的置信区间
    rng = np.random.default_rng(0)
    Ws = np.exp(rng.normal(np.log(W), (np.log(W_hi) - np.log(W_lo)) / 3.92, 20000))
    bms = rng.choice(bm_all, 20000)
    bds = rng.choice(bd_all, 20000)
    nm = -np.expm1(np.log(Ws) / np.where(bms < -0.1, bms, np.nan))
    nd = np.expm1(np.log(Ws) / np.where(bds > 0.1, bds, np.nan))
    q = lambda a: np.nanpercentile(a, [10, 50, 90])
    out += ["\n## 合成结论\n",
            f"质量弹性 β_m：中位数 {np.median(bm_all):+.2f}，范围 [{bm_all.min():+.2f}, {bm_all.max():+.2f}]",
            f"直径弹性 β_d：中位数 {np.median(bd_all):+.2f}，范围 [{bd_all.min():+.2f}, {bd_all.max():+.2f}]\n",
            f"真实数据：01-11 号有效权重 ×{W}（95% CI {W_lo}–{W_hi}）。合成参数不确定性与数据不确定性后：\n",
            "| 解释 | 10% 分位 | 中位数 | 90% 分位 |", "|---|---|---|---|",
            "| 01-11 号球更轻 | " + " | ".join(f"{x:.1%} ({x * M0 * 1000:.2f} g)" for x in q(nm)) + " |",
            "| 01-11 号球更大 | " + " | ".join(f"{x:.1%} ({x * D0 * 1000:.1f} mm)" for x in q(nd)) + " |",
            f"\n码放位置记忆（搅拌 {T_MIX:.0f} 秒后，码放在前的 01-11 号被抽中比例 / 理论）：中位数 ×{np.median(mems):.3f}，"
            f"范围 [×{min(mems):.3f}, ×{max(mems):.3f}]。真实数据为 ×{W}。"]
    text = "\n".join(out) + "\n"
    (ROOT / "report_drawsim.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
