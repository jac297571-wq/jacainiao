"""大乐透历史数据分析：随机性检验 + 冷热号回测 + 一等奖撞号（分奖）分析"""
import csv, random, statistics as st
from collections import Counter
from math import comb
from pathlib import Path

D = list(csv.DictReader((Path(__file__).resolve().parent.parent / "data" / "dlt_history.csv").open(encoding="utf-8")))
R = [sorted(int(d[f"r{i}"]) for i in range(1, 6)) for d in D]
B = [sorted(int(d[f"b{i}"]) for i in (1, 2)) for d in D]
N = len(D)
TOTAL = comb(35, 5) * comb(12, 2)


def chi2(counter, k, n_obs):
    e = n_obs / k
    return sum((counter.get(i, 0) - e) ** 2 / e for i in range(1, k + 1))


print(f"# 样本：{N} 期（{D[0]['issue']} ~ {D[-1]['issue']}）\n")
print(f"## 1. 一等奖概率：1 / {TOTAL:,}")
print(f"   每周 3 期、每期买 1 注，期望约 {TOTAL / 156:,.0f} 年中一次\n")

# 2. 频率卡方检验
rc = Counter(x for r in R for x in r)
bc = Counter(x for b in B for x in b)
x2r, x2b = chi2(rc, 35, 5 * N), chi2(bc, 12, 2 * N)
print("## 2. 号码频率卡方检验（看号码是否均匀）")
print(f"   前区 χ²={x2r:.1f}（自由度34，5%临界值 48.6）")
print(f"   后区 χ²={x2b:.1f}（自由度11，5%临界值 19.7）")
print(f"   前区最热 {rc.most_common(3)}  最冷 {rc.most_common()[-3:]}")
print(f"   后区最热 {bc.most_common(3)}  最冷 {bc.most_common()[-3:]}")
print("   分时期前区 χ²（期望值≈34，>48.6 即显著不均匀）：")
for y0, y1 in [(2007, 2011), (2012, 2016), (2017, 2021), (2022, 2026)]:
    idx = [i for i, d in enumerate(D) if y0 <= int(d["date"][:4]) <= y1]
    c = Counter(x for i in idx for x in R[i])
    print(f"     {y0}-{y1}: {len(idx)}期 χ²={chi2(c, 35, 5 * len(idx)):.1f}  最热 {[k for k, _ in c.most_common(4)]}")
print()


# 3. 冷热号策略回测：用过去 W 期统计选号，看下一期命中数
def hits(pick_r, pick_b, i):
    return len(set(pick_r) & set(R[i])), len(set(pick_b) & set(B[i]))


rng = random.Random(42)
W = 100
START = next(i for i, d in enumerate(D) if d["date"] >= "2017")
for label, lo in [("全部年份", W), ("仅2017年后", START)]:
  res = {"热号": [], "冷号": [], "随机": []}
  for i in range(lo, N):
    c = Counter(x for r in R[i - W:i] for x in r)
    cb = Counter(x for b in B[i - W:i] for x in b)
    order = sorted(range(1, 36), key=lambda x: -c[x])
    border = sorted(range(1, 13), key=lambda x: -cb[x])
    res["热号"].append(hits(order[:5], border[:2], i))
    res["冷号"].append(hits(order[-5:], border[-2:], i))
    res["随机"].append(hits(rng.sample(range(1, 36), 5), rng.sample(range(1, 13), 2), i))
  n = N - lo
  se = (5 * 5 / 35 * 30 / 35 * 30 / 34 / n) ** 0.5
  print(f"## 3. 冷热号回测·{label}（窗口{W}期，{n}期；理论期望 前区 0.714±{se:.3f} / 后区 0.333）")
  for k, v in res.items():
      print(f"   {k}: 前区平均命中 {st.mean(a for a, _ in v):.3f}  后区 {st.mean(b for _, b in v):.3f}")
  print()

# 4. 一等奖撞号：哪类号码中奖人数多（=奖金被瓜分）
print("## 4. 一等奖注数 vs 号码特征（每亿元销量的一等奖注数，越高=越多人买同一组号）")
groups = {}
for d, r in zip(D, R):
    if not d["sales"] or not d["first_n"]:
        continue
    rate = int(d["first_n"]) / int(d["sales"]) * 1e8
    key = "全部≤31（像生日）" if max(r) <= 31 else "含32-35"
    groups.setdefault(key, []).append(rate)
    consec = sum(1 for a, b in zip(r, r[1:]) if b - a == 1)
    groups.setdefault("有≥2对连号" if consec >= 2 else "连号≤1对", []).append(rate)
    groups.setdefault("和值<70（号码偏小）" if sum(r) < 70 else "和值≥70", []).append(rate)
for k, v in groups.items():
    print(f"   {k:<16} 期数{len(v):>5}  均值 {st.mean(v):.2f}  中位数 {st.median(v):.2f}")
top = sorted(((int(d['first_n']), d['issue'], r, B[i]) for i, (d, r) in enumerate(zip(D, R)) if d['first_n']), reverse=True)[:8]
print("   一等奖注数最多的几期：")
for n, iss, r, b in top:
    print(f"     {iss}: {n:>3} 注  前区 {r} 后区 {b}")
print()

# 5. 返奖：一等奖实际分到多少
print("## 5. 近 3 年一等奖单注奖金")
recent = [int(d["first_prize"]) for d in D[-468:] if d["first_prize"] and int(d["first_n"] or 0) > 0]
print(f"   中位数 {st.median(recent):,.0f} 元  最低 {min(recent):,} 元  最高 {max(recent):,} 元")
