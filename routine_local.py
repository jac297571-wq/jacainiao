"""本机例行任务：不需要大模型，按开奖日依次跑抓数、核对、选号、雷达、撞号模型，然后提交推送并弹通知。

用法（仓库根目录）：
  python3 routine_local.py dlt        大乐透（周一/三/六 21:25 开奖）
  python3 routine_local.py ssq        双色球（周二/四/日 21:15 开奖）
  python3 routine_local.py jp         日本（ロト6 周一/四，ミニロト 周二，ロト7 周五；ナンバーズ周一至周五）
  python3 routine_local.py all        按今天的日期把该跑的都跑一遍（适合补跑）
选项：--no-push 不推送；--no-wait 数据没更新时不等待重试
每次的完整输出存在 logs/routine_<日期>_<彩种>.log，简短汇报弹 macOS 通知；
设置环境变量 NTFY_TOPIC 时，同时推送到手机（ntfy.sh 上的这个主题，手机装 ntfy App 订阅同名主题）。
安装定时任务：bash scripts/install_launchd.sh
"""
import csv
import datetime as dt
import os
import platform
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
LOGS = ROOT / "logs"
CN = dt.timezone(dt.timedelta(hours=8))
JP = dt.timezone(dt.timedelta(hours=9))
RETRIES, WAIT = 4, 900


class Run:
    def __init__(self, name):
        self.name = name
        self.out = []
        self.summary = []

    def sh(self, *cmd, keep=None):
        """跑一条命令，输出写进日志；keep 是正则，匹配的行进简短汇报"""
        r = subprocess.run([PY, *cmd] if cmd[0].endswith(".py") else list(cmd), cwd=ROOT, capture_output=True, text=True)
        text = r.stdout + (("\n[stderr]\n" + r.stderr) if r.returncode else "")
        self.out.append(f"$ {' '.join(cmd)}\n{text}")
        if r.returncode:
            self.summary.append(f"⚠️ {cmd[0]} 出错")
        if keep:
            self.summary += [ln.strip() for ln in r.stdout.splitlines() if re.search(keep, ln)]
        return r.stdout

    def wait_for(self, fetch, ok, what, no_wait):
        for i in range(RETRIES + 1):
            for f in fetch:
                self.sh(*f)
            if ok():
                return True
            if no_wait or i == RETRIES:
                break
            self.out.append(f"{what} 还没更新，{WAIT // 60} 分钟后重试（{i + 1}/{RETRIES}）")
            time.sleep(WAIT)
        self.summary.append(f"⚠️ {what} 数据未更新，没有生成下一期")
        return False


def last_csv(path, col):
    rows = list(csv.DictReader((ROOT / path).open(encoding="utf-8")))
    return rows[-1][col] if rows else ""


def ssq_last_date():
    lines = [ln.split() for ln in (ROOT / "data/other/ssq.txt").open(encoding="utf-8", errors="ignore") if ln.strip()]
    return lines[-1][1] if lines else ""


def dlt(no_wait):
    R = Run("大乐透")
    today = dt.datetime.now(CN).date().isoformat()
    if not R.wait_for([("scripts/fetch.py",), ("scripts/fetch_ballset.py",)],
                      lambda: last_csv("data/dlt_history.csv", "date") == today, "大乐透", no_wait):
        return R
    R.sh("cifulucky.py", "check", keep=r"^\d{5}: 开奖")
    R.summary.append("A 轨上期：" + "、".join(re.findall(r"前\d\+后\d", R.out[-1].split("开奖")[-1])[:5]))
    R.sh("cifulucky.py", "watch")
    R.sh("cifulucky.py", "sprt", keep=r"^累计")
    R.sh("cifulucky.py", "pick", keep=r"合计头奖概率|^\s+[1-5]\. ")
    R.sh("radar.py", keep=r"证明档告警")
    R.sh("popularity.py", "track", keep=r"大乐透：|合计")
    R.sh("popularity.py", "checkb", "大乐透", keep=r"B 轨")
    R.sh("popularity.py", "pick", "大乐透", keep=r"期望值|^\s+B:")
    R.sh("pool_pick.py", "check", keep=r"收敛选号第|等$|等\s*$")
    R.sh("pool_pick.py", "pick", keep=r"^前区池|注（合计")
    return R


def ssq(no_wait):
    R = Run("双色球")
    today = dt.datetime.now(CN).date().isoformat()
    # ssq_lucky.py pick --update 同时负责抓数和选号；数据没到时先只抓数等待
    for i in range(RETRIES + 1):
        R.sh("ssq_lucky.py", "pick", "--update", keep=r"合计头奖概率|^\s+[1-5]\. |前瞻检验对象")
        if ssq_last_date() == today:
            break
        if no_wait or i == RETRIES:
            R.summary = ["⚠️ 双色球数据未更新（以上选号基于旧数据，已有文件不会被覆盖）"]
            return R
        time.sleep(WAIT)
    R.sh("ssq_lucky.py", "check", keep=r"^\d{7}: 开奖")
    R.sh("ssq_lucky.py", "sprt", keep=r"^已观察")
    R.sh("radar.py", keep=r"证明档告警|^双色球")
    R.sh("popularity.py", "track", keep=r"双色球：|合计")
    R.sh("popularity.py", "checkb", "双色球", keep=r"B 轨")
    R.sh("popularity.py", "pick", "双色球", keep=r"期望值|^\s+B:")
    return R


JP_GAMES = {0: ["loto6"], 1: ["miniloto"], 3: ["loto6"], 4: ["loto7"]}
JP_NAMES = {"loto6": "ロト6", "loto7": "ロト7", "miniloto": "ミニロト"}


def jp(no_wait):
    R = Run("日本")
    now = dt.datetime.now(JP)
    today = now.date().isoformat()
    games = JP_GAMES.get(now.weekday(), [])
    if games:
        if not R.wait_for([("japan/scripts/fetch_loto.py",)],
                          lambda: all(last_csv(f"japan/data/{g}.csv", "date") == today for g in games), "日本ロト", no_wait):
            games = []
    else:
        R.sh("japan/scripts/fetch_loto.py")
    for g in games:
        R.sh("japan/jp_lucky.py", g, "check", keep=r"^第\d+回：开奖")
        R.summary.append("A 轨上回：" + "、".join(re.findall(r"中 \d 个", R.out[-1].split("开奖")[-1])[:5]))
        R.sh("japan/jp_lucky.py", g, "pick", keep=r"合计头奖概率|^\s+[1-5]\. ")
        if g != "loto6":
            R.sh("japan/jp_lucky.py", g, "sprt", keep=r"^已观察")
    R.sh("radar.py", keep=r"证明档告警")
    nw = R.sh("numbers_watch.py", "--update")
    alarms = [ln for ln in nw.splitlines() if ln.startswith("ナンバーズ") and "无告警" not in ln]
    R.summary = ([f"🚨 {ln.strip()}" for ln in alarms] or ["ナンバーズ无告警"]) + R.summary
    for g in games:
        R.sh("popularity.py", "track", keep=JP_NAMES[g] + "：|合计")
        R.sh("popularity.py", "checkb", JP_NAMES[g], keep=r"B 轨")
        R.sh("popularity.py", "pick", JP_NAMES[g], keep=r"期望值|^\s+B:")
    R.quiet = not games and not alarms
    return R


TAG = {"大乐透": "DLT", "双色球": "SSQ", "日本": "Japan"}


def notify(title, body):
    if platform.system() == "Darwin":
        s = body.replace('"', "'")[:240]
        subprocess.run(["osascript", "-e", f'display notification "{s}" with title "{title}"'])
    topic = os.environ.get("NTFY_TOPIC")
    if topic:
        req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=body.encode(), headers={"Title": "lottery " + TAG.get(title, "")})
        try:
            urllib.request.urlopen(req, timeout=30)
        except Exception as e:  # 推送失败不影响主流程
            print("ntfy 推送失败：", e)


def finish(R, push):
    LOGS.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    log = LOGS / f"routine_{stamp}_{R.name}.log"
    report = "\n".join(R.summary)
    log.write_text(report + "\n\n" + "\n".join(R.out), encoding="utf-8")
    print(f"===== {R.name} =====\n{report}\n（完整输出：{log.relative_to(ROOT)}）")
    subprocess.run(["git", "add", "-A", "--", ".", ":!logs"], cwd=ROOT)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        subprocess.run(["git", "commit", "-qm", f"{R.name} {dt.date.today()} 开奖核对与选号（本机例行）"], cwd=ROOT)
        if push:
            subprocess.run(["git", "push", "-q"], cwd=ROOT)
    if not getattr(R, "quiet", False):
        notify(R.name, report)


def main():
    args = sys.argv[1:]
    which = next((a for a in args if not a.startswith("--")), "all")
    push, no_wait = "--no-push" not in args, "--no-wait" in args
    subprocess.run(["git", "pull", "-q", "--ff-only"], cwd=ROOT)
    todo = {"dlt": [dlt], "ssq": [ssq], "jp": [jp]}.get(which)
    if todo is None:
        wd = dt.datetime.now(CN).weekday()
        todo = [jp] + ([dlt] if wd in (0, 2, 5) else []) + ([ssq] if wd in (1, 3, 6) else [])
    for f in todo:
        finish(f(no_wait), push)


if __name__ == "__main__":
    main()
