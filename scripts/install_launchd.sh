#!/bin/bash
# 在 Mac 上安装三个定时任务（launchd，按本机时间；Mac 在日本 = JST）。
#   大乐透 周一/三/六 23:47 JST（北京 22:47）  双色球 周二/四/日 23:37 JST（北京 22:37）  日本 周一至五 22:51 JST
# 合盖睡眠时到点不会跑，唤醒后 launchd 会补跑一次。
# 手机推送（可选）：NTFY_TOPIC=一个只有你知道的名字 bash scripts/install_launchd.sh，手机装 ntfy App 订阅同名主题。
# 卸载：bash scripts/install_launchd.sh uninstall
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$(command -v python3)"
DIR="$HOME/Library/LaunchAgents"
mkdir -p "$DIR" "$REPO/logs"
plist() {  # $1=名字 $2=参数 $3=小时 $4=分钟 $5...=星期（0=周日）
  local name=$1 arg=$2 h=$3 m=$4; shift 4
  local f="$DIR/com.jacainiao.$name.plist" cal=""
  for d in "$@"; do cal="$cal<dict><key>Weekday</key><integer>$d</integer><key>Hour</key><integer>$h</integer><key>Minute</key><integer>$m</integer></dict>"; done
  launchctl unload "$f" 2>/dev/null || true
  [ "$UNINSTALL" = 1 ] && { rm -f "$f"; echo "已卸载 $name"; return; }
  cat > "$f" <<P
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>com.jacainiao.$name</string>
<key>ProgramArguments</key><array><string>$PY</string><string>$REPO/routine_local.py</string><string>$arg</string></array>
<key>WorkingDirectory</key><string>$REPO</string>
<key>EnvironmentVariables</key><dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string><key>NTFY_TOPIC</key><string>${NTFY_TOPIC}</string></dict>
<key>StartCalendarInterval</key><array>$cal</array>
<key>StandardOutPath</key><string>$REPO/logs/launchd_$name.log</string>
<key>StandardErrorPath</key><string>$REPO/logs/launchd_$name.log</string>
</dict></plist>
P
  launchctl load "$f"
  echo "已安装 $name"
}
[ "$1" = uninstall ] && UNINSTALL=1
plist dlt dlt 23 47 1 3 6
plist ssq ssq 23 37 0 2 4
plist jp jp 22 51 1 2 3 4 5
