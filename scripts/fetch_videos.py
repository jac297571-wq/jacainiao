"""从新浪“体彩开奖直播”页面批量下载开奖视频，并截取大乐透段的缩略图网格。

用法：python3 scripts/fetch_videos.py <链接列表文件(每行: 日期 URL)> <输出目录> [最多几个]
"""
import json, re, subprocess, sys, time, urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0", "Referer": "http://video.sina.com.cn/"}
API = ("https://api.ivideo.sina.com.cn/public/video/play?video_id={}&appver=V11220.191231.02&appname=sinaplayer_pc"
       "&applt=web&tags=sinaplayer_pc&player=all&jsonp=&plid=2020012001&prid=&uid=&tid=&pid=1&ran=0.5&r=video.sina.com.cn")


def get(url, binary=False, tries=6, timeout=60):
    for i in range(tries):
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()
            return data if binary else data.decode("utf-8", "ignore")
        except Exception as e:
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed: {url}")


def main(listfile, outdir, limit=20):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    lines = [l.split() for l in Path(listfile).read_text().splitlines() if l.strip()]
    done = 0
    for date, url in reversed(lines):
        if done >= limit:
            break
        page = get(url)
        title = (re.findall(r"title:'([^']+)'", page) or [""])[0]
        vid = (re.findall(r"video_id:(\d+)", page) or [None])[0]
        if "体彩开奖" not in title or not vid:
            continue
        tag = title.replace("体彩开奖直播", "").strip() or date
        mp4 = out / f"{tag}.mp4"
        if not mp4.exists():
            try:
                meta = json.loads(get(API.format(vid)))
                src = meta["data"]["videos"][0]["dispatch_result"]["url"]
                # curl 断点续传 + 重试，CDN 经常断开连接
                tmp = mp4.with_suffix(".part")
                for _ in range(12):
                    r = subprocess.run(["curl", "-sS", "-L", "-C", "-", "-m", "300", "-A", UA["User-Agent"],
                                        "-H", f"Referer: {UA['Referer']}", "-o", str(tmp), src])
                    if r.returncode == 0:
                        break
                    time.sleep(3)
                if r.returncode != 0:
                    raise RuntimeError("curl failed")
                tmp.rename(mp4)
            except Exception as e:
                print(date, tag, "下载失败，跳过：", e, flush=True)
                continue
        grid = out / f"{tag}_grid.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "300", "-i", str(mp4), "-vf",
                        "fps=1/8,scale=256:-1,tile=6x6", "-frames:v", "1", str(grid)], check=False)
        print(date, tag, title, mp4.stat().st_size, flush=True)
        done += 1


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20)
