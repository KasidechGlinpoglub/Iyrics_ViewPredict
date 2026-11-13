"""เก็บข้อมูลเทรนจาก YouTube (ใช้ YouTube Data API v3 ถ้ามี YOUTUBE_API_KEY, ไม่งั้นใช้ yt-dlp)

1) ดึงรายชื่อเพลงจาก playlist (เช่น playlist เพลงฮิต) -> ได้ title, artist, views, url และช่อง lyrics ว่างให้กรอก
     python scripts/collect_youtube.py playlist "https://www.youtube.com/playlist?list=..." -o data/raw_songs.csv
   (ต้องมี YOUTUBE_API_KEY, ใช้โควตาแค่ 1 หน่วยต่อ 50 เพลง)

2) กรอกเนื้อเพลงในคอลัมน์ lyrics ของ data/raw_songs.csv

3) เติม/อัปเดตยอดวิวล่าสุด แล้วบันทึกเป็นไฟล์สำหรับเทรน (ข้ามแถวที่ยังไม่มีเนื้อเพลง)
     python scripts/collect_youtube.py fill data/raw_songs.csv -o data/songs.csv

4) เทรนใหม่:  python train.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import load_env  # noqa: E402
from core.youtube import YouTubeClient, YouTubeError, extract_video_id  # noqa: E402

COLUMNS = ["title", "artist", "lyrics", "views", "url"]


def get_client(required: bool) -> YouTubeClient | None:
    load_env()
    key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if key:
        return YouTubeClient(key)
    if required:
        sys.exit("ต้องตั้งค่า YOUTUBE_API_KEY ในไฟล์ .env ก่อน (ดู README)")
    return None


def fetch_api(client: YouTubeClient, urls: list[str]) -> dict[str, dict]:
    ids = {u: extract_video_id(u) for u in urls}
    infos = client.videos([v for v in ids.values() if v])
    return {u: infos[v] for u, v in ids.items() if v in infos}


def fetch_ytdlp(urls: list[str], delay: float) -> dict[str, dict]:
    try:
        import yt_dlp
    except ImportError:
        sys.exit("ไม่มี YOUTUBE_API_KEY และไม่ได้ติดตั้ง yt-dlp — ตั้งค่า key หรือรัน: pip install yt-dlp")
    out = {}
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}) as ydl:
        for i, url in enumerate(urls, 1):
            try:
                info = ydl.extract_info(url, download=False)
            except Exception as e:  # วิดีโอถูกลบ/เป็นส่วนตัว ฯลฯ
                print(f"[{i}/{len(urls)}] ข้าม {url}: {e}")
                continue
            out[url] = {
                "views": info.get("view_count"),
                "title": info.get("track") or info.get("title") or "",
                "channel": info.get("artist") or info.get("channel") or info.get("uploader") or "",
                "url": info.get("webpage_url") or url,
            }
            time.sleep(delay)
    return out


def cmd_playlist(args) -> None:
    client = get_client(required=True)
    ids = client.playlist_video_ids(args.playlist, limit=args.limit)
    infos = client.videos(ids)
    rows = [
        {"title": v["title"], "artist": v["channel"], "lyrics": "", "views": v["views"], "url": v["url"]}
        for v in (infos.get(i) for i in ids) if v and v["views"] is not None
    ]
    df = pd.DataFrame(rows, columns=COLUMNS)
    if args.output.exists() and not args.overwrite:  # รวมกับไฟล์เดิม ไม่ทับเนื้อเพลงที่กรอกไว้แล้ว
        old = pd.read_csv(args.output, encoding="utf-8-sig")
        df = pd.concat([old, df[~df["url"].isin(old.get("url", pd.Series(dtype=str)))]], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"ได้ {len(rows)} เพลงจาก playlist -> {args.output} (รวม {len(df)} แถว)")
    print("ขั้นต่อไป: กรอกเนื้อเพลงในคอลัมน์ lyrics แล้วรันคำสั่ง fill")


def cmd_fill(args) -> None:
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"url", "lyrics"} <= set(df.columns):
        sys.exit("ไฟล์ input ต้องมีคอลัมน์ url และ lyrics")
    for col in ("title", "artist", "views"):
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype(object)  # คอลัมน์ว่างทั้งคอลัมน์ถูกอ่านเป็นตัวเลข ใส่ข้อความไม่ได้
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    has_lyrics = df["lyrics"].fillna("").astype(str).str.strip() != ""
    skipped = int((~has_lyrics).sum())
    df = df[has_lyrics & (df["url"] != "")].reset_index(drop=True)

    client = get_client(required=False)
    urls = df["url"].tolist()
    print(f"ดึงยอดวิว {len(urls)} เพลง ด้วย {'YouTube Data API' if client else 'yt-dlp'}...")
    try:
        meta = fetch_api(client, urls) if client else fetch_ytdlp(urls, args.delay)
    except YouTubeError as e:
        sys.exit(f"YouTube API: {e}")

    for i, url in enumerate(urls):
        m = meta.get(url)
        if not m:
            print(f"  ไม่พบวิดีโอ: {url}")
            continue
        df.at[i, "views"] = m["views"]
        df.at[i, "url"] = m["url"]
        if pd.isna(df.at[i, "title"]) or not str(df.at[i, "title"]).strip():
            df.at[i, "title"] = m["title"]
        if pd.isna(df.at[i, "artist"]) or not str(df.at[i, "artist"]).strip():
            df.at[i, "artist"] = m["channel"]

    df = df.dropna(subset=["views"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"บันทึก {len(df)} เพลง -> {args.output}" + (f" (ข้าม {skipped} แถวที่ยังไม่มีเนื้อเพลง)" if skipped else ""))
    print("ขั้นต่อไป: python train.py")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("playlist", help="ดึงรายชื่อเพลง + ยอดวิวจาก playlist")
    p.add_argument("playlist", help="ลิงก์หรือ ID ของ playlist")
    p.add_argument("-o", "--output", type=Path, default=Path("data/raw_songs.csv"))
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--overwrite", action="store_true", help="เขียนทับไฟล์เดิมแทนการรวม")
    p.set_defaults(func=cmd_playlist)

    f = sub.add_parser("fill", help="เติมยอดวิวล่าสุดให้ไฟล์ที่มี url + lyrics")
    f.add_argument("input", type=Path)
    f.add_argument("-o", "--output", type=Path, default=Path("data/songs.csv"))
    f.add_argument("--delay", type=float, default=1.0, help="หน่วงเวลาระหว่างเพลงเมื่อใช้ yt-dlp (วินาที)")
    f.set_defaults(func=cmd_fill)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
