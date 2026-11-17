"""เก็บข้อมูลเทรนจาก YouTube (ใช้ YouTube Data API v3 ถ้ามี YOUTUBE_API_KEY, ไม่งั้นใช้ yt-dlp)

1) ดึงรายการคลิปจากช่องหรือ playlist (ต้องมี YOUTUBE_API_KEY)
     python scripts/collect_youtube.py channels "@genelab" "@smallroommusic" -o data/raw_songs.csv

2) ตรวจว่าเป็น MV ทางการ ยืนยันชื่อเพลง/ศิลปิน แล้วค้นเนื้อร้องจาก LRCLIB
     python scripts/collect_youtube.py lyrics data/raw_songs.csv

3) เติม/อัปเดตยอดวิวล่าสุด แล้วบันทึกเป็นไฟล์สำหรับเทรน (ข้ามแถวที่ยังไม่มีเนื้อเพลง)
     python scripts/collect_youtube.py fill data/raw_songs.csv -o data/songs.csv

4) เทรนใหม่:  python train.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import load_env  # noqa: E402
from core import lrclib  # noqa: E402
from core.youtube import YouTubeClient, YouTubeError, extract_video_id  # noqa: E402

COLUMNS = [
    "video_id", "title", "artist", "channel", "channel_id", "video_title",
    "lyrics", "lyrics_source", "published_at", "views_collected_at", "views", "url",
]


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
            fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            upload_date = info.get("upload_date")
            out[url] = {
                "video_id": info.get("id") or extract_video_id(url),
                "views": info.get("view_count"),
                "views_collected_at": fetched_at if info.get("view_count") is not None else "",
                "title": info.get("title") or "",
                "track": info.get("track") or "",
                "artist": info.get("artist") or "",
                "channel": info.get("channel") or info.get("uploader") or "",
                "channel_id": info.get("channel_id") or "",
                "description": info.get("description") or "",
                "published_at": datetime.strptime(upload_date, "%Y%m%d").date().isoformat() if upload_date else "",
                "url": info.get("webpage_url") or url,
            }
            time.sleep(delay)
    return out


def video_row(video: dict) -> dict:
    # ใช้เฉพาะเครดิตที่ระบุโดยตรง ชื่อช่องของค่ายอาจรวมศิลปินหลายคน
    matches: dict[str, set[str]] = {}
    for line in video.get("description", "").splitlines():
        match = re.fullmatch(r"(artist|ศิลปิน|song|track|เพลง|ชื่อเพลง)\s*[:：]\s*(.+)", line.strip(), re.IGNORECASE)
        if match:
            field = "artist" if match[1].lower() in {"artist", "ศิลปิน"} else "title"
            matches.setdefault(field, set()).add(match[2].strip())
    credits = {field: values.pop() for field, values in matches.items() if len(values) == 1}
    return {
        "video_id": video["video_id"],
        "title": video.get("track") or credits.get("title") or video["title"],
        "artist": video.get("artist") or credits.get("artist", ""),
        "channel": video["channel"],
        "channel_id": video["channel_id"],
        "video_title": video["title"],
        "published_at": video["published_at"],
        "views_collected_at": video["views_collected_at"],
        "views": video["views"],
        "url": video["url"],
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        columns = list(dict.fromkeys([*COLUMNS, *df.columns]))
        df.reindex(columns=columns).to_csv(tmp_path, index=False, encoding="utf-8-sig")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def collect_videos(client: YouTubeClient, ids: list[str], output: Path, overwrite: bool) -> None:
    ids = list(dict.fromkeys(ids))
    old = pd.DataFrame()
    if output.exists() and not overwrite:
        old = pd.read_csv(output, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if "url" not in old:
            sys.exit(f"ไฟล์เดิม {output} ขาดคอลัมน์ url")
        # รองรับไฟล์เก่าที่มีเฉพาะ URL รวมถึงลิงก์ youtu.be
        old_ids = old["url"].map(extract_video_id)
        if "video_id" in old:
            old_ids = old["video_id"].where(old["video_id"].str.strip() != "", old_ids)
        seen = set(old_ids.dropna())
        ids = [video_id for video_id in ids if video_id not in seen]
    infos = client.videos(ids)
    rows = [video_row(infos[video_id]) for video_id in ids if video_id in infos]
    new = pd.DataFrame(rows, dtype=object)
    df = pd.concat([old, new], ignore_index=True)
    write_csv(df, output)
    unavailable = len(ids) - len(rows)
    missing_views = sum(row["views"] is None for row in rows)
    print(f"เพิ่ม {len(rows)} คลิป -> {output} (รวม {len(df)} แถว)")
    if unavailable or missing_views:
        print(f"เข้าถึงไม่ได้ {unavailable} คลิป · ไม่มียอดวิว {missing_views} คลิป")
    print("ขั้นต่อไป: ตรวจคลิป ชื่อเพลง และศิลปิน แล้วเติมเนื้อร้อง")


def cmd_playlist(args) -> None:
    client = get_client(required=True)
    ids = client.playlist_video_ids(args.playlist, limit=args.limit)
    collect_videos(client, ids, args.output, args.overwrite)


def cmd_channels(args) -> None:
    client = get_client(required=True)
    playlists = dict.fromkeys(client.uploads_playlist(channel) for channel in args.channels)
    ids = []
    for playlist in playlists:
        ids.extend(client.playlist_video_ids(playlist, limit=args.limit))
    collect_videos(client, ids, args.output, args.overwrite)


def cmd_lyrics(args) -> None:
    df = pd.read_csv(args.input, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [col.strip().lower() for col in df.columns]
    if not df.columns.is_unique:
        sys.exit("ไฟล์ input มีชื่อคอลัมน์ซ้ำ")
    if not {"title", "artist"} <= set(df.columns):
        sys.exit("ไฟล์ input ต้องมีคอลัมน์ title และ artist")
    for col in ("lyrics", "lyrics_source", "lyrics_status"):
        if col not in df:
            df[col] = ""

    queries = 0
    error = None
    with requests.Session() as session:
        for i, row in df.iterrows():
            if row["lyrics"].strip():
                df.at[i, "lyrics_status"] = "matched" if row["lyrics_status"] == "matched" else "existing"
                continue
            title, artist = row["title"].strip(), row["artist"].strip()
            if not title or not artist:
                df.at[i, "lyrics_status"] = "needs_review"
                continue
            if queries:
                time.sleep(0.3)
            queries += 1
            try:
                status, lyrics, source = lrclib.match_lyrics(title, artist, lrclib.search(session, title, artist))
            except lrclib.LRCLIBError as exc:
                df.at[i, "lyrics_status"] = "lookup_error"
                error = exc
                break
            df.loc[i, ["lyrics_status", "lyrics", "lyrics_source"]] = [status, lyrics, source]

    output = args.output or args.input
    write_csv(df, output)
    counts = df["lyrics_status"].replace("", "pending").value_counts()
    print(f"ค้น LRCLIB {queries} เพลง -> {output}")
    for status, count in counts.items():
        print(f"  {status}: {count}")
    if error:
        sys.exit(f"บันทึกผลที่ทำได้แล้ว · {error}")


def cmd_fill(args) -> None:
    df = pd.read_csv(args.input, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"url", "lyrics"} <= set(df.columns):
        sys.exit("ไฟล์ input ต้องมีคอลัมน์ url และ lyrics")
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["url"] = df["url"].str.strip()
    has_lyrics = df["lyrics"].str.strip() != ""
    skipped = int((~has_lyrics).sum())
    df = df[has_lyrics & (df["url"] != "")].reset_index(drop=True)

    client = get_client(required=False)
    urls = df["url"].tolist()
    print(f"ดึงยอดวิว {len(urls)} เพลง ด้วย {'YouTube Data API' if client else 'yt-dlp'}...")
    meta = (fetch_api(client, urls) if client else fetch_ytdlp(urls, args.delay)) if urls else {}

    refreshed = []
    for i, url in enumerate(urls):
        m = meta.get(url)
        if not m or m["views"] is None:
            print(f"  ไม่พบยอดวิว: {url}")
            continue
        row = video_row(m)
        for col, value in row.items():
            if value not in ("", None) and (col not in {"title", "artist"} or not df.at[i, col].strip()):
                df.at[i, col] = str(value)
        refreshed.append(i)

    df = df.loc[refreshed]
    write_csv(df, args.output)
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

    c = sub.add_parser("channels", help="ดึงรายการอัปโหลดจากช่อง YouTube โดยไม่จำกัดอายุคลิป")
    c.add_argument("channels", nargs="+", help="@handle, channel ID หรือลิงก์ของแต่ละช่อง")
    c.add_argument("-o", "--output", type=Path, default=Path("data/raw_songs.csv"))
    c.add_argument("--limit", type=int, help="จำนวนคลิปสูงสุดต่อช่อง (ค่าเริ่มต้นคือทั้งหมด)")
    c.add_argument("--overwrite", action="store_true", help="เขียนทับไฟล์เดิมแทนการรวม")
    c.set_defaults(func=cmd_channels)

    l = sub.add_parser("lyrics", help="เติมเนื้อร้องจาก LRCLIB ด้วยชื่อเพลงและศิลปิน")
    l.add_argument("input", type=Path)
    l.add_argument("-o", "--output", type=Path, help="ไฟล์ผลลัพธ์ (ค่าเริ่มต้นคือไฟล์ input)")
    l.set_defaults(func=cmd_lyrics)

    f = sub.add_parser("fill", help="เติมยอดวิวล่าสุดให้ไฟล์ที่มี url + lyrics")
    f.add_argument("input", type=Path)
    f.add_argument("-o", "--output", type=Path, default=Path("data/songs.csv"))
    f.add_argument("--delay", type=float, default=1.0, help="หน่วงเวลาระหว่างเพลงเมื่อใช้ yt-dlp (วินาที)")
    f.set_defaults(func=cmd_fill)

    args = ap.parse_args()
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        ap.error("--limit ต้องมากกว่า 0")
    try:
        args.func(args)
    except YouTubeError as e:
        sys.exit(f"YouTube API: {e}")


if __name__ == "__main__":
    main()
