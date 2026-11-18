"""สร้าง CSV เพลงไทยจริงจากเนื้อร้องที่เผยแพร่ในคำอธิบาย MV ของ YouTube"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unicodedata import normalize

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.youtube import YouTubeError, _video_info  # noqa: E402
from scripts.collect_youtube import get_client, video_row, write_csv  # noqa: E402

# ขนาดรายการอัปโหลดที่ตรวจต่อช่องในรอบนี้ กำหนดก่อนดึงยอดวิว
CHANNELS = {
    "@genelab": 500, "UC1T-ZwcYVQB6EpqOQD10tmw": 500, "@yuppofficial": 500,
    "@TangBadVoice": 150, "@safeplanetband": 150, "@timethaiofficial": 200,
    "@whattheduckmusic": 550, "@spicydisc": 450, "@genierock": 400,
    "@grammygoldofficial": 300, "@HighCloudEntertainment": 200,
    "@yewband": 100, "@ZweednRoll": 120, "UCz15_ns8kgNkCAkagNjEjBg": 180,
    "UCk-hrmSLwSsp4OE_s5f1_Sg": 350,
}
ARTIST_CHANNELS = {
    "@TangBadVoice", "@safeplanetband", "@timethaiofficial", "@yewband",
    "@ZweednRoll", "UCz15_ns8kgNkCAkagNjEjBg",
}
THAI_TIME = timezone(timedelta(hours=7))
TITLE_ARTIST_CREDITS = {"CViaEbGKVqw": ("ชีวิตคืออะไร? (Winter Is Coming)", "THE TOYS")}
THAI_LETTER = re.compile(r"[ก-ฮ]")
MV_MARK = re.compile(r"official\s*(?:music\s*video|m/?v)\b|music\s+video\b|[\[(]\s*m/?v\s*[\])]", re.I)
VIDEO_CREDIT = re.compile(r"(?im)^[^\w\n]*(?:directed by\b|director\s*[:：]|music video (?:by|credits?)\b)")
NON_MV = re.compile(
    r"\b(?:teaser|trailer|shorts|cover|remix|reaction|visualizer|medley|unplugged|acoustic|"
    r"instrumental|karaoke version|original by|(?:official )?(?:lyrics?|audio) video|official (?:lyrics?|audio)|"
    r"live (?:session|at|version|performance)|dance (?:practice|performance)|behind.the.scenes|sped up)\b",
    re.I,
)
LYRICS_HEADING = re.compile(r"[\W_]*(?:lyrics?|เนื้อเพลง|เนื้อร้อง)[\W_]*", re.I)
SECTION = re.compile(
    r"[\W_]*(?:(?:pre[\s-]?)?chorus|verse|intro|outro|hook|bridge|rap|refrain|ท่อนฮุก|ฮุก|ซ้ำ)"
    r"\s*\d*[\W_]*", re.I,
)
SECTION_PREFIX = re.compile(r"^[\[(](?:(?:pre[\s-]?)?chorus|verse|intro|outro|hook|bridge|rap)\b[^\])]*[\])]\s*", re.I)
AFTER_LYRICS = re.compile(
    r"^(?:[#©℗]|(?:credits?|(?:music|video|song|production|special thanks|cast|starring)\s+credits?)\b|"
    r"(?:executive\s+)?(?:producer|director|arranger|composer|artist|songwriter|editor|cast|"
    r"written by|produced by|directed by|music video by|mixed by|mastered by|"
    r"lyrics by|music by|wedding video by|กิตติกรรมประกาศ|คำร้อง|ทำนอง|เรียบเรียง|ผู้กำกับ)\s*(?:[:：]|\b)|"
    r"(?:(?:lead|backing)\s+)?(?:vocals?|guitars?|bass|drums?|drum editor|piano|keys?|keyboards?|"
    r"synthesizers?|arrange|music|lyrics?|lyrics director|producers|ig|pr|fanpage|show|"
    r"band manager|recorded|recording engineer|mixed|mastered)\b[^:\n]{0,28}[:：]|"
    r"[^:\n]{1,45}\s+members?\s*[:：]|"
    r"follow us\b|follow\s+@|ติดตาม(?:ข่าวสาร|ผลงาน|ศิลปิน|ได้ที่|ทุกช่องทาง|ความเคลื่อนไหว|\s)|"
    r"(?:available (?:on|now)|all streaming|streaming|download|show booking|for work|"
    r"facebook|instagram|tiktok|booking|contact|website|bandcamp|spotify)\s*(?:[:：]|\b))",
    re.I,
)
LINK = re.compile(r"https?://|www\.|mailto:|\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I)
SEPARATOR = re.compile(r"[\s_\-–—=━─▂▃•*]{3,}")
SHORT_MARK = re.compile(r"\[\d{1,3}:\d{2}(?:[.:]\d+)?\]")
FEATURE = re.compile(r"\s+(?:feat\.?|ft\.?)\s+.*$", re.I)


def key(text: str) -> str:
    return "".join(c for c in normalize("NFKC", text).casefold() if c.isalnum() or "\u0e31" <= c <= "\u0e4e")


def extract_lyrics(description: str) -> str:
    lines = [line.strip() for line in description.splitlines()]
    starts = [i for i, line in enumerate(lines) if LYRICS_HEADING.fullmatch(normalize("NFKC", line))]
    if len(starts) != 1:
        return ""
    lyrics = []
    for line in lines[starts[0] + 1:]:
        normalized = normalize("NFKC", line)
        if not line or SECTION.fullmatch(normalized):
            continue
        if SEPARATOR.fullmatch(normalized):
            if lyrics:
                break
            continue
        if AFTER_LYRICS.search(normalized.lstrip(" [【(*•▶▷►")) or LINK.search(normalized):
            break
        line = SHORT_MARK.sub("", SECTION_PREFIX.sub("", line)).strip(" *")
        if line and any(c.isalpha() for c in line):
            lyrics.append(line)
    text = "\n".join(lyrics)
    thai_letters = len(THAI_LETTER.findall(text))
    all_letters = sum(c.isalpha() for c in text)
    if len(lyrics) < 8 or len({key(line) for line in lyrics}) < 6 or thai_letters < 100:
        return ""
    return text if thai_letters >= all_letters * 0.55 else ""


def song_metadata(video: dict, source: str) -> dict:
    video["views_collected_at"] = ""
    song = video_row(video)
    original_title = normalize("NFC", song["video_title"])
    prefix = MV_MARK.split(original_title, maxsplit=1)[0].rstrip(" |[-(【")
    prefix = re.split(r"\s+\|\s+", prefix, maxsplit=1)[0]
    sides = re.split(r"\s+[-–—]\s*|(?<=\S)[-–—]\s+", prefix, maxsplit=1)
    if len(sides) == 2:
        left, right = (part.strip(" '\"‘’“”") for part in sides)
        channel = key(video["channel"])
        artist_at_right = (
            source in ARTIST_CHANNELS and channel
            and key(right).startswith(channel) and not key(left).startswith(channel)
        )
        if source == "@genelab":
            title, artist = left, right
        elif song["artist"] and key(left) != key(song["artist"]) and key(right) == key(song["artist"]):
            title, artist = left, right
        elif artist_at_right:
            title, artist = left, right
        elif source == "@whattheduckmusic" and THAI_LETTER.search(left) and not THAI_LETTER.search(right):
            title, artist = left, right
        else:
            title, artist = right, left
        if song["title"] == song["video_title"]:
            song["title"] = re.sub(r"\s*\(prod\.?\s+by\s+[^)]+\)\s*$", "", title, flags=re.I)
        if not song["artist"]:
            song["artist"] = artist
    if video["video_id"] in TITLE_ARTIST_CREDITS:
        song["title"], song["artist"] = TITLE_ARTIST_CREDITS[video["video_id"]]
    return song


def scan_uploads() -> list[dict]:
    def scan(source: str) -> list[dict]:
        client = get_client(required=True)
        playlist = client.uploads_playlist(source)
        ids = client.playlist_video_ids(playlist, limit=CHANNELS[source])
        return [{"source": source, **item} for item in client.video_metadata(ids)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        return [item for items in pool.map(scan, CHANNELS) for item in items]


def select_songs(items, *, sources=CHANNELS, audit=None) -> tuple[list[dict], Counter]:
    counts = Counter()
    candidates = []
    seen_ids = set()

    def record(video_id, channel_id, reason):
        if audit is not None:
            audit.append(dict(video_id=video_id, channel_id=channel_id, reason=reason))

    for item in items:
        source = item["source"]
        if source not in sources or item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])
        video = _video_info(item)
        title = normalize("NFKC", video["title"])
        description = video["description"]
        def reject(reason):
            counts[reason] += 1
            record(video["video_id"], video["channel_id"], reason)

        duration = item["contentDetails"].get("duration", "")
        seconds = re.fullmatch(r"PT(?:(\d+)M)?(?:(\d+)S)?", duration)
        if not seconds or not 60 < int(seconds[1] or 0) * 60 + int(seconds[2] or 0) <= 720:
            reject("duration_excluded")
            continue
        if NON_MV.search(title) or not (MV_MARK.search(title) or VIDEO_CREDIT.search(description)):
            reject("mv_excluded")
            continue
        lyrics = extract_lyrics(description)
        if not lyrics:
            reject("lyrics_excluded")
            continue
        row = song_metadata(video, source)
        if not row["published_at"] or not row["artist"] or row["title"] == row["video_title"]:
            reject("metadata_excluded")
            continue
        row.update(lyrics=lyrics, lyrics_source=row["url"], lyrics_status="official_description")
        candidates.append(row)

    accepted = []
    titles = set()
    fingerprints = []
    for row in sorted(candidates, key=lambda row: (row["published_at"], row["video_id"])):
        title = re.sub(r"\s*\([ -~]+\)\s*$", "", row["title"]) if THAI_LETTER.search(row["title"]) else row["title"]
        song_key = (key(title), key(FEATURE.sub("", row["artist"])))
        lyrics = key(row["lyrics"])
        grams = {lyrics[i:i + 5] for i in range(len(lyrics) - 4)}
        if song_key in titles or any(
            min(len(grams), len(other)) >= 0.9 * max(len(grams), len(other))
            and (overlap := len(grams & other)) >= 0.9 * (len(grams) + len(other) - overlap)
            for other in fingerprints
        ):
            counts["duplicate_song"] += 1
            record(row["video_id"], row["channel_id"], "duplicate_song")
            continue
        titles.add(song_key)
        fingerprints.append(grams)
        accepted.append(row)
        record(row["video_id"], row["channel_id"], "candidate")
    return accepted, counts


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uploads", type=Path, help="ไฟล์ JSONL สำหรับใช้ซ้ำแทนการอ่านรายการอัปโหลดใหม่")
    parser.add_argument("--output", type=Path, default=Path("data/songs.csv"))
    parser.add_argument("--preview", action="store_true", help="นับเพลงที่ผ่านก่อนดึงยอดวิวหรือเขียน CSV")
    args = parser.parse_args()

    try:
        if args.uploads:
            with args.uploads.open(encoding="utf-8") as file:
                items = [json.loads(line) for line in file]
        else:
            items = scan_uploads()
        songs, excluded = select_songs(items)
        print(f"ผ่านการคัด {len(songs)} เพลง · คัดออก {dict(excluded)}", flush=True)
        if args.preview:
            print(dict(Counter(row["channel"] for row in songs)))
            return
        if not songs:
            sys.exit("ไม่พบ MV ที่มีเนื้อร้องผ่านเกณฑ์")
        info = get_client(required=True).videos([row["video_id"] for row in songs])
    except YouTubeError as exc:
        sys.exit(f"YouTube API: {exc}")

    snapshot = []
    for row in songs:
        current = info.get(row["video_id"])
        if current and current["views"] is not None and current["views"] > 0:
            row.update(views=current["views"], views_collected_at=current["views_collected_at"])
            snapshot.append(row)
    days = {
        datetime.fromisoformat(row["views_collected_at"]).astimezone(THAI_TIME).date() for row in snapshot
    }
    if len(days) != 1:
        sys.exit("ต้องมียอดวิวจากวันเดียวกันตามเวลาไทยจึงจะบันทึก CSV")
    write_csv(pd.DataFrame(snapshot), args.output)
    print(f"บันทึก {len(snapshot)} เพลงจากยอดวิววันที่ {days.pop()} -> {args.output}")
    print(f"ข้าม {len(songs) - len(snapshot)} เพลงที่ยอดวิวหายหรือเป็นศูนย์")


if __name__ == "__main__":
    main()
