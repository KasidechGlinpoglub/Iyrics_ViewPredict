"""Apply the reviewed corrections to the 2026-11-18 training snapshot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from unicodedata import normalize

EXCLUDED = {
    "wRHDTwiy35Q": "vocals_only",
    "JT6ddU99W5E": "sweet_version",
    "BNEqSSiN8g8": "live_session",
    "oAndJrPfBJQ": "live_session",
    "X6IA6PJ00Ss": "alternate_version",
    "TeszY4l-6ys": "artist_below_three_songs",
    "6dB_OzEUth4": "artist_below_three_songs",
}
TITLE_FIXES = {
    "IJDoyqMJNfc": "ทางเลือก",
    "CiMWjerf0Z0": "แฟนแล่นแฟน",
    "HGjyZibW1Hw": "ถิ่มน้องสา",
    "K0pnO4hl_bg": "หยั่งลึก",
    "lCd_Xpba3aQ": "เจ็บสุดท้าย",
    "Qkfy_CZoPys": "เฟซบุ๊กทุกข์ใจ",
    "C0CbBt7buGw": "พระจันทร์ที่ไม่เต็มดวง",
}
ARTIST_ALIASES = {"เล็ก พงษธร": "เล็ก รัชเมศฐ์"}
# Offsets are zero-based and refer to these exact official descriptions.
REPAIRS = {
    "BUQUzLyLVNM": (11, 78, "98b03865b5ea56491821935e30af61471d4ec65c96c11c5b98da86b81ad7925c"),
    "-BslM4IsL7Q": (24, 68, "b3f9a4687b94d08b2a0429afb051d3d0b4ba011bdec9b639acfe04e3abe94199"),
    "Xz7z_pfCsJA": (18, 213, "747102279cc5ccf48dc74b31f0788baa70d09cbf8274f32dd5d66807cdbdd133"),
}

CREDIT = re.compile(
    r"^(?:(?:เครดิต(?:เพลง|ดนตรี)?|music video|mv (?:crew|credits?)|credits?)\s*[:：]?\s*$"
    r"|(?:เนื้อเพลง|เนื้อร้อง|เพลง|ชื่อเพลง|title|song|album|อัลบั้ม|ศิลปิน|artist|คำร้อง|ทำนอง|"
    r"เรียบเรียง|ผู้กำกับ|ร้อง(?:นำ|\s*/\s*กีตาร์)?|คุมร้อง|กีตาร์|เบส|กลอง|พิณ|"
    r"มิกซ์(?:มาสเตอร์)?|ศิลปินสังกัดค่ายเพลง|สนับสนุน|ติดต่อ(?:งาน|โฆษณา)?|ติดต่องานแสดง|"
    r"executive producers?|producers?|vocals?|guitars?|bass|drums?|keyboards?|"
    r"mix(?:ed|ing)?(?:\s+and\s+master(?:ing)?)?|master(?:ed|ing)?|record(?:ed|ing)?|"
    r"directors?|production|ออกแบบบ?ตัวหนังสือ|sound engineer|record at|mixdown|beat by|promoter|"
    r"sponsor support|special thanks|ig|line(?: id)?|(?:เพจ\s+)?facebook|instagram|tiktok|youtube|fanpage)\s*[:：]"
    r"|(?:written|produced|recorded|directed|mixed|mastered|arranged)\s+by\s+"
    r"|(?:เนื้อเพลง|เนื้อร้อง)\s*#"
    r"|(?:เนื้อเพลง|เนื้อร้อง|lyrics?)\s*$)",
    re.I,
)
SECTION = re.compile(
    r"^[\s\[\](){}*–—_-]*(?:(?:pre[\s-]?)?(?:chorus|hook)|verse|intro|outro|"
    r"(?:guitar\s+)?solo(?:\s+guitar)?|bridge|rap|refrain|pre|ท่อนฮุก|ฮุก)\s*\d*[\s\[\](){}*–—_-]*$"
    r"|^[\s(\[\*]*(?:repeat|ซ้ำ)[\s*\d(),/\[\]]*(?:(?:pre[\s-]?)?(?:hook|chorus)[\s()\[\]]*)?$"
    r"|^\(?\s*[x×]\s*\d+\s*\)?$",
    re.I,
)
PROMOTION = re.compile(
    r"^(?:follow\s+(?:us\b|@|what the duck\b|sony music thailand\b|whoop\s*$|me\s*🩷)|"
    r"(?:available|listen)\s+on\b|(?:ช่องทาง\s*)?ติดตาม(?:ข่าวสาร|ผลงาน|ศิลปิน|ได้ที่|ทุกช่องทาง)|"
    r"เบอร์โทรนี้เพิ่มเป็นเพื่อนทางไลน์ได้\s*$|กด(?:ไลก์|ติดตาม|subscribe)|"
    r"(?:ดาวน์โหลด|download)\s*(?:เพลง|ริงโทน|[:：*#])|"
    r"(?:ติดต่อ(?:งาน|โฆษณา)|โทร\s*[:：]|for\s+(?:work|booking)|booking\s*[:：]).*)"
    r"|^https?://|^www\.|(?:^|\s)ig\s*[:：@]|^#\S+(?:\s+#\S+)*$",
    re.I,
)
PHONE = re.compile(r"(?<!\d)(?:\+66|0)\d{1,2}[- .]?\d{3}[- .]?\d{3,4}(?!\d)")
MEMBER_CREDIT = re.compile(
    r"^.+\([^()]+\)\s+(?:ig\s+\w+|guitars?|bass|drums?|vocals?|keyboards?)$"
    r"|^.+\s+on\s+(?:guitars?|bass|drums?|vocals?|keyboards?)$",
    re.I,
)
# In this snapshot these decorated boundaries introduce a footer, unlike bare separators.
FOOTER = re.compile(
    r"^[-–—━─]{3,}\s*(?:mv crew|music|milk bkk|ติดตามหน่าฮ่าน)\s*[-–—━─]{3,}$"
    r"|^✦\s*music team\s*$",
    re.I,
)


def comparison_key(value: str) -> str:
    return "".join(c for c in normalize("NFKC", value).casefold() if c.isalnum() or "\u0e31" <= c <= "\u0e4e")


def clean_lyrics(text: str, row: dict) -> tuple[str, Counter]:
    invisible = str.maketrans({"\u200b": None, "\ufeff": None, "\u00a0": " "})
    lines = [normalize("NFC", line.translate(invisible)).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    names = {comparison_key(row[field]) for field in ("artist", "primary_artist")}
    removed = Counter()
    kept = []
    for i, line in enumerate(lines):
        if FOOTER.fullmatch(line):
            removed["credit_or_promotion_footer"] += len(lines) - i
            break
        candidate = line.lstrip(" [【(*•▶▷►📌🩷").rstrip("]】")
        if not any(c.isalnum() for c in line):
            reason = "separator"
        elif CREDIT.search(candidate) or MEMBER_CREDIT.fullmatch(line):
            reason = "credit_or_heading"
        elif SECTION.search(line):
            reason = "section_or_repeat"
        elif PROMOTION.search(candidate) or PHONE.search(candidate) or (
            candidate.casefold().startswith("follow ") and comparison_key(candidate[7:]) in names
        ):
            reason = "contact_or_promotion"
        else:
            kept.append(line)
            continue
        removed[reason] += 1
    while kept and comparison_key(kept[0]) in names:
        kept.pop(0)
        removed["artist_at_boundary"] += 1
    while kept and comparison_key(kept[-1]) in names:
        kept.pop()
        removed["artist_at_boundary"] += 1
    return "\n".join(kept), removed


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames, list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)


def view_tier(views: float) -> str:
    if views >= 100_000_000:
        return "Fucking Viral"
    if views >= 20_000_000:
        return "ดังมาก"
    if views >= 1_000_000:
        return "ดัง"
    if views >= 100_000:
        return "เฉย ๆ"
    return "ไม่ดัง"


def repair(row: dict, source: dict) -> None:
    start, end, fingerprint = REPAIRS[row["video_id"]]
    description = source["description"]
    if source["channelId"] != row["channel_id"] or hashlib.sha256(description.encode()).hexdigest() != fingerprint:
        raise ValueError(f"Source does not match the reviewed description for {row['video_id']}")
    text = "\n".join(description.splitlines()[start:end])
    if row["video_id"] == "Xz7z_pfCsJA":
        # The uploader used one token per line. Preserve their paragraph boundaries and every token.
        text = "\n".join(" ".join(block.split()) for block in re.split(r"\n\s*\n", text.strip()))
        if text.split() != row["lyrics"].split():
            raise ValueError("Yellow Line has different tokens from the reviewed snapshot")
    row.update(lyrics=text, lyrics_source=row["url"], lyrics_status="official_description")


def review(input_dir: Path, sources: Path, output: Path) -> None:
    song_fields, songs = read_csv(input_dir / "songs.csv")
    manifest_fields, manifest = read_csv(input_dir / "songs_manifest.csv")
    original_manifest = {r["video_id"]: r for r in manifest}
    if len(songs) != 721 or len({r["video_id"] for r in songs}) != len(songs) or len(original_manifest) != len(songs):
        raise ValueError("Expected the original 721 unique songs")
    shared = set(song_fields) & set(manifest_fields)
    if len(manifest) != len(songs) or any(
        any(row[field] != original_manifest.get(row["video_id"], {}).get(field) for field in shared) for row in songs
    ):
        raise ValueError("Original songs and manifest do not agree")
    descriptions = {item["id"]: item["snippet"] for item in json.loads(sources.read_text(encoding="utf-8"))}
    aliases = [r for r in songs if r["primary_artist"] in {*ARTIST_ALIASES, *ARTIST_ALIASES.values()}]
    # Keep the five earliest MVs across both credited names using published date, then ID.
    excess = {r["video_id"]: "artist_above_five_songs" for r in sorted(
        aliases, key=lambda r: (r["published_at"], r["video_id"])
    )[5:]}
    excluded = EXCLUDED | excess
    cleaned = []
    audit = []
    artist_views = defaultdict(list)
    for row in songs:
        video_id = row["video_id"]
        if video_id in excluded:
            audit.append({"video_id": video_id, "action": "excluded", "detail": excluded[video_id]})
            continue
        if video_id in REPAIRS:
            repair(row, descriptions[video_id])
            audit.append({"video_id": video_id, "action": "restored_from_source", "detail": row["lyrics_source"]})
        if video_id in TITLE_FIXES:
            row["title"] = TITLE_FIXES[video_id]
            audit.append({"video_id": video_id, "action": "title_corrected", "detail": row["title"]})
        if row["primary_artist"] in ARTIST_ALIASES:
            row["primary_artist"] = ARTIST_ALIASES[row["primary_artist"]]
            row["artist_key"] = comparison_key(row["primary_artist"])
            audit.append({"video_id": video_id, "action": "artist_merged", "detail": row["primary_artist"]})
        row["channel"] = row["channel"].strip()
        text, removed = clean_lyrics(row["lyrics"], row)
        if text != row["lyrics"]:
            audit.append({"video_id": video_id, "action": "lyrics_cleaned", "detail": json.dumps(removed, sort_keys=True)})
            row["lyrics"] = text
        if not row["lyrics"].strip():
            raise ValueError(f"No lyrics left for {video_id}")
        artist_views[row["artist_key"]].append(int(row["views"]))
        cleaned.append(row)

    if len(artist_views) < 150 or any(not 3 <= len(views) <= 5 for views in artist_views.values()):
        raise ValueError("Expected at least 150 artists with 3–5 songs each")
    if len({r["channel_id"] for r in cleaned}) <= 60:
        raise ValueError("Expected more than 60 channels")
    medians = {artist: median(views) for artist, views in artist_views.items()}
    final_manifest = []
    for row in cleaned:
        value = medians[row["artist_key"]]
        row["artist_median_mv_views"] = str(float(value))
        row["artist_tier"] = view_tier(value)
        day = datetime.fromisoformat(row["views_collected_at"]).astimezone(timezone(timedelta(hours=7))).date()
        views = int(row["views"])
        band = "under_100k" if views < 100_000 else "100k_to_under_1m" if views < 1_000_000 else "1m_or_more"
        manifest_row = original_manifest[row["video_id"]] | {field: row[field] for field in shared}
        if (
            str(day) != "2026-11-18" or manifest_row["snapshot_date_th"] != str(day)
            or view_tier(views) != row["song_view_tier"] or manifest_row["view_band"] != band
        ):
            raise ValueError(f"Unexpected snapshot date or song tier for {row['video_id']}")
        final_manifest.append(manifest_row)

    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "songs.csv", song_fields, cleaned)
    write_csv(output / "songs_manifest.csv", manifest_fields, final_manifest)
    write_csv(output / "review_audit.csv", ["video_id", "action", "detail"], audit)
    print(f"{len(cleaned)} songs, {len(artist_views)} artists, {len({r['channel_id'] for r in cleaned})} channels")
    print(dict(Counter(r["song_view_tier"] for r in cleaned)))
    print(dict(Counter(r["action"] for r in audit)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review(args.input, args.sources, args.output)
