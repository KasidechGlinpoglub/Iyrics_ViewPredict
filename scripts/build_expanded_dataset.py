"""Select and lock the expanded MV cohort, then obtain one dated view snapshot."""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import random
import re

from core.config import load_env
from core.view_bands import VIEW_NAMES, view_band
from core.youtube import YouTubeError
from scripts.collect_expanded_candidates import write_rows
from scripts.collect_youtube import get_client
from scripts.prepare_songs import key
from scripts.review_training_snapshot import clean_lyrics
from scripts.validate_expanded_dataset import MIN_ARTISTS, MIN_CHANNELS, read_csv, roster_summary, snapshot_summary

COLLAB = re.compile(r"\s+(?:feat(?:uring)?\.?|ft\.?|x|with|vs\.?|[&×+])\s+", re.I)
REPLY_ARTISTS = ("ดิดคิตตี้", "แสงดาวptmusic")
NON_ORIGINAL_CHANNELS = {"เพลงฮิตขึ้นหิ้ง"}
SECONDARY_VERSION = re.compile(
    r"\b(?:vocals?[- ]only|sweet version|alternate|remastered|live session|artist version)\b", re.I)


def credits(row):
    return {key(name) for name in COLLAB.split(row["artist"]) if key(name)}


def shingles(text):
    return {text[i:i + 5] for i in range(len(text) - 4)}


def select(base, sources):
    rows = [dict(row, origin="original") for row in base]
    channels = {row["channel_id"] for row in rows}
    artists = Counter(row["artist_key"] for row in rows)
    known = {key(name): row["artist_key"] for row in base
             for name in (row["artist_key"], row["primary_artist"], COLLAB.split(row["artist"])[0])}
    ids = {row["video_id"] for row in rows}
    titles = {(row["artist_key"], key(row["title"])) for row in rows}
    lyrics = {key(row["lyrics"]) for row in rows}
    fingerprints = [shingles(text) for text in lyrics]
    options = defaultdict(list)
    for path in sources:
        for row in sorted(read_csv(path), key=lambda row: (row["published_at"], row["video_id"])):
            primary = row.get("primary_artist") or COLLAB.split(row.get("artist", ""))[0].strip()
            normalized = key(primary)
            canonical = re.sub(r"(?:อาร์สยาม|ไหทองคํา)$", "", normalized)
            artist = known.get(normalized, known.get(canonical, canonical))
            if (row["channel"].strip() in NON_ORIGINAL_CHANNELS or "[ai artist]" in primary.casefold()
                    or SECONDARY_VERSION.search(row.get("video_title", row["title"]))):
                continue
            row = dict(row, artist=row.get("artist") or primary, primary_artist=primary)
            cleaned, _ = clean_lyrics(row["lyrics"], row)
            title, text = key(row["title"]), key(cleaned)
            if (not artist or not title or not text or row["video_id"] in ids or (artist, title) in titles
                    or text in lyrics or not row.get("lyrics_source") or len(cleaned.splitlines()) < 8):
                continue
            grams = shingles(text)
            if not grams:
                continue
            if any(min(len(grams), len(other)) >= 0.9 * max(len(grams), len(other))
                   and (overlap := len(grams & other)) >= 0.9 * (len(grams) + len(other) - overlap)
                   for other in fingerprints):
                continue
            ids.add(row["video_id"])
            titles.add((artist, title))
            lyrics.add(text)
            fingerprints.append(grams)
            options[artist].append(dict(row, artist_key=artist, lyrics=cleaned, origin="added"))
    eligible = {a: tracks for a, tracks in options.items() if len(tracks) >= 3 and a not in artists}
    print(f"{len(eligible)} new artist keys with at least three candidate songs; "
          f"{len({r['channel_id'] for group in eligible.values() for r in group} - channels)} potentially new channels",
          flush=True)
    while eligible and (len(artists) < MIN_ARTISTS or len(channels) < MIN_CHANNELS):
        artist = min(eligible, key=lambda a: (
            -len({row["channel_id"] for row in eligible[a]} - channels), a))
        tracks = eligible.pop(artist)
        picked = []
        while tracks and len(picked) < 5:
            index = next((i for i, row in enumerate(tracks) if row["channel_id"] not in channels), None)
            if index is None and len(picked) >= 3:
                break
            row = tracks.pop(index if index is not None else 0)
            channels.add(row["channel_id"])
            picked.append(row)
        rows.extend(picked)
        artists[artist] = len(picked)
    # Songs by an existing artist can add a new channel when there is room under the cap.
    for artist, tracks in sorted(options.items()):
        for row in tracks:
            if len(channels) >= MIN_CHANNELS:
                break
            if artist in artists and artists[artist] < 5 and row["channel_id"] not in channels and row not in rows:
                rows.append(row)
                channels.add(row["channel_id"])
                artists[artist] += 1
    return rows


def choose_split(rows):
    parent = {}

    def group(name):
        parent.setdefault(name, name)
        while parent[name] != name:
            name = parent[name]
        return name

    def connect(names):
        roots = sorted({group(name) for name in names})
        for name in roots[1:]:
            parent[name] = roots[0]

    connect(REPLY_ARTISTS)
    for row in rows:
        connect({row["artist_key"], *credits(row)})
    grouped = defaultdict(list)
    for row in rows:
        grouped[group(row["artist_key"])].append(row)
    eligible = sorted(key for key, members in grouped.items()
                      if all(row["origin"] == "added" for row in members))
    random.Random(42).shuffle(eligible)
    heldout, total = set(), 0
    for group_id in eligible:
        if total >= 0.2 * len(rows):
            break
        heldout.add(group_id)
        total += len(grouped[group_id])
    if not heldout:
        raise ValueError("No new artist groups are eligible for the test set")
    for group_id, members in grouped.items():
        for row in members:
            row.update(split_group=group_id, split="test" if group_id in heldout else "train")
    return rows


def lock_rows(rows, output):
    summary = roster_summary(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if any(summary[k] for k in ("artist_shortfall", "channel_shortfall", "invalid_artist_counts")):
        raise ValueError("Candidate roster falls short of the final collection thresholds")
    rows = choose_split(rows)
    if len({r["video_id"] for r in rows}) != len(rows):
        raise ValueError("Selected videos must be unique")
    roster = [{k: row[k] for k in ("video_id", "channel_id", "artist_key", "split_group", "split", "origin")}
              | {"lyrics_sha256": sha256(row["lyrics"].encode()).hexdigest()} for row in rows]
    with (output / "roster-lock.json").open("x", encoding="utf8") as stream:
        json.dump(roster, stream, ensure_ascii=False, indent=2)
    write_rows(output / "selected.csv", rows, SONG_FIELDS)


SONG_FIELDS = ["video_id", "title", "artist", "primary_artist", "artist_key", "channel", "channel_id",
               "published_at", "url", "lyrics", "lyrics_source", "lyrics_status", "origin", "split_group", "split"]


def snapshot(path):
    selected = read_csv(path / "selected.csv")
    roster = json.loads((path / "roster-lock.json").read_text(encoding="utf8"))
    keys = [r["video_id"] for r in roster]
    if any((path / filename).exists() for filename in ("songs.csv", "songs_manifest.csv", "songs_split.csv")):
        raise ValueError("A snapshot already exists here; keep each observation in its own directory")
    for row, locked in zip(selected, roster, strict=True):
        if (any(row[field] != v for field, v in locked.items() if field != "lyrics_sha256")
                or sha256(row["lyrics"].encode()).hexdigest() != locked["lyrics_sha256"]):
            raise ValueError("Selected rows differ from the locked roster")
    client = get_client(required=True)
    current = client.videos(keys)
    if set(current) != set(keys) or any(not row["views"] for row in current.values()):
        raise ValueError("Some locked videos have no positive view count")
    rows = [dict(row, views=str(current[row["video_id"]]["views"]),
                 views_collected_at=current[row["video_id"]]["views_collected_at"]) for row in selected]
    summary = snapshot_summary(rows)
    if not summary["meets_requirements"]:
        raise ValueError(f"Snapshot does not pass validation: {summary}")
    for side in ("train", "test"):
        counts = Counter(view_band(int(r["views"])) for r in rows if r["split"] == side)
        if len(counts) != len(VIEW_NAMES):
            raise ValueError(f"{side} misses a view class; preserve the roster for diagnosis")
        summary[side] = {name: counts[i] for i, name in enumerate(VIEW_NAMES)}
    fields = [*SONG_FIELDS, "views", "views_collected_at"]
    write_rows(path / "songs.csv", rows, fields)
    write_rows(path / "songs_manifest.csv", rows, [f for f in fields if f != "lyrics"])
    write_rows(path / "songs_split.csv", rows, ["video_id", "split_group", "split"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preview", "lock"):
        p = commands.add_parser(name)
        p.add_argument("base", type=Path)
        p.add_argument("candidates", type=Path, nargs="+")
        if name == "lock":
            p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("snapshot")
    p.add_argument("output", type=Path)
    p.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "snapshot":
            if args.env_file:
                load_env(args.env_file)
            snapshot(args.output)
            return
        rows = select(read_csv(args.base), args.candidates)
        if args.command == "preview":
            print(json.dumps(roster_summary(rows), ensure_ascii=False, indent=2))
        else:
            args.output.mkdir(parents=True, exist_ok=True)
            lock_rows(rows, args.output)
    except (OSError, ValueError, YouTubeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
