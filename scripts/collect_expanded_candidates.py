"""Archive channel uploads and screen Thai MV candidates without reading view counts."""
import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.config import load_env
from core.youtube import YouTubeError
from scripts.collect_youtube import get_client
from scripts.prepare_songs import select_songs
from scripts.validate_expanded_dataset import read_csv


def write_rows(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cache_channel(row, cache):
    channel_id = row["channel_id"]
    path = cache / f"{channel_id}.json"
    if path.exists():
        return "cached"
    try:
        client = get_client(required=True)
        playlist = client.uploads_playlist(channel_id)
        ids = client.playlist_video_ids(playlist, limit=int(row["upload_limit"]))
        items = client.video_metadata(ids)
    except YouTubeError as error:
        return error.code
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    pending.replace(path)
    return "collected"


def main():
    import re

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("channels", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--offline", action="store_true", help="Screen only uploads already cached")
    args = parser.parse_args()
    channels = read_csv(args.channels)
    by_id = {row["channel_id"]: row for row in channels}
    if not channels or len(by_id) != len(channels) or any(
        not re.fullmatch(r"UC[A-Za-z0-9_-]{22}", key) or int(row["upload_limit"]) < 1 for key, row in by_id.items()
    ):
        parser.error("Expected unique channel IDs and positive upload limits")
    cache = args.output / "uploads"
    cache.mkdir(parents=True, exist_ok=True)
    if args.env_file:
        load_env(args.env_file)
    if not args.offline:
        get_client(required=True)
        with ThreadPoolExecutor(max_workers=8) as workers:
            statuses = list(workers.map(lambda row: cache_channel(row, cache), channels))
        print(dict(Counter(statuses)), flush=True)
    else:
        statuses = ["cached" if (cache / f"{key}.json").exists() else "pending" for key in by_id]

    def uploads():
        for key in by_id:
            saved = cache / f"{key}.json"
            if saved.exists():
                for item in json.loads(saved.read_text(encoding="utf-8")):
                    yield {"source": key, **item}

    audit = []
    candidates, excluded = select_songs(uploads(), sources=by_id, audit=audit)
    fields = ["video_id", "title", "artist", "channel", "channel_id", "video_title",
              "published_at", "url", "lyrics", "lyrics_source", "lyrics_status"]
    write_rows(args.output / "candidates.csv", candidates, fields)
    write_rows(args.output / "screening.csv", audit, ["video_id", "channel_id", "reason"])
    counts = Counter(row["channel_id"] for row in candidates)
    write_rows(args.output / "channels.csv",
               ({**row, "status": status, "candidates": counts[row["channel_id"]]}
                for row, status in zip(channels, statuses)),
               ["channel_id", "name", "reason", "upload_limit", "status", "candidates"])
    print(f"{len(candidates)} candidates in {len(counts)} channels, rejections: {dict(excluded)}", flush=True)


if __name__ == "__main__":
    main()
