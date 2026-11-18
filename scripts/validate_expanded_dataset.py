"""Check the final Thai MV dataset against the agreed collection protocol."""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.view_bands import VIEW_NAMES, view_band

MIN_ARTISTS = 320
MIN_CHANNELS = 151
THAI_TIME = timezone(timedelta(hours=7))
REQUIRED = {"video_id", "channel_id", "artist_key", "title", "lyrics", "lyrics_source",
            "published_at", "views_collected_at", "views"}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def roster_summary(rows: list[dict]) -> dict:
    artists = Counter(row["artist_key"] for row in rows)
    channels = {row["channel_id"] for row in rows}
    return {
        "songs": len(rows), "artists": len(artists), "channels": len(channels),
        "artist_shortfall": max(0, MIN_ARTISTS - len(artists)),
        "channel_shortfall": max(0, MIN_CHANNELS - len(channels)),
        "invalid_artist_counts": {a: n for a, n in sorted(artists.items()) if not 3 <= n <= 5},
    }


def snapshot_summary(rows: list[dict]) -> dict:
    ids, dates, counts = set(), set(), Counter()
    if not rows:
        raise ValueError("The dataset is empty")
    for i, row in enumerate(rows, 2):
        if any(not (row.get(field) or "").strip() for field in REQUIRED):
            raise ValueError(f"Missing required value at row {i}")
        key = row["video_id"]
        if key in ids:
            raise ValueError(f"Duplicate video_id: {key}")
        ids.add(key)
        published = datetime.fromisoformat(row["published_at"])
        observed = datetime.fromisoformat(row["views_collected_at"])
        if published.utcoffset() is None or observed.utcoffset() is None or published > observed:
            raise ValueError(f"Invalid publication or observation time: {key}")
        dates.add(observed.astimezone(THAI_TIME).date().isoformat())
        views = int(row["views"])
        if views <= 0:
            raise ValueError(f"Expected positive views: {key}")
        counts[view_band(views)] += 1
    summary = roster_summary(rows)
    summary.update(snapshot_dates=sorted(dates), classes={name: counts[i] for i, name in enumerate(VIEW_NAMES)})
    summary["meets_requirements"] = (len(dates) == 1 and len(counts) == len(VIEW_NAMES)
                                     and not any(summary[key] for key in (
                                         "artist_shortfall", "channel_shortfall", "invalid_artist_counts")))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    try:
        result = snapshot_summary(read_csv(args.dataset))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["meets_requirements"] else 1)
