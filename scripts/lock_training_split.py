"""Lock the train/test assignment for the reviewed Thai MV snapshot."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

DATA = Path(__file__).resolve().parents[1] / "data"
SEED = 42
CLASSES = ("ไม่ดัง", "เฉย ๆ", "ดัง", "ดังมาก")
GROUP_ALIASES = {"แสงดาวptmusic": "ดิดคิตตี้"}


def model_class(song_view_tier: str) -> str:
    return "ดังมาก" if song_view_tier == "Fucking Viral" else song_view_tier


def split_snapshot(manifest: list[dict[str, str]]) -> list[dict[str, str]]:
    songs = sorted(manifest, key=lambda row: row["video_id"])
    ids = [row["video_id"] for row in songs]
    artists = [row["artist_key"] for row in songs]
    labels = [model_class(row["song_view_tier"]) for row in songs]
    if len(ids) != len(set(ids)) or "" in ids or "" in artists:
        raise ValueError("The manifest must have unique video IDs and nonempty artist keys")
    if set(labels) != set(CLASSES):
        raise ValueError("The manifest must contain all four model classes")

    groups = [GROUP_ALIASES.get(artist, artist) for artist in artists]
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_indices, test_indices = next(folds.split(ids, labels, groups))
    if {groups[i] for i in train_indices} & {groups[i] for i in test_indices}:
        raise ValueError("A split group crosses the train/test boundary")
    for indices in (train_indices, test_indices):
        if {labels[i] for i in indices} != set(CLASSES):
            raise ValueError("A partition is missing a model class")

    test = set(test_indices)
    return [
        {"video_id": video_id, "split_group": group, "split": "test" if i in test else "train"}
        for i, (video_id, group) in enumerate(zip(ids, groups))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA / "songs_split.csv")
    args = parser.parse_args()
    with (DATA / "songs_manifest.csv").open(encoding="utf-8-sig", newline="") as file:
        manifest = list(csv.DictReader(file))
    assignment = split_snapshot(manifest)
    with args.output.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["video_id", "split_group", "split"])
        writer.writeheader()
        writer.writerows(assignment)

    labels = {row["video_id"]: model_class(row["song_view_tier"]) for row in manifest}
    for split in ("train", "test"):
        rows = [row for row in assignment if row["split"] == split]
        counts = Counter(labels[row["video_id"]] for row in rows)
        print(f"{split}: {len(rows)} songs, {len({row['split_group'] for row in rows})} groups")
        for label in CLASSES:
            groups = {row["split_group"] for row in rows if labels[row["video_id"]] == label}
            print(f"  {label}: {counts[label]} songs from {len(groups)} groups")


if __name__ == "__main__":
    main()
