"""Compare original and expanded training pools on the same blind three-class test."""
import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from core.model import LyricsModel, load_dataset, train
from core.text import tokenize
from core.view_bands import VIEW_CUTOFFS, VIEW_NAMES
from scripts.validate_expanded_dataset import read_csv, snapshot_summary


def evaluate(source, split, output, prior_manifests=()):
    if output.exists():
        raise ValueError("An evaluation already exists in this directory")
    if not snapshot_summary(read_csv(source))["meets_requirements"]:
        raise ValueError("The expanded dataset must pass the collection thresholds")
    df = load_dataset(source, split_path=split)
    if set(df["origin"]) != {"original", "added"} or df.loc[df["split"].eq("test"), "origin"].ne("added").any():
        raise ValueError("The common holdout must be made up of newly added songs")
    reference = df[df["origin"].eq("original") | df["split"].eq("test")].reset_index(drop=True)
    results = {
        "original": train(reference, locked=True),
        "expanded": train(df, locked=True),
    }
    test = df[df["split"].eq("test")]
    actual = np.digitize(test["views"], VIEW_CUTOFFS)
    training = np.digitize(df.loc[df["split"].eq("train"), "views"], VIEW_CUTOFFS)
    ids = range(len(VIEW_NAMES))
    majority = int(np.bincount(training, minlength=len(VIEW_NAMES)).argmax())
    baseline = np.repeat(majority, len(actual))
    report = {
        "dataset_sha256": sha256(source.read_bytes()).hexdigest(),
        "split_sha256": sha256(split.read_bytes()).hexdigest(),
        "class_names": VIEW_NAMES, "n_test": len(test), "test_groups": test["split_group"].nunique(),
        "majority": {"class": VIEW_NAMES[majority], "correct": int(np.count_nonzero(baseline == actual)),
                     "accuracy": float(np.mean(baseline == actual)),
                     "macro_f1": float(f1_score(actual, baseline, labels=list(ids), average="macro", zero_division=0)),
                     "confusion_matrix": confusion_matrix(actual, baseline, labels=list(ids)).tolist()},
        **{name: bundle["report"] for name, bundle in results.items()},
    }
    report["correct_gain"] = report["expanded"]["test_correct"] - report["original"]["test_correct"]
    if prior_manifests:
        prior_artists = {row["artist_key"] for path in prior_manifests for row in read_csv(path)}
        exposed_groups = set(test.loc[test["artist_key"].isin(prior_artists), "split_group"])
        fresh_mask = ~test["split_group"].isin(exposed_groups).to_numpy()
        fresh = test.loc[fresh_mask]
        if not fresh.empty:
            text = [tokenize(lyric) for lyric in fresh["lyrics"]]
            pred = {name: bundle["pipeline"].predict(text) for name, bundle in results.items()}
            pred["majority"] = baseline[fresh_mask]
            report["artists_outside_prior_manifests"] = {
                "n_test": len(fresh), "groups": fresh["split_group"].nunique(),
                "manifest_hashes": [sha256(path.read_bytes()).hexdigest() for path in prior_manifests],
                "correct": {name: int(np.count_nonzero(values == actual[fresh_mask]))
                            for name, values in pred.items()},
            }
    output.mkdir(parents=True, exist_ok=False)
    LyricsModel(results["expanded"]).save(output / "lyrics_nb.joblib")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("songs", "split", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--prior-manifest", action="append", type=Path, default=[])
    args = parser.parse_args()
    try:
        evaluate(args.songs, args.split, args.output, args.prior_manifest)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
