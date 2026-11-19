"""Export the evaluated lyric predictor without the original training lyrics."""
import argparse
from pathlib import Path

from core.model import LyricsModel

MODEL_FIELDS = (
    "pipeline", "sim_vectorizer", "sim_matrix", "edges", "centers", "class_names", "class_ranges",
    "levels", "class_counts", "report", "is_demo",
)
SONG_FIELDS = ("title", "artist", "views", "url", "level")
EXAMPLE = {
    "title": "ฝนที่หน้าต่าง",
    "artist": "ตัวอย่างสำหรับเดโม",
    "lyrics": (
        "คืนนี้ฝนตกเบา ๆ ที่หน้าต่าง\n"
        "ฉันมองแสงไฟบนถนนสายเดิม\n"
        "วันที่เราเคยเดินกลับบ้านด้วยกัน\n"
        "ตอนนี้เหลือแค่เสียงเพลงในห้อง\n"
        "ถ้าเธอได้ยินในคืนที่คิดถึง\n"
        "ขอให้รู้ว่าฉันยังรออยู่ตรงนี้"
    ),
}


def package(model: LyricsModel) -> LyricsModel:
    bundle = {field: model.b[field] for field in MODEL_FIELDS}
    # Keep row order for cosine similarity. Only display metadata is needed after training.
    bundle["songs"] = [{field: song[field] for field in SONG_FIELDS} for song in model.b["songs"]]
    bundle["example"] = EXAMPLE.copy()
    return LyricsModel(bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Use a separate path when preparing a replacement.")
    original = LyricsModel.load(args.source)
    exported = package(original)
    if original.predict(EXAMPLE["lyrics"]) != exported.predict(EXAMPLE["lyrics"]):
        parser.error("Packaged model changed the prediction on the authored example")
    exported.save(args.output)


if __name__ == "__main__":
    main()
