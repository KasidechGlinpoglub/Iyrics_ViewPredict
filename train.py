"""เทรนโมเดล Naive Bayes ทำนายระดับยอดวิวจากเนื้อเพลง

ใช้งาน:
    python train.py                       # ใช้ data/songs.csv (ถ้าไม่มีจะใช้ data/demo_songs.csv)
    python train.py --data my.csv --classes 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.model import LyricsModel, load_dataset, train

BASE = Path(__file__).resolve().parent
REAL_DATA = BASE / "data" / "songs.csv"
DEMO_DATA = BASE / "data" / "demo_songs.csv"
MODEL_PATH = BASE / "models" / "lyrics_nb.joblib"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="ไฟล์ CSV: title, artist, lyrics, views, url")
    ap.add_argument("--classes", type=int, default=4, help="จำนวนระดับยอดวิว (2-5) ค่าเริ่มต้น 4")
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    args = ap.parse_args()

    data_path = args.data or (REAL_DATA if REAL_DATA.exists() else DEMO_DATA)
    if not data_path.exists():
        sys.exit(f"ไม่พบไฟล์ข้อมูล {data_path}\nสร้างข้อมูลจำลองก่อนด้วย: python scripts/make_demo_data.py")
    is_demo = data_path.resolve() == DEMO_DATA.resolve()

    df = load_dataset(data_path)
    print(f"ข้อมูล: {data_path.name} — {len(df)} เพลง{' (ข้อมูลจำลอง)' if is_demo else ''}")
    print("กำลังตัดคำและเทรนโมเดล (GridSearchCV 5-fold)...")
    bundle = train(df, n_classes=args.classes, is_demo=is_demo)
    model = LyricsModel(bundle)
    model.save(args.out)

    rep = bundle["report"]
    (args.out.parent / "report.json").write_text(
        json.dumps({"data": str(data_path.name), "classes": model.classes, **rep}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nระดับยอดวิว (แบ่งด้วย quantile):")
    for c in model.classes:
        print(f"  {c['name']:<8} {c['range']:<24} {c['count']} เพลง")
    print(f"\nพารามิเตอร์ที่ดีที่สุด : {rep['best_params']}")
    print(f"CV macro-F1           : {rep['cv_f1_macro']:.3f}")
    print(f"Test accuracy         : {rep['test_accuracy']:.3f}  (baseline ทายคลาสที่พบมากสุด = {rep['baseline_accuracy']:.3f})")
    print(f"Test macro-F1         : {rep['test_f1_macro']:.3f}")
    print(f"Test MAE (log10 views): {rep['test_mae_log10']:.3f}  (≈ คลาดเคลื่อน ×{10 ** rep['test_mae_log10']:.1f} เท่า)")
    print(f"ขนาดคลังคำ            : {rep['vocab_size']} คำ")
    print("Confusion matrix (แถว = จริง, คอลัมน์ = ทำนาย):")
    for row in rep["confusion_matrix"]:
        print("  ", row)
    print(f"\nบันทึกโมเดล -> {args.out}")


if __name__ == "__main__":
    main()
