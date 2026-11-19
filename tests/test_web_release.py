"""Smoke-test the artifact and Flask entry points used by Vercel."""
from io import BytesIO
import unittest
from unittest.mock import patch

from openpyxl import load_workbook
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import app
from core.model import train


class WebReleaseTest(unittest.TestCase):
    def test_packaged_artifact_and_endpoints(self):
        with patch.multiple(app, MODEL_PATH=app.PUBLIC_MODEL_PATH, _model=None, youtube=None):
            client = app.app.test_client()
            info = client.get("/api/info").get_json()
            self.assertEqual(info["report"]["n_train"], 711)
            self.assertEqual(info["report"]["test_correct"], 95)
            self.assertEqual(len(info["classes"]), 3)
            self.assertTrue(all("lyrics" not in song for song in app.get_model().b["songs"]))

            example = client.get("/api/example").get_json()
            self.assertEqual(example["artist"], "ตัวอย่างสำหรับเดโม")
            prediction = client.post("/api/predict", json={"lyrics": example["lyrics"]})
            self.assertEqual(prediction.status_code, 200)
            self.assertEqual(len(prediction.get_json()["probabilities"]), 3)
            self.assertEqual(client.get("/").status_code, 200)
            for asset in ("style.css", "app.js", "vinyl.webp"):
                response = client.get(f"/static/{asset}")
                self.assertEqual(response.status_code, 200)
                response.close()

    def test_sheet_exports_plain_text(self):
        with patch.multiple(app, MODEL_PATH=app.PUBLIC_MODEL_PATH, _model=None):
            lyrics = "=คิดถึงหัวใจในคืนฝนพรำ\x01"
            response = app.app.test_client().post("/api/export/xlsx", json={"lyrics": lyrics})
            self.assertEqual(response.status_code, 200)
            book = load_workbook(BytesIO(response.data))
            cell = book["Heatmap รายบรรทัด"]["B5"]
            self.assertEqual(cell.data_type, "s")
            self.assertEqual(cell.value, lyrics.rstrip("\x01"))

    def test_export_rejects_invalid_youtube_numbers(self):
        with patch.multiple(app, MODEL_PATH=app.PUBLIC_MODEL_PATH, _model=None):
            client = app.app.test_client()
            for fields in ({"views": 10**400}, {"lyric_similarity": float("inf")}):
                with self.subTest(fields=fields):
                    row = {"views": 1_000, "lyric_similarity": .5, **fields}
                    response = client.post("/api/export/xlsx", json={
                        "lyrics": "คิดถึงหัวใจในคืนฝนพรำ", "youtube": [row],
                    })
                    self.assertEqual(response.status_code, 400)

    def test_quantile_cutoffs_are_fit_on_training_rows(self):
        rows = pd.DataFrame({
            "views": np.arange(1_000, 121_000, 1_000),
            "title": "ตัวอย่าง", "artist": "", "url": "",
            "lyrics": "คืนฝนพรำ หัวใจยังคิดถึง",
        })
        fit_ids, _ = train_test_split(np.arange(len(rows)), test_size=.2, random_state=42)
        expected = np.quantile(np.log10(rows.iloc[fit_ids]["views"]), [1/3, 2/3])
        model = train(rows, levels="quantile", n_classes=3)
        np.testing.assert_allclose(model["edges"], expected)


if __name__ == "__main__":
    unittest.main()
