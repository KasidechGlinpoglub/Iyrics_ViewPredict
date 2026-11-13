"""โมเดลทำนายยอดวิวจากเนื้อเพลงด้วย Multinomial Naive Bayes

แนวคิด
- Naive Bayes เป็นตัวจำแนกประเภท จึงแบ่งยอดวิวเป็น "ระดับ" (class) ด้วย quantile ของ log10(views)
  ทำให้แต่ละระดับมีจำนวนเพลงใกล้เคียงกัน
- Feature = TF-IDF ของคำที่ตัดด้วย PyThaiNLP
- ยอดวิวที่คาด = 10 ^ ( Σ P(ระดับ|เนื้อเพลง) × ค่ากลาง log10(views) ของระดับนั้น )
- เพลงที่เนื้อหาใกล้เคียง = cosine similarity ของ TF-IDF
- Heatmap รายคำ = P(ระดับ|คำ) จาก feature_log_prob_ ของ Naive Bayes (normalize ทีละคำ)
- Heatmap รายบรรทัด = predict_proba ของแต่ละบรรทัด
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .text import identity, split_lines, tokenize

CLASS_NAMES = {
    2: ["ต่ำ", "สูง"],
    3: ["ต่ำ", "ปานกลาง", "สูง"],
    4: ["ต่ำ", "ปานกลาง", "สูง", "สูงมาก"],
    5: ["ต่ำมาก", "ต่ำ", "ปานกลาง", "สูง", "สูงมาก"],
}
MAX_LINES = 300


# ---------------------------------------------------------------- utilities
def format_views(n: float) -> str:
    """แสดงยอดวิวแบบ YouTube ภาษาไทย เช่น 12.4 ล้าน, 3.5 แสน"""
    n = float(n)
    for unit, size in (("พันล้าน", 1e9), ("ล้าน", 1e6), ("แสน", 1e5), ("หมื่น", 1e4), ("พัน", 1e3)):
        if n >= size:
            value = n / size
            text = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{text} {unit}"
    return f"{int(round(n)):,}"


def load_dataset(path: str | Path) -> pd.DataFrame:
    """อ่าน CSV (title, artist, lyrics, views, url) และทำความสะอาดข้อมูล"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"title", "lyrics", "views"} - set(df.columns)
    if missing:
        raise ValueError(f"ไฟล์ {path} ขาดคอลัมน์: {', '.join(sorted(missing))}")
    for col in ("artist", "url"):
        if col not in df.columns:
            df[col] = ""
    df["views"] = pd.to_numeric(df["views"], errors="coerce")
    df = df.dropna(subset=["lyrics", "views"])
    df = df[(df["views"] > 0) & (df["lyrics"].astype(str).str.strip() != "")]
    for col in ("title", "artist", "url"):
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["lyrics"] = df["lyrics"].astype(str)
    df = df.drop_duplicates(subset=["title", "artist"]).reset_index(drop=True)
    return df


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vect", CountVectorizer(analyzer=identity)),
            ("tfidf", TfidfTransformer(sublinear_tf=True)),
            ("nb", MultinomialNB()),
        ]
    )


def _class_ranges(edges: list[float], n_classes: int) -> list[str]:
    bounds = [format_views(10**e) for e in edges]
    ranges = []
    for c in range(n_classes):
        if c == 0:
            ranges.append(f"< {bounds[0]}")
        elif c == n_classes - 1:
            ranges.append(f"≥ {bounds[-1]}")
        else:
            ranges.append(f"{bounds[c - 1]} – {bounds[c]}")
    return ranges


def _class_centers(log_views: np.ndarray, y: np.ndarray, n_classes: int) -> list[float]:
    return [float(np.median(log_views[y == c])) for c in range(n_classes)]


# ---------------------------------------------------------------- training
def train(df: pd.DataFrame, n_classes: int = 4, seed: int = 42, is_demo: bool = False) -> dict:
    if n_classes not in CLASS_NAMES:
        raise ValueError(f"รองรับจำนวนระดับ {sorted(CLASS_NAMES)} เท่านั้น")
    if len(df) < n_classes * 10:
        raise ValueError(f"ข้อมูลน้อยเกินไป ({len(df)} เพลง) ควรมีอย่างน้อย {n_classes * 10} เพลง")

    tokens = [tokenize(t) for t in df["lyrics"]]
    log_views = np.log10(df["views"].to_numpy(dtype=float))
    edges = np.quantile(log_views, [i / n_classes for i in range(1, n_classes)])
    y = np.digitize(log_views, edges)

    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    docs_tr = [tokens[i] for i in tr]
    docs_te = [tokens[i] for i in te]

    grid = GridSearchCV(
        _make_pipeline(),
        {
            "vect__min_df": [1, 2],
            "tfidf__use_idf": [True, False],
            "nb__alpha": [0.05, 0.1, 0.3, 0.5, 1.0],
        },
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        scoring="f1_macro",
        n_jobs=1,
    )
    grid.fit(docs_tr, y[tr])

    # ---- ประเมินผลบน test set
    proba_te = grid.predict_proba(docs_te)
    pred_te = proba_te.argmax(axis=1)
    centers_tr = _class_centers(log_views[tr], y[tr], n_classes)
    est_log_te = proba_te @ np.array(centers_tr)
    majority = Counter(y[tr]).most_common(1)[0][0]
    report = {
        "n_songs": int(len(df)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "best_params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in grid.best_params_.items()},
        "cv_f1_macro": round(float(grid.best_score_), 4),
        "test_accuracy": round(float(accuracy_score(y[te], pred_te)), 4),
        "test_f1_macro": round(float(f1_score(y[te], pred_te, average="macro")), 4),
        "baseline_accuracy": round(float(np.mean(y[te] == majority)), 4),
        "test_mae_log10": round(float(np.mean(np.abs(est_log_te - log_views[te]))), 4),
        "confusion_matrix": confusion_matrix(y[te], pred_te, labels=list(range(n_classes))).tolist(),
    }

    # ---- โมเดลจริงสำหรับใช้งาน: เทรนด้วยข้อมูลทั้งหมดด้วยพารามิเตอร์ที่ดีที่สุด
    final = _make_pipeline().set_params(**grid.best_params_)
    final.fit(tokens, y)
    report["vocab_size"] = int(len(final.named_steps["vect"].vocabulary_))

    sim_vectorizer = TfidfVectorizer(analyzer=identity, sublinear_tf=True)
    sim_matrix = sim_vectorizer.fit_transform(tokens)

    songs = df[["title", "artist", "views", "url", "lyrics"]].copy()
    songs["views"] = songs["views"].astype(np.int64)
    songs["level"] = y

    return {
        "pipeline": final,
        "sim_vectorizer": sim_vectorizer,
        "sim_matrix": sim_matrix,
        "songs": songs.to_dict("records"),
        "edges": edges.tolist(),
        "centers": _class_centers(log_views, y, n_classes),
        "class_names": CLASS_NAMES[n_classes],
        "class_ranges": _class_ranges(edges.tolist(), n_classes),
        "class_counts": np.bincount(y, minlength=n_classes).tolist(),
        "report": report,
        "is_demo": is_demo,
    }


# ---------------------------------------------------------------- inference
class LyricsModel:
    def __init__(self, bundle: dict):
        self.b = bundle
        self.pipe: Pipeline = bundle["pipeline"]
        self.vocab: dict = self.pipe.named_steps["vect"].vocabulary_
        flp = self.pipe.named_steps["nb"].feature_log_prob_  # (n_classes, n_words)
        # P(ระดับ | คำ) โดยสมมติ prior เท่ากัน = softmax ของ log P(คำ | ระดับ) ตามแกนระดับ
        flp = flp - flp.max(axis=0, keepdims=True)
        word_class = np.exp(flp)
        self.word_class = word_class / word_class.sum(axis=0, keepdims=True)
        self.centers = np.array(bundle["centers"])

    @classmethod
    def load(cls, path: str | Path) -> "LyricsModel":
        return cls(joblib.load(path))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.b, path, compress=3)

    # ---- public info
    @property
    def classes(self) -> list[dict]:
        return [
            {"name": n, "range": r, "count": c}
            for n, r, c in zip(self.b["class_names"], self.b["class_ranges"], self.b["class_counts"])
        ]

    def info(self) -> dict:
        return {"classes": self.classes, "report": self.b["report"], "is_demo": self.b["is_demo"]}

    def random_example(self, rng: np.random.Generator | None = None) -> dict:
        rng = rng or np.random.default_rng()
        song = self.b["songs"][int(rng.integers(len(self.b["songs"])))]
        return {"title": song["title"], "artist": song["artist"], "lyrics": song["lyrics"]}

    # ---- prediction
    def predict(self, lyrics: str, top_k: int = 5) -> dict:
        tokens = tokenize(lyrics)
        known = [t for t in tokens if t in self.vocab]
        if not known:
            raise ValueError("ไม่พบคำในเนื้อเพลงที่โมเดลรู้จัก ลองใส่เนื้อเพลงให้ยาวขึ้น")

        proba = self.pipe.predict_proba([tokens])[0]
        level = int(proba.argmax())
        est_views = float(10 ** (proba @ self.centers))

        return {
            "prediction": {
                "views": round(est_views),
                "views_text": format_views(est_views),
                "level": level,
                "level_name": self.b["class_names"][level],
                "level_range": self.b["class_ranges"][level],
                "confidence": round(float(proba[level]), 4),
            },
            "probabilities": [
                {"name": c["name"], "range": c["range"], "prob": round(float(p), 4)}
                for c, p in zip(self.classes, proba)
            ],
            "tokens": {"total": len(tokens), "known": len(known), "unique": len(set(tokens))},
            "similar": self.similar(tokens, top_k),
            "heatmap": self.heatmap(lyrics, tokens),
        }

    def similar(self, tokens: list[str], top_k: int = 5) -> list[dict]:
        query = self.b["sim_vectorizer"].transform([tokens])
        sims = linear_kernel(query, self.b["sim_matrix"]).ravel()  # TF-IDF ถูก L2-normalize แล้ว = cosine
        out = []
        for i in np.argsort(-sims)[:top_k]:
            if sims[i] <= 0:
                break
            s = self.b["songs"][int(i)]
            out.append(
                {
                    "title": s["title"],
                    "artist": s["artist"],
                    "url": s["url"],
                    "views": int(s["views"]),
                    "views_text": format_views(s["views"]),
                    "level_name": self.b["class_names"][int(s["level"])],
                    "similarity": round(float(sims[i]), 4),
                }
            )
        return out

    def heatmap(self, lyrics: str, tokens: list[str] | None = None) -> dict:
        tokens = tokenize(lyrics) if tokens is None else tokens
        counts = Counter(tokens)
        n_classes = len(self.b["class_names"])
        uniform = 1.0 / n_classes

        words = []
        for word, count in counts.items():
            j = self.vocab.get(word)
            if j is None:
                continue
            scores = self.word_class[:, j]
            words.append(
                {
                    "word": word,
                    "count": count,
                    "scores": [round(float(s), 4) for s in scores],
                    "top": int(scores.argmax()),
                    "_strength": float(scores.max() - uniform),
                }
            )
        words.sort(key=lambda w: (-w["count"], -w["_strength"]))
        for w in words:
            del w["_strength"]

        lines = split_lines(lyrics)[:MAX_LINES]
        line_tokens = [tokenize(line) for line in lines]
        has_known = [any(t in self.vocab for t in toks) for toks in line_tokens]
        line_rows = [{"text": line, "scores": None, "top": None} for line in lines]
        known_idx = [i for i, ok in enumerate(has_known) if ok]
        if known_idx:
            proba = self.pipe.predict_proba([line_tokens[i] for i in known_idx])
            for i, p in zip(known_idx, proba):
                line_rows[i]["scores"] = [round(float(x), 4) for x in p]
                line_rows[i]["top"] = int(p.argmax())

        return {
            "classes": [f"{c['name']} ({c['range']})" for c in self.classes],
            "class_names": self.b["class_names"],
            "words": words,
            "unknown_words": sorted(w for w in counts if w not in self.vocab),
            "lines": line_rows,
        }
