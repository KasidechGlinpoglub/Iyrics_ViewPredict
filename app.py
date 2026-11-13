"""เว็บแอปทำนายยอดวิว YouTube จากเนื้อเพลง (Naive Bayes + NLP) + ข้อมูลสดจาก YouTube Data API

รันในเครื่อง:   python app.py   แล้วเปิด http://127.0.0.1:5000
รันบนเซิร์ฟเวอร์: gunicorn app:app   (ดู README หัวข้อ Deploy)
"""
from __future__ import annotations

import os
import statistics
import sys
import threading
import time
from collections import defaultdict, deque

from flask import Flask, jsonify, render_template, request, send_file

from core.config import BASE_DIR, env_int  # โหลด .env ก่อนส่วนอื่น
from core.export import build_workbook
from core.model import LyricsModel, format_views, text_similarity
from core.text import tokenize
from core.youtube import YouTubeClient, YouTubeError, extract_video_id

MODEL_PATH = BASE_DIR / "models" / "lyrics_nb.joblib"
MAX_CHARS = 20_000
YT_RESULTS_SHOWN = 6

app = Flask(__name__)
app.json.ensure_ascii = False
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024

if os.environ.get("TRUST_PROXY") == "1":  # อยู่หลัง reverse proxy (Render / Hugging Face) ให้ได้ IP จริงของผู้ใช้
    from werkzeug.middleware.proxy_fix import ProxyFix

    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

_api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
youtube = YouTubeClient(_api_key, daily_search_limit=env_int("YT_DAILY_SEARCH_LIMIT", 90)) if _api_key else None
_model: LyricsModel | None = None


class RateLimiter:
    """จำกัดจำนวนครั้งต่อ IP ในช่วงเวลา (sliding window) กันคนยิงจนโควตา YouTube หมด"""

    def __init__(self, limit: int, window_seconds: float):
        self.limit, self.window = limit, window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            if len(self._hits) > 5000:
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > self.window]:
                    del self._hits[k]
            return True


search_limiter = RateLimiter(env_int("YT_SEARCH_PER_IP", 20), 10 * 60)
stats_limiter = RateLimiter(60, 10 * 60)


def get_model() -> LyricsModel:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("ยังไม่มีโมเดล — รัน python train.py ก่อน")
        _model = LyricsModel.load(MODEL_PATH)
    return _model


def _json_body() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _lyrics_from(data: dict, required: bool = True) -> str:
    lyrics = str(data.get("lyrics", "") or "").strip()
    if required and not lyrics:
        raise ValueError("กรุณาใส่เนื้อเพลง")
    if len(lyrics) > MAX_CHARS:
        raise ValueError(f"เนื้อเพลงยาวเกิน {MAX_CHARS:,} ตัวอักษร")
    return lyrics


def _require_youtube(limiter: RateLimiter) -> YouTubeClient:
    if youtube is None:
        raise YouTubeError("ยังไม่ได้ตั้งค่า YOUTUBE_API_KEY (ดูวิธีใน README)", "no_key")
    if not limiter.allow(request.remote_addr or "unknown"):
        raise YouTubeError("ค้นหาถี่เกินไป รอสักครู่แล้วลองใหม่", "rate_limited")
    return youtube


# ------------------------------------------------------------------ errors
@app.errorhandler(ValueError)
def _bad_request(err):
    return jsonify(error=str(err)), 400


@app.errorhandler(FileNotFoundError)
def _no_model(err):
    return jsonify(error=str(err)), 503


@app.errorhandler(YouTubeError)
def _youtube_error(err: YouTubeError):
    status = {"no_key": 503, "quota": 429, "rate_limited": 429}.get(err.code, 502)
    return jsonify(error=str(err), code=err.code), status


# ------------------------------------------------------------------ pages & model API
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/info")
def info():
    data = get_model().info()
    data["youtube"] = {
        "enabled": youtube is not None,
        "searches_today": youtube.searches_today if youtube else 0,
        "daily_search_limit": youtube.daily_search_limit if youtube else 0,
    }
    return jsonify(data)


@app.get("/api/example")
def example():
    return jsonify(get_model().random_example())


@app.post("/api/predict")
def predict():
    return jsonify(get_model().predict(_lyrics_from(_json_body())))


@app.post("/api/export/xlsx")
def export_xlsx():
    data = _json_body()
    lyrics = _lyrics_from(data)
    result = get_model().predict(lyrics)
    yt = data.get("youtube") if isinstance(data.get("youtube"), list) else []
    return send_file(
        build_workbook(result, lyrics, youtube_results=yt[:YT_RESULTS_SHOWN]),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="lyrics_heatmap.xlsx",
    )


# ------------------------------------------------------------------ YouTube (real-time)
@app.post("/api/youtube/search")
def youtube_search():
    """ค้นเพลงจริงบน YouTube จากคำเด่นในเนื้อเพลง แล้วเรียงตามความคล้ายของเนื้อเพลงกับชื่อ+คำอธิบายวิดีโอ"""
    data = _json_body()
    lyrics = _lyrics_from(data, required=False)
    tokens = tokenize(lyrics) if lyrics else []
    keywords = get_model().keywords(tokens) if tokens else []
    query = " ".join(str(data.get("query", "") or "").split())[:100] or " ".join(keywords)
    if not query:
        raise ValueError("ไม่มีคำค้น ใส่เนื้อเพลงหรือคำค้นก่อน")

    yt = _require_youtube(search_limiter)
    ids, cached = yt.search(query)
    infos = yt.videos(ids)
    videos = [infos[i] for i in ids if i in infos]
    sims = text_similarity(tokens, [f"{v['title']}\n{v['description']}" for v in videos])

    ranked = sorted(zip(videos, sims), key=lambda vs: -round(vs[1], 3))[:YT_RESULTS_SHOWN]  # stable: เสมอกันคงลำดับของ YouTube
    results = [
        {
            "title": v["title"],
            "channel": v["channel"],
            "url": v["url"],
            "thumbnail": v["thumbnail"],
            "published_at": v["published_at"],
            "views": v["views"],
            "views_text": format_views(v["views"]) if v["views"] is not None else "ซ่อนยอดวิว",
            "likes": v["likes"],
            "lyric_similarity": round(s, 4),
        }
        for v, s in ranked
    ]
    views = [r["views"] for r in results if r["views"] is not None]
    median = statistics.median(views) if views else None
    return jsonify(
        query=query,
        keywords=keywords,
        results=results,
        median_views=median,
        median_views_text=format_views(median) if median is not None else None,
        cached=cached,
        fetched_at=time.strftime("%H:%M:%S"),
    )


@app.post("/api/youtube/stats")
def youtube_stats():
    """ยอดวิวล่าสุดของวิดีโอตามลิงก์ (ใช้กับเพลงที่คล้ายกันในชุดข้อมูล)"""
    urls = _json_body().get("urls")
    if not isinstance(urls, list):
        raise ValueError("ต้องส่ง urls เป็นลิสต์")
    pairs = [(str(u), extract_video_id(str(u))) for u in urls[:50]]
    pairs = [(u, vid) for u, vid in pairs if vid]
    if not pairs:
        return jsonify(stats={})
    infos = _require_youtube(stats_limiter).videos([vid for _, vid in pairs])
    stats = {}
    for url, vid in pairs:
        v = infos.get(vid)
        if v and v["views"] is not None:
            stats[url] = {"views": v["views"], "views_text": format_views(v["views"]), "title": v["title"]}
    return jsonify(stats=stats, fetched_at=time.strftime("%H:%M:%S"))


# ------------------------------------------------------------------ warm-up
def _warm_up() -> None:
    """โหลดโมเดลและพจนานุกรมตัดคำไว้ก่อน ให้คำขอแรกไม่ช้า (ทำงานทั้งตอนรัน python app.py และ gunicorn)"""
    if MODEL_PATH.exists():
        get_model().predict("ทดสอบระบบ รัก")


_warm_up()

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console Windows บางเครื่องพิมพ์ภาษาไทยไม่ได้
    port = env_int("PORT", 5000)
    print(f"YouTube API: {'เปิดใช้งาน' if youtube else 'ยังไม่ได้ตั้งค่า YOUTUBE_API_KEY'}")
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=False)
