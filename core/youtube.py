"""เชื่อมต่อ YouTube Data API v3 (API ทางการของ Google) เพื่อดึงข้อมูลแบบ real-time

ค่าโควตา (ฟรี 10,000 หน่วย/วัน, รีเซ็ตเที่ยงคืนเวลาแปซิฟิก):
- search.list         100 หน่วย/ครั้ง   -> จำกัดจำนวนค้นหาต่อวันด้วย YT_DAILY_SEARCH_LIMIT
- videos.list           1 หน่วย/50 วิดีโอ
- playlistItems.list    1 หน่วย/50 รายการ
มี cache ในหน่วยความจำ: ผลค้นหาเก็บ 6 ชม., ยอดวิวเก็บ 10 นาที เพื่อประหยัดโควตา
"""
from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

import requests

API_URL = "https://www.googleapis.com/youtube/v3"
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_VIDEO_IN_URL = re.compile(r"(?:[?&]v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})")
_PLAYLIST_IN_URL = re.compile(r"[?&]list=([A-Za-z0-9_-]+)")
_PACIFIC = timezone(timedelta(hours=-8))  # ประมาณเวลารีเซ็ตโควตา (ไม่คิด DST)


class YouTubeError(Exception):
    """ข้อความใน exception นี้แสดงให้ผู้ใช้เห็นได้ (ไม่มี API key ปนอยู่)"""

    def __init__(self, message: str, code: str = "api_error"):
        super().__init__(message)
        self.code = code


def extract_video_id(url_or_id: str) -> str | None:
    s = str(url_or_id or "").strip()
    if _VIDEO_ID.match(s):
        return s
    m = _VIDEO_IN_URL.search(s)
    return m.group(1) if m else None


def extract_playlist_id(url_or_id: str) -> str:
    s = str(url_or_id or "").strip()
    m = _PLAYLIST_IN_URL.search(s)
    return m.group(1) if m else s


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


class TTLCache:
    def __init__(self, ttl_seconds: float, maxsize: int):
        self.ttl, self.maxsize = ttl_seconds, maxsize
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires, value = item
            if expires < time.monotonic():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key, value) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)


class YouTubeClient:
    def __init__(self, api_key: str, daily_search_limit: int = 90, timeout: float = 10.0,
                 session: requests.Session | None = None):
        self._key = api_key
        self.daily_search_limit = daily_search_limit
        self.timeout = timeout
        self.session = session or requests.Session()
        self._search_cache = TTLCache(6 * 3600, 500)
        self._video_cache = TTLCache(10 * 60, 2000)
        self._count_lock = threading.Lock()
        self._count_day = None
        self._count = 0

    # ------------------------------------------------------------ low level
    def _get(self, endpoint: str, params: dict) -> dict:
        try:
            r = self.session.get(f"{API_URL}/{endpoint}", params={**params, "key": self._key}, timeout=self.timeout)
        except requests.RequestException:
            # ห้ามส่งข้อความ exception ต่อ เพราะมี URL ที่มี key ปนอยู่
            raise YouTubeError("เชื่อมต่อ YouTube ไม่ได้ ลองใหม่อีกครั้ง", "network") from None
        if r.status_code == 200:
            return r.json()
        raise self._to_error(r)

    @staticmethod
    def _to_error(r: requests.Response) -> YouTubeError:
        try:
            err = r.json().get("error", {}) or {}
        except ValueError:
            err = {}
        reasons = {e.get("reason", "") for e in err.get("errors", []) or []}
        reasons |= {d.get("reason", "") for d in err.get("details", []) or [] if isinstance(d, dict)}
        message = str(err.get("message", ""))
        if reasons & {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded", "RATE_LIMIT_EXCEEDED"}:
            return YouTubeError("โควตา YouTube API ของวันนี้หมดแล้ว (รีเซ็ตประมาณ 14:00–15:00 น. เวลาไทย)", "quota")
        if reasons & {"keyInvalid", "API_KEY_INVALID"} or "API key not valid" in message:
            return YouTubeError("YOUTUBE_API_KEY ไม่ถูกต้อง ตรวจสอบ key ในไฟล์ .env", "bad_key")
        if reasons & {"accessNotConfigured", "SERVICE_DISABLED"} or "has not been used" in message:
            return YouTubeError("ยังไม่ได้เปิดใช้ YouTube Data API v3 ในโปรเจกต์ Google Cloud ของ key นี้", "not_enabled")
        if r.status_code == 403:
            return YouTubeError("API key นี้ไม่มีสิทธิ์เรียก YouTube Data API (ตรวจ API restrictions ของ key)", "forbidden")
        return YouTubeError(f"YouTube API ตอบกลับผิดพลาด ({r.status_code})", "api_error")

    def _take_search_quota(self) -> None:
        today = datetime.now(_PACIFIC).date()
        with self._count_lock:
            if self._count_day != today:
                self._count_day, self._count = today, 0
            if self._count >= self.daily_search_limit:
                raise YouTubeError(
                    f"ค้นหาบน YouTube ครบ {self.daily_search_limit} ครั้งที่ตั้งไว้สำหรับวันนี้แล้ว", "quota")
            self._count += 1

    @property
    def searches_today(self) -> int:
        with self._count_lock:
            return self._count if self._count_day == datetime.now(_PACIFIC).date() else 0

    # ------------------------------------------------------------ public
    def search(self, query: str, max_results: int = 15, region: str = "TH", language: str = "th") -> tuple[list[str], bool]:
        """ค้นหาวิดีโอเพลง คืนค่า (video_ids, มาจาก cache หรือไม่)"""
        key = (" ".join(query.lower().split()), max_results, region, language)
        hit = self._search_cache.get(key)
        if hit is not None:
            return hit, True
        self._take_search_quota()
        data = self._get("search", {
            "part": "snippet",
            "q": query,
            "type": "video",
            "videoCategoryId": "10",  # หมวด Music
            "maxResults": max_results,
            "regionCode": region,
            "relevanceLanguage": language,
            "safeSearch": "moderate",
        })
        ids = [it["id"]["videoId"] for it in data.get("items", []) if it.get("id", {}).get("videoId")]
        self._search_cache.set(key, ids)
        return ids, False

    def videos(self, ids: list[str]) -> dict[str, dict]:
        """ยอดวิว/ไลก์/ชื่อ/คำอธิบาย ของวิดีโอ (สูงสุด 50 ต่อ 1 หน่วยโควตา)"""
        ids = list(dict.fromkeys(i for i in ids if i))
        out, missing = {}, []
        for vid in ids:
            hit = self._video_cache.get(vid)
            if hit is None:
                missing.append(vid)
            else:
                out[vid] = hit
        for start in range(0, len(missing), 50):
            chunk = missing[start:start + 50]
            data = self._get("videos", {"part": "snippet,statistics", "id": ",".join(chunk), "maxResults": 50})
            for item in data.get("items", []):
                info = _video_info(item)
                self._video_cache.set(info["video_id"], info)
                out[info["video_id"]] = info
        return out

    def playlist_video_ids(self, playlist: str, limit: int = 500) -> list[str]:
        playlist_id = extract_playlist_id(playlist)
        ids, token = [], None
        while len(ids) < limit:
            params = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": 50}
            if token:
                params["pageToken"] = token
            data = self._get("playlistItems", params)
            ids += [it["contentDetails"]["videoId"] for it in data.get("items", []) if it.get("contentDetails")]
            token = data.get("nextPageToken")
            if not token:
                break
        return ids[:limit]


def _video_info(item: dict) -> dict:
    sn = item.get("snippet", {}) or {}
    st = item.get("statistics", {}) or {}
    thumbs = sn.get("thumbnails", {}) or {}
    thumb = (thumbs.get("medium") or thumbs.get("default") or {}).get("url", "")

    def as_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    return {
        "video_id": item["id"],
        "url": watch_url(item["id"]),
        "title": sn.get("title", ""),
        "channel": sn.get("channelTitle", ""),
        "description": sn.get("description", ""),
        "published_at": sn.get("publishedAt", ""),
        "thumbnail": thumb,
        "views": as_int(st.get("viewCount")),
        "likes": as_int(st.get("likeCount")),
    }
