"""ค้นเนื้อร้องจาก LRCLIB ด้วยชื่อเพลงและชื่อศิลปิน"""
from __future__ import annotations

import re
from unicodedata import normalize

import requests

API_URL = "https://lrclib.net/api"
USER_AGENT = "Iyrics_ViewPredict/1.0 (https://github.com/KasidechGlinpoglub/Iyrics_ViewPredict)"
LRC_TIMESTAMPS = re.compile(r"^(?:\[\d{1,3}:\d{2}(?:[.:]\d+)?\]\s*)+")


class LRCLIBError(Exception):
    pass


def search(session: requests.Session, title: str, artist: str) -> list[dict]:
    try:
        response = session.get(
            f"{API_URL}/search",
            params={"track_name": title, "artist_name": artist},
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        if response.status_code in {429, 503}:
            retry_after = response.headers.get("Retry-After")
            wait = f" (Retry-After: {retry_after})" if retry_after else ""
            raise LRCLIBError(f"LRCLIB ตอบ HTTP {response.status_code} ให้รอก่อนรันใหม่{wait}")
        response.raise_for_status()
        records = response.json()
    except requests.HTTPError as exc:
        raise LRCLIBError(f"LRCLIB ตอบ HTTP {exc.response.status_code}") from None
    except requests.exceptions.JSONDecodeError:
        raise LRCLIBError("LRCLIB ส่งข้อมูลที่อ่านไม่ได้") from None
    except requests.RequestException:
        raise LRCLIBError("เชื่อมต่อ LRCLIB ไม่สำเร็จ ลองรันใหม่") from None

    if not isinstance(records, list) or any(
        not isinstance(record, dict)
        or type(record.get("id")) is not int
        or not isinstance(record.get("trackName"), str)
        or not isinstance(record.get("artistName"), str)
        or any(record.get(field) is not None and not isinstance(record[field], str)
               for field in ("plainLyrics", "syncedLyrics"))
        for record in records
    ):
        raise LRCLIBError("รูปแบบผลค้นหาจาก LRCLIB ไม่ถูกต้อง")
    return records


def match_lyrics(title: str, artist: str, records: list[dict]) -> tuple[str, str, str]:
    """คืน (สถานะ, เนื้อร้อง, แหล่งที่มา) จากผลค้นหาของชื่อเพลงกับศิลปิน"""
    def key(value: str | None) -> str:
        return " ".join(normalize("NFKC", value or "").casefold().split())

    target = (key(title), key(artist))
    exact = {
        record["id"]: record
        for record in records
        if (key(record.get("trackName")), key(record.get("artistName"))) == target
    }
    if len(exact) > 1 or (records and not exact):
        return "needs_review", "", ""
    if not exact:
        return "not_found", "", ""

    record_id, record = next(iter(exact.items()))
    lyrics = (record.get("plainLyrics") or "").strip()
    if not lyrics:
        lines = (record.get("syncedLyrics") or "").splitlines()
        lyrics = "\n".join(LRC_TIMESTAMPS.sub("", line) for line in lines if LRC_TIMESTAMPS.match(line)).strip()
    if not lyrics or record.get("instrumental"):
        return "not_found", "", ""
    return "matched", lyrics, f"{API_URL}/get/{record_id}"
