"""จับคู่เนื้อเพลงกับวิดีโอ YouTube

1) คำค้น: เลือกคำที่ "หายากในภาษาไทยทั่วไป" (ใช้ความถี่คำจาก Thai National Corpus ใน PyThaiNLP)
   และพบบ่อยในเนื้อเพลง เช่น "ภูมิแพ้" ชนะ "พี่", "น้อง" ที่พบได้ในทุกเพลง
2) ให้คะแนนแต่ละวิดีโอจาก 3 สัญญาณ
   - lines : สัดส่วนท่อนเนื้อเพลงที่พบแบบตรงตัวในชื่อ/คำอธิบายวิดีโอ (หลายวิดีโอใส่เนื้อเพลงไว้)
   - title : ชื่อเพลงในชื่อวิดีโอประกอบด้วยคำที่อยู่ในเนื้อเพลง (เพลงส่วนใหญ่ตั้งชื่อจากท่อนฮุก)
   - words : cosine similarity ของคำ ถ่วงน้ำหนักด้วยความหายากของคำ
   คะแนนรวม = max(lines, 0.6 × title + 0.4 × words)
3) รวมวิดีโอของเพลงเดียวกัน (MV / Lyric Video / Cover) ให้เหลือรายการเดียว แสดงตัวที่ยอดวิวสูงสุด
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

from .text import split_lines, tokenize

COMMON_FREQ = 10_000         # คำที่พบใน TNC มากกว่านี้ทั่วไปเกินไปสำหรับใช้เป็นคำค้น (พี่, น้อง, รัก, ผู้หญิง)
RARE_FLOOR = 100             # คำที่พบน้อยกว่านี้มักเป็นเศษจากการตัดคำผิด ("หลอกรัก" -> "หลอ"+"กรัก") หรือชื่อคน
                             # ไม่ใช้เป็นคำค้น และไม่ให้ความหายากเกินระดับนี้
UNKNOWN_THAI_FREQ = 300      # คำไทยที่ไม่มีใน TNC (สแลง/ตัดคำผิด) ถือว่าหายากปานกลาง ไม่ใช่หายากที่สุด
LATIN_FREQ = 3_000           # คำภาษาอังกฤษไม่มีข้อมูลความถี่ ให้ค่ากลาง ๆ
TITLE_RARITY_FULL = 18.0     # ผลรวมความหายากที่ถือว่าชื่อเพลงเจาะจงพอ (ประมาณ 2 คำที่ไม่ธรรมดา)
MIN_LINE_CHARS = 10          # ท่อนสั้นกว่านี้ไม่ใช้เทียบ (เช่น "โอ้โอ้")
THEME_FREQ = (500, 5_000)    # คำหายากปานกลาง: เจาะจงพอบอกธีมเพลง แต่ทั่วไปพอที่เพลงอื่นจะใช้ด้วย
GROUP_RARITY = 15.0         # ชื่อวิดีโอมีคำจากท่อนฮุกที่หายากรวมถึงระดับนี้ = เป็นเพลงเดียวกัน (คนละเวอร์ชัน)
# คำที่มักอยู่ในชื่อวิดีโอแต่ไม่ใช่ชื่อเพลง ไม่นำมาคิดสัดส่วนความตรงของชื่อ
TITLE_NOISE = frozenset(
    """
    official audio video mv music lyric lyrics visualizer cover live teaser trailer karaoke version ver remix
    acoustic instrumental piano guitar tab fingerstyle chords full hd 4k ost feat ft prod session concert
    เนื้อเพลง คาราโอเกะ คอร์ด เวอร์ชัน ต้นฉบับ
    """.split()
)

_NORM = re.compile(r"[^฀-๿a-z0-9]")
_TITLE_SPLIT = re.compile(r"\s+[-–—|:/]\s+|[()\[\]{}【】「」『』|]|\s(?:feat|ft)\.?\s", re.IGNORECASE)


@lru_cache(maxsize=1)
def _tnc() -> tuple[dict[str, int], float]:
    from pythainlp.corpus import tnc

    freqs = dict(tnc.word_freqs())
    return freqs, float(sum(freqs.values()))


@lru_cache(maxsize=1)
def _tnc_dict():
    from pythainlp.util import Trie

    return Trie(_tnc()[0].keys())


@lru_cache(maxsize=50_000)
def word_freq(word: str) -> int:
    freqs = _tnc()[0]
    if word in freqs:
        return freqs[word]
    if word.isascii():
        return LATIN_FREQ
    # คำประสมที่ TNC เก็บแยกคำ เช่น "คนเดียว" = "คน" + "เดียว" -> ทั่วไปเท่ากับส่วนที่หายากที่สุด
    from pythainlp.tokenize import word_tokenize

    parts = [p for p in word_tokenize(word, custom_dict=_tnc_dict(), engine="newmm", keep_whitespace=False) if p.strip()]
    if len(parts) > 1 and all(p in freqs for p in parts):
        return min(freqs[p] for p in parts)
    return UNKNOWN_THAI_FREQ


def rarity(word: str) -> float:
    """ความหายากของคำในภาษาไทยทั่วไป (แบบ IDF): ยิ่งพบน้อยยิ่งค่าสูง"""
    return math.log(_tnc()[1] / (max(word_freq(word), RARE_FLOOR) + 1))


def normalize(text: str) -> str:
    """ตัดช่องว่าง/เครื่องหมายทิ้ง ใช้เทียบข้อความแบบตรงตัว"""
    return _NORM.sub("", str(text).lower())


def _weights(tokens: list[str]) -> dict[str, float]:
    return {w: (1 + math.log(c)) * rarity(w) for w, c in Counter(tokens).items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def search_keywords(tokens: list[str], k: int = 3, lo: int = RARE_FLOOR, hi: int = COMMON_FREQ) -> list[str]:
    """คำค้น YouTube: คำที่ทั้งหายากในภาษาทั่วไป (ความถี่ใน TNC อยู่ระหว่าง lo..hi) และซ้ำบ่อยในเนื้อเพลง"""
    weights = _weights(tokens)
    ranked = sorted(weights, key=lambda w: -weights[w])  # stable: เสมอกันให้คำที่มาก่อนชนะ
    picked = [w for w in ranked if lo <= word_freq(w) <= hi][:k]
    if len(picked) < 2:  # เนื้อเพลงที่มีแต่คำทั่วไป
        picked += [w for w in ranked if w not in picked][: k - len(picked)]
    return picked


def _title_support(title: str, hook_words: set[str]) -> tuple[float, frozenset[str]]:
    """ชื่อวิดีโอแต่ละส่วน (แยกด้วย - | ( ) ฯลฯ) ประกอบด้วยคำที่ซ้ำในเนื้อเพลงมากแค่ไหน
    คืนค่า (คะแนนของส่วนที่ตรงที่สุด, คำที่ตรง) — ใช้คำที่ตรงเป็นกุญแจรวมเวอร์ชันของเพลงเดียวกัน"""
    best, best_words = 0.0, frozenset()
    for segment in _TITLE_SPLIT.split(title):
        words = {w for w in tokenize(segment) if w not in TITLE_NOISE}
        if not words:
            continue
        hits = frozenset(w for w in words if w in hook_words)
        matched = sum(rarity(w) for w in hits)
        coverage = matched / sum(rarity(w) for w in words)
        # หลักคือ "มีคำฮุกที่เจาะจงอยู่ในชื่อ" ส่วนคำเกิน (ชื่อศิลปิน/ช่อง) หักคะแนนได้ไม่เกินครึ่ง
        score = min(1.0, matched / TITLE_RARITY_FULL) * (0.5 + 0.5 * coverage)
        if score > best:
            best, best_words = score, hits
    return best, best_words


def score_videos(lyrics: str, tokens: list[str], videos: list[dict]) -> list[dict]:
    counts = Counter(tokens)
    # ชื่อเพลงมักมาจากท่อนฮุกที่ร้องซ้ำ คำที่โผล่ครั้งเดียว (เช่นเครดิตชื่อศิลปินที่แปะมากับเนื้อเพลง) ไม่นับ
    min_count = 2 if len(tokens) >= 30 else 1
    hook_words = {w for w, c in counts.items() if c >= min_count}
    lyric_weights = _weights(tokens)
    lines = [ln for ln in {normalize(x) for x in split_lines(lyrics)} if len(ln) >= MIN_LINE_CHARS]

    scored = []
    for v in videos:
        text = f"{v['title']}\n{v.get('description', '')}"
        text_norm = normalize(text)
        s_lines = sum(ln in text_norm for ln in lines) / len(lines) if lines else 0.0
        s_title, title_words = _title_support(v["title"], hook_words) if hook_words else (0.0, frozenset())
        s_words = _cosine(lyric_weights, _weights(tokenize(text))) if tokens else 0.0
        blend = 0.6 * s_title + 0.4 * s_words
        score = max(s_lines, blend)
        if s_lines >= 0.3 and s_lines >= blend:
            reason = "lyrics"
        elif s_title >= 0.5:
            reason = "title"
        else:
            reason = "words"
        key_words = sorted(w for w in title_words if word_freq(w) <= COMMON_FREQ)  # ไม่ใช้คำทั่วไปอย่าง "พี่"
        if sum(rarity(w) for w in key_words) >= GROUP_RARITY:
            # "ภูมิแพ้กรุงเทพ (Feat...)", "TEASER MV ภูมิแพ้กรุงเทพ", "...พี่ป้าง ภูมิแพ้กรุงเทพ@Fest" -> กลุ่มเดียวกัน
            group = "|".join(key_words)
        else:
            first_segment = next((s for s in _TITLE_SPLIT.split(v["title"]) if normalize(s)), v["title"])
            group = normalize(first_segment) or v["video_id"]
        # เป็นเพลงเดียวกับที่ผู้ใช้ใส่: คำอธิบายมีเนื้อเพลงครึ่งหนึ่งขึ้นไป หรือชื่อวิดีโอยกท่อนเพลงมา (เช่นคลิปคาราโอเกะ)
        is_input_song = s_lines >= 0.5 or _quotes_lyrics(normalize(v["title"]), lines)
        scored.append({**v, "score": score, "reason": reason, "group": group, "is_input_song": is_input_song,
                       "signals": {"lines": round(s_lines, 3), "title": round(s_title, 3), "words": round(s_words, 3)}})
    return scored


def _quotes_lyrics(title_norm: str, lines: list[str], window: int = 15) -> bool:
    """ชื่อวิดีโอมีข้อความต่อเนื่องยาว window ตัวอักษรขึ้นไปที่ตรงกับท่อนเพลงหรือไม่"""
    for line in lines:
        for i in range(0, max(1, len(line) - window + 1), 5):
            chunk = line[i:i + window]
            if len(chunk) == window and chunk in title_norm:
                return True
    return False


def find_on_youtube(yt, lyrics: str, tokens: list[str], user_query: str = "", candidates: int = 25,
                    min_songs: int = 4) -> tuple[list[dict], list[str], bool]:
    """ค้นบน YouTube แล้วจัดอันดับ คืนค่า (เพลงที่รวมเวอร์ชันแล้ว เรียงตามคะแนน, คำค้นที่ใช้, มาจาก cache ทั้งหมดหรือไม่)

    รอบ 1: คำที่หายากที่สุด -> มักเจอตัวเพลงนั้นเอง (และ cover/live/karaoke ของมัน)
    รอบ 2: ถ้าได้เพลงไม่ถึง min_songs เพลง ค้นต่อด้วยคำ "บอกธีม" 2 คำ (หายากปานกลาง เช่น เมืองหลวง ฝุ่น)
           โดยตัดคำในชื่อเพลงที่เจอแล้ว เพื่อหาเพลงอื่นที่เนื้อหาใกล้เคียง
    ผู้ใช้แก้คำค้นเอง -> ค้นรอบเดียวตามนั้น
    """
    from .youtube import YouTubeError

    queries = [user_query or " ".join(search_keywords(tokens))]
    ids, all_cached = yt.search(queries[0], max_results=candidates)
    infos = yt.videos(ids)
    videos = [infos[i] for i in ids if i in infos]
    grouped = group_versions(score_videos(lyrics, tokens, videos))

    if not user_query and tokens and len(grouped) < min_songs:
        top = grouped[0] if grouped else None
        found_words = set(top["group"].split("|")) if top and top["score"] >= 0.5 else set()
        second = " ".join(search_keywords([t for t in tokens if t not in found_words], k=2, lo=THEME_FREQ[0], hi=THEME_FREQ[1]))
        if second and second != queries[0]:
            try:
                ids2, cached2 = yt.search(second, max_results=candidates)
            except YouTubeError:  # โควตาหมดระหว่างรอบ 2: ใช้ผลรอบแรกไปก่อน
                return grouped, queries, all_cached
            seen = {v["video_id"] for v in videos}
            infos2 = yt.videos([i for i in ids2 if i not in seen])
            videos += [infos2[i] for i in ids2 if i in infos2]
            grouped = group_versions(score_videos(lyrics, tokens, videos))
            queries.append(second)
            all_cached = all_cached and cached2
    return grouped, queries, all_cached


def pick_results(grouped: list[dict], limit: int, min_score: float = 0.1, min_shown: int = 3) -> list[dict]:
    """ตัดผลที่แทบไม่เกี่ยวกับเนื้อเพลงทิ้ง แต่ให้เหลืออย่างน้อย min_shown รายการเป็นตัวเลือก"""
    strong = [v for v in grouped if v["score"] >= min_score]
    return (strong if len(strong) >= min_shown else grouped[:min_shown])[:limit]


def group_versions(scored: list[dict]) -> list[dict]:
    """รวมเพลงเดียวกันหลายเวอร์ชัน: คะแนน = คะแนนสูงสุดในกลุ่ม, แสดงวิดีโอที่ยอดวิวสูงสุด (มักเป็นตัว official)"""
    groups: dict[str, list[dict]] = {}
    for v in scored:
        groups.setdefault(v["group"], []).append(v)
    # ทุกกลุ่มที่มีวิดีโอเป็น "เพลงเดียวกับที่ผู้ใช้ใส่" รวมเป็นกลุ่มเดียว (MV + คาราโอเกะที่ตั้งชื่อตามท่อนเพลง ฯลฯ)
    same = [k for k, members in groups.items() if any(m.get("is_input_song") for m in members)]
    if len(same) > 1:
        merged_members = [m for k in same for m in groups.pop(k)]
        groups = {same[0]: merged_members, **groups}
    merged = []
    for members in groups.values():
        best = max(members, key=lambda x: x["score"])
        shown = max(members, key=lambda x: x["views"] if x["views"] is not None else -1)
        merged.append({**shown, "score": best["score"], "reason": best["reason"],
                       "signals": best["signals"], "versions": len(members)})
    merged.sort(key=lambda x: -round(x["score"], 3))  # stable: เสมอกันคงลำดับความเกี่ยวข้องของ YouTube
    return merged
