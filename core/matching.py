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
from pathlib import Path

from .text import split_lines, tokenize

COMMON_FREQ = 10_000         # คำที่พบใน TNC มากกว่านี้ทั่วไปเกินไปสำหรับใช้เป็นคำค้น (พี่, น้อง, รัก, ผู้หญิง)
RARE_FLOOR = 100             # คำที่พบน้อยกว่านี้มักเป็นเศษจากการตัดคำผิด ("หลอกรัก" -> "หลอ"+"กรัก") หรือชื่อคน
                             # ไม่ใช้เป็นคำค้น และไม่ให้ความหายากเกินระดับนี้
UNKNOWN_THAI_FREQ = 300      # คำไทยที่ไม่มีใน TNC (สแลง/ตัดคำผิด) ถือว่าหายากปานกลาง ไม่ใช่หายากที่สุด
EN_MIN_FREQ = 5              # คำอังกฤษที่หายากกว่านี้ (หรือไม่มีในรายการ) มักเป็นชื่อเฉพาะ/คำสร้อย ไม่ใช้เป็นคำค้น
TITLE_RARITY_FULL = 18.0     # ผลรวมความหายากที่ถือว่าชื่อเพลงเจาะจงพอ (ประมาณ 2 คำที่ไม่ธรรมดา)
MIN_LINE_CHARS = 10          # ท่อนสั้นกว่านี้ไม่ใช้เทียบ (เช่น "โอ้โอ้")
LINE_CHUNK = 10              # เทียบท่อนเพลงทีละช่วง 10 ตัวอักษร: สะกดต่างกันเล็กน้อยก็ยังนับว่าเจอ
LINE_HIT_RATIO = 0.6         # ท่อนเพลงถือว่า "พบ" เมื่อช่วงตัวอักษรตรงกัน 60% ขึ้นไป
THEME_FREQ = (500, 5_000)    # คำหายากปานกลาง: เจาะจงพอบอกธีมเพลง แต่ทั่วไปพอที่เพลงอื่นจะใช้ด้วย
GROUP_RARITY = 15.0         # ชื่อวิดีโอมีคำจากท่อนฮุกที่หายากรวมถึงระดับนี้ = เป็นเพลงเดียวกัน (คนละเวอร์ชัน)
# คำที่มักอยู่ในชื่อวิดีโอแต่ไม่ใช่ชื่อเพลง ไม่นำมาคิดสัดส่วนความตรงของชื่อ
TITLE_NOISE = frozenset(
    """
    official audio video mv music lyric lyrics visualizer cover live teaser trailer karaoke version ver remix
    acoustic instrumental piano guitar tab fingerstyle chords full hd 4k ost feat ft prod session concert
    special performance stage clip original sped slowed reverb tiktok shorts remastered edit mix new song
    color coded romanized eng han rom sub subs thaisub
    เนื้อเพลง คาราโอเกะ คอร์ด เวอร์ชัน เวอร์ชั่น ต้นฉบับ ร้องสด แสดงสด คอนเสิร์ต ฉบับ ฟังเพลง เพลงใหม่ ล่าสุด
    ซับไทย แปลไทย คำแปล
    """.split()
)
# คำที่แทบทุกเพลงใช้ (ไม่ได้บอกว่าเป็นเพลงไหน) ไม่ใช้เป็นคำค้น แต่ยังใช้ให้คะแนนตามปกติ
LYRIC_GENERIC = frozenset(
    """
    love baby heart tonight forever girl boy darling honey lover dream dreams feel feeling night
    หัวใจ ความรัก คิดถึง เสียใจ ร้องไห้ น้ำตา คนดี ที่รัก ความฝัน
    """.split()
)

_DATA = Path(__file__).resolve().parent / "data"

# ข้อความที่อ้างถึง "เพลงอื่น" ในชื่อคลิป เช่น "ปลายทางที่นางรอง [ORIGINAL : คอยทางที่นางรอง]", "(เพลงแก้ ...)"
# ไม่ใช่ชื่อเพลงของคลิปนี้ ตัดออกก่อนเทียบชื่อ
#   "original" ต้องตามด้วย ":" (กันชื่อเพลงที่มีคำนี้จริง เช่น "Original Love")
_REFERENCE = re.compile(
    r"(?:original(?:\s+song)?(?:\s+by)?\s*[:：]|ต้นฉบับ(?:เพลง)?|เพลงแก้|แก้เพลง|ตอบเพลง|answer\s+to)\s*[:：-]?\s*[^\])】」』}]*",
    re.IGNORECASE,
)


def _strip_reference(title: str) -> str:
    return _REFERENCE.sub(" ", title)


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


@lru_cache(maxsize=1)
def _english() -> dict[str, int]:
    """ความถี่คำภาษาอังกฤษ (wordfreq, Zipf) แปลงเป็น "จำนวนครั้งต่อขนาดคลัง TNC" ให้เทียบกับคำไทยได้"""
    scale = _tnc()[1] / 1e9  # zipf = log10(ครั้งต่อพันล้านคำ)
    freqs = {}
    with open(_DATA / "en_word_freq.tsv", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            word, zipf = line.rstrip("\n").split("\t")
            freqs[word] = max(1, round(10 ** float(zipf) * scale))
    return freqs


@lru_cache(maxsize=1)
def _lyric_common() -> frozenset[str]:
    """คำที่เพลงจำนวนมากใช้ (ข้อมูลจากคำอธิบายคลิปเนื้อเพลง + รายการพื้นฐาน) เช่น บ่ อ้าย ฮัก หัวใจ love baby"""
    with open(_DATA / "lyric_common.txt", encoding="utf-8") as f:
        words = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
    return frozenset(words | LYRIC_GENERIC)


@lru_cache(maxsize=50_000)
def word_freq(word: str) -> int:
    """ความถี่ของคำ (สเกลเดียวกับ TNC) — คำที่ทั่วไปในเพลงถือว่าทั่วไป แม้ในภาษาไทยมาตรฐานจะหายาก
    (คำอีสาน "บ่ อ้าย ฮัก" พบน้อยใน TNC แต่อยู่ในเพลงลูกทุ่ง/หมอลำแทบทุกเพลง ใช้แยกเพลงไม่ได้)"""
    base = _base_freq(word)
    return max(base, 2 * COMMON_FREQ) if word in _lyric_common() else base


def _base_freq(word: str) -> int:
    if word.isascii():  # คำอังกฤษใช้คลังภาษาอังกฤษ (TNC มีคำอังกฤษปนอยู่บ้างแต่จำนวนน้อยมาก ใช้ไม่ได้)
        return _english().get(word, 0)  # 0 = ไม่รู้จัก (ชื่อเฉพาะ/คำสร้อย/สะกดแปลก)
    freqs = _tnc()[0]
    if word in freqs:
        return freqs[word]
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


@lru_cache(maxsize=20_000)
def _query_strength(word: str) -> float:
    """คำไทยพยางค์เดียวใช้ค้นได้ไม่ดี (กำกวม/คำลงท้ายภาษาถิ่น อี้ บัก เซา/เศษจากการตัดคำ) ลดน้ำหนักลงครึ่งหนึ่ง
    คำหลายพยางค์ (ภูมิแพ้ บั้งไฟ ปลอกหมอน) และคำภาษาอังกฤษใช้น้ำหนักเต็ม"""
    if word.isascii():
        return 1.0
    from pythainlp.tokenize import syllable_tokenize

    return 0.5 if len(syllable_tokenize(word, engine="dict")) <= 1 else 1.0


def search_keywords(tokens: list[str], k: int = 3, lo: int = RARE_FLOOR, hi: int = COMMON_FREQ) -> list[str]:
    """คำค้น YouTube: คำที่ทั้งหายากในภาษาทั่วไป (ความถี่ใน TNC อยู่ระหว่าง lo..hi) และซ้ำบ่อยในเนื้อเพลง"""
    weights = {w: s * _query_strength(w) for w, s in _weights(tokens).items()}
    ranked = sorted(weights, key=lambda w: -weights[w])  # stable: เสมอกันให้คำที่มาก่อนชนะ

    def usable(w: str) -> bool:
        f = word_freq(w)
        low = min(lo, EN_MIN_FREQ) if w.isascii() else lo  # คำอังกฤษตัดคำด้วยช่องว่าง ไม่มีปัญหาเศษคำ
        # คำทั่วไปในเพลง (บ่ อ้าย หัวใจ love) ถูกกันออกผ่าน word_freq แล้ว; คำแบบ lyrics/official/coded มาจากคำอธิบายคลิป
        return w not in TITLE_NOISE and low <= f <= hi and f > 0

    picked = [w for w in ranked if usable(w)][:k]
    if len(picked) < 2:  # เนื้อเพลงที่มีแต่คำทั่วไป
        picked += [w for w in ranked if w not in picked][: k - len(picked)]
    return picked


def _title_support(title: str, hook_words: set[str]) -> tuple[float, frozenset[str]]:
    """ชื่อวิดีโอแต่ละส่วน (แยกด้วย - | ( ) ฯลฯ) ประกอบด้วยคำที่ซ้ำในเนื้อเพลงมากแค่ไหน
    คืนค่า (คะแนนของส่วนที่ตรงที่สุด, คำที่ตรง) — ใช้คำที่ตรงเป็นกุญแจรวมเวอร์ชันของเพลงเดียวกัน"""
    best, best_words = 0.0, frozenset()
    for segment in _TITLE_SPLIT.split(_strip_reference(title)):
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
    all_lines = [normalize(x) for x in split_lines(lyrics)]
    # ชื่อเพลงมักมาจากท่อนฮุกที่ร้องซ้ำ: ถ้าเนื้อเพลงเขียนท่อนซ้ำครบ ใช้เฉพาะคำที่โผล่ 2 ครั้งขึ้นไป
    # (กันคำที่โผล่ครั้งเดียว เช่นชื่อคน) แต่ถ้าเขียนฮุกครั้งเดียว (เว็บเนื้อเพลงใช้ * แทนการซ้ำ) ใช้ได้ทุกคำ
    written_out = len(all_lines) != len(set(all_lines))
    min_count = 2 if written_out and len(tokens) >= 30 else 1
    hook_words = {w for w, c in counts.items() if c >= min_count}
    lyric_weights = _weights(tokens)
    lines = [(ln, _chunks(ln)) for ln in dict.fromkeys(all_lines) if len(ln) >= MIN_LINE_CHARS]

    scored = []
    for v in videos:
        text = f"{v['title']}\n{v.get('description', '')}"
        text_norm = normalize(text)
        s_lines = sum(_line_found(ln, ch, text_norm) for ln, ch in lines) / len(lines) if lines else 0.0
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
        else:  # ใช้ทุกส่วนของชื่อ ไม่ใช่ส่วนแรก (ชื่อแบบ "ศิลปิน - เพลง" จะรวมทุกเพลงของศิลปินเป็นกลุ่มเดียว)
            group = _title_key(v["title"]) or v["video_id"]
        # เป็นเพลงเดียวกับที่ผู้ใช้ใส่: คำอธิบายมีเนื้อเพลงครึ่งหนึ่งขึ้นไป หรือชื่อวิดีโอยกท่อนเพลงมา (เช่นคลิปคาราโอเกะ)
        is_input_song = s_lines >= 0.5 or _quotes_lyrics(normalize(v["title"]), [ln for ln, _ in lines])
        scored.append({**v, "score": score, "reason": reason, "group": group, "is_input_song": is_input_song,
                       "signals": {"lines": round(s_lines, 3), "title": round(s_title, 3), "words": round(s_words, 3)}})
    return scored


def _title_key(title: str) -> str:
    """กุญแจรวมเวอร์ชันจากชื่อวิดีโอ: ทุกส่วนของชื่อ (ตัดคำทั่วไปอย่าง Official/MV) เรียงตามตัวอักษร
    "แกล้ง - Silly Fools【MV】" กับ "Silly Fools - แกล้ง (Lyric)" ได้กุญแจเดียวกัน"""
    parts = {"".join(w for w in tokenize(seg) if w not in TITLE_NOISE) for seg in _TITLE_SPLIT.split(_strip_reference(title))}
    return "|".join(sorted(p for p in parts if p))


def _chunks(line: str) -> list[str]:
    """แบ่งท่อนเพลงเป็นช่วงละ LINE_CHUNK ตัวอักษร (ซ้อนกันครึ่งช่วง) ครอบคลุมทั้งท่อน"""
    if len(line) <= LINE_CHUNK:
        return [line]
    starts = list(range(0, len(line) - LINE_CHUNK + 1, LINE_CHUNK // 2))
    if starts[-1] != len(line) - LINE_CHUNK:
        starts.append(len(line) - LINE_CHUNK)
    return [line[i:i + LINE_CHUNK] for i in starts]


def _line_found(line: str, chunks: list[str], text_norm: str) -> bool:
    """ท่อนเพลงอยู่ในข้อความไหม — ยอมให้สะกด/เว้นวรรค/คำสร้อยต่างกันเล็กน้อย"""
    if line in text_norm:
        return True
    return sum(c in text_norm for c in chunks) >= LINE_HIT_RATIO * len(chunks)


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


def pick_results(grouped: list[dict], limit: int, min_score: float = 0.1, min_shown: int = 3,
                 min_views: int = 1_000) -> list[dict]:
    """ตัดผลที่แทบไม่เกี่ยวกับเนื้อเพลงทิ้ง แต่ให้เหลืออย่างน้อย min_shown รายการเป็นตัวเลือก
    เพลงอื่น (ไม่ใช่เพลงที่ผู้ใช้ใส่) ต้อง (1) YouTube ไม่ได้จัดว่าเป็นหัวข้ออื่นที่ไม่ใช่เพลง (เช่น Health)
    และ (2) มีคนดูอย่างน้อย min_views ครั้ง — กันคลิปความรู้/โฆษณา/คลิปส่วนตัวที่ผู้อัปโหลดตั้งหมวดเป็น Music"""
    def is_candidate(v: dict) -> bool:
        if v.get("is_input_song"):
            return True
        if v.get("is_music") is False:
            return False
        return v["views"] is None or v["views"] >= min_views

    pool = [v for v in grouped if is_candidate(v)]
    strong = [v for v in pool if v["score"] >= min_score]
    return (strong if len(strong) >= min_shown else pool[:min_shown])[:limit]


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
        merged.append({**shown, "score": best["score"], "reason": best["reason"], "signals": best["signals"],
                       "versions": len(members), "is_input_song": any(m.get("is_input_song") for m in members)})
    merged.sort(key=lambda x: -round(x["score"], 3))  # stable: เสมอกันคงลำดับความเกี่ยวข้องของ YouTube
    return merged
