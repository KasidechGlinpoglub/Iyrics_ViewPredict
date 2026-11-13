"""การเตรียมข้อความ (NLP preprocessing) สำหรับเนื้อเพลงไทย/อังกฤษ

ขั้นตอน: normalize -> ลบอักขระที่ไม่ใช่ตัวอักษร -> ตัดคำด้วย PyThaiNLP (newmm)
-> ตัด stopwords / คำสั้นเกินไป / ตัวเลข
"""
from __future__ import annotations

import re

from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize

EN_STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at for with from by as is am are was were be been
    being i me my mine you your yours he him his she her it its we us our they them their
    this that these those so do does did done not no yes just can could will would shall
    should have has had there here what when where who how all any up down out into over
    oh ooh uh yeah yea hey ah eh la na da woo whoa ha
    don won wouldn couldn shouldn didn doesn isn aren wasn weren ain haven hasn hadn ve ll re im
    gonna wanna gotta cause cuz till ya yo
    """.split()
)
# คำสร้อย/คำอุทานที่พบบ่อยในเพลงไทย ไม่ช่วยบอกยอดวิว
TH_FILLERS = frozenset(["โอ้", "เย้", "ฮู้ว", "ลาลา", "นานา", "อา", "เอ๋ย", "เอย", "นะ", "จ้า", "ล่ะ"])
TH_STOPWORDS = frozenset(thai_stopwords()) | TH_FILLERS

_URL = re.compile(r"https?://\S+")
_NOT_TEXT = re.compile(r"[^฀-๿a-z0-9\s']")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = normalize(str(text)).lower()
    text = _URL.sub(" ", text)
    text = text.replace("ๆ", " ")  # ไม้ยมก: "ซ้ำๆ" -> "ซ้ำ"
    text = _NOT_TEXT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def _keep(token: str) -> bool:
    if len(token) < 2 or token.isdigit():
        return False
    return token not in TH_STOPWORDS and token not in EN_STOPWORDS


def tokenize(text: str) -> list[str]:
    """ตัดคำเนื้อเพลง คืนค่าเป็นลิสต์ของคำที่ใช้เป็น feature"""
    tokens = word_tokenize(clean_text(text), engine="newmm", keep_whitespace=False)
    return [t for t in (tok.strip().strip("'") for tok in tokens) if _keep(t)]


def split_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text).splitlines() if line.strip()]


# ---- ทำความสะอาดเนื้อเพลงที่ผู้ใช้วางมา (จากเว็บเนื้อเพลง/คำอธิบาย YouTube)
_SECTION = (r"(?:pre[- ]?chorus|post[- ]?chorus|chorus|verse|hook|bridge|intro|outro|refrain|rap|interlude|"
            r"ท่อนฮุก|ฮุก|ท่อนแร็พ|แร็พ|ท่อน\s*\d*|ซ้ำ(?:\s*\*+)?|\*+)")
_LABEL_LINE = re.compile(rf"^[\s\[(（【{{]*{_SECTION}[\s\d:：.)\]）】}}x×*]*$", re.IGNORECASE)
_LABEL_PREFIX = re.compile(rf"^\s*(?:[\[(（【]\s*{_SECTION}[^\])）】]*[\])）】]|{_SECTION}\s*[:：]|\*+)\s*", re.IGNORECASE)
_REPEAT_MARK = re.compile(r"[\[(（]\s*(?:ซ้ำ|repeat|x\s*\d|×\s*\d|\*+)[^\])）]*[\])）]|\s[x×]\s?\d\s*$", re.IGNORECASE)
_CREDIT = re.compile(
    r"^\s*(?:ศิลปิน|นักร้อง|ร้องโดย|เพลง|ชื่อเพลง|คำร้อง|ทำนอง|เรียบเรียง|โปรดิวซ์|ค่าย|อัลบั้ม|ต้นฉบับ|"
    r"artist|singer|title|song|lyrics?(?:\s+by)?|music(?:\s+by)?|composer|composed by|written by|"
    r"produced by|producer|prod\.?|arranged by|arranger|album|label|feat\.?|ft\.?)\s*[:：/|]",
    re.IGNORECASE,
)
_JUNK_LINE = re.compile(r"https?://|www\.|^#|@\w|©|℗", re.IGNORECASE)
# เครดิตที่ไม่มีเครื่องหมาย ":" เช่น "Edited by …", "ผู้กำกับ …", "Rapper X Feat. Y"
_CREDIT_FREE = re.compile(
    r"\b(?:edited|mixed|mastered|recorded|directed|produced|written|composed|arranged|performed|shot|filmed|"
    r"music|lyrics|video|cover(?:ed)?|sung)\s+by\b|"
    r"^\s*(?:ผู้กำกับ|กำกับ|ตัดต่อ|ถ่ายภาพ|ผู้แต่ง|แต่งเพลง|ผู้ประพันธ์|ประพันธ์|เรียบเรียง|โปรดิวเซอร์|มิกซ์|"
    r"มาสเตอร์|บันทึกเสียง|director|editor|colorist|dop)\b",
    re.IGNORECASE,
)
_FEAT = re.compile(r"\b(?:feat|ft)\.?\s", re.IGNORECASE)
# บรรทัดหัวเพลงแบบ "ชื่อเพลง - ศิลปิน" / "Artist - Title (Lyrics)": ข้อความสั้นสองฝั่งคั่นด้วย - – | (มีช่องว่างรอบ)
_HEADER_LINE = re.compile(r"^[^-–—|]{1,45}\s[-–—|]\s[^-–—|]{1,60}$")


def clean_lyrics(text: str) -> str:
    """ตัดสิ่งที่ไม่ใช่เนื้อร้อง: ป้ายท่อน ([Chorus], ท่อนฮุก:, *), เครื่องหมายซ้ำ ((ซ้ำ *), x2),
    บรรทัดเครดิต (คำร้อง: / Composer:) และลิงก์/แฮชแท็ก"""
    out = []
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line or _LABEL_LINE.match(line) or _CREDIT.match(line) or _JUNK_LINE.search(line):
            continue
        if _HEADER_LINE.match(line) and len(line.split()) <= 10:  # หัวเพลง ไม่ใช่เนื้อร้อง
            continue
        if _CREDIT_FREE.search(line) or (_FEAT.search(line) and len(line.split()) <= 8):  # เครดิต/รายชื่อศิลปิน
            continue
        line = _REPEAT_MARK.sub(" ", _LABEL_PREFIX.sub("", line)).strip(" *-–—")
        if line:
            out.append(line)
    return "\n".join(out)


def identity(tokens):
    """analyzer ของ scikit-learn สำหรับเอกสารที่ตัดคำไว้แล้ว (ต้องอยู่ระดับโมดูลเพื่อให้ pickle ได้)"""
    return tokens
