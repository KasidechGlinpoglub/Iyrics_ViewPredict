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


def identity(tokens):
    """analyzer ของ scikit-learn สำหรับเอกสารที่ตัดคำไว้แล้ว (ต้องอยู่ระดับโมดูลเพื่อให้ pickle ได้)"""
    return tokens
