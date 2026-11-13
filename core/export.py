"""Export ผลวิเคราะห์เป็นไฟล์ Excel พร้อมตาราง heatmap (ระบายสีตามค่าความน่าจะเป็น)"""
from __future__ import annotations

from datetime import datetime
from io import BytesIO

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# sequential ramp (ฟ้าอ่อน -> น้ำเงินเข้ม) ใช้ชุดเดียวกับหน้าเว็บ
RAMP = [
    "CDE2FB", "B7D3F6", "9EC5F4", "86B6EF", "6DA7EC", "5598E7", "3987E5",
    "2A78D6", "256ABF", "1C5CAB", "184F95", "104281", "0D366B",
]
WHITE_TEXT_FROM = 7  # index ใน RAMP ที่เริ่มใช้ตัวอักษรสีขาว
INK = "0B0B0B"
MUTED_FILL = "F0EFEC"
HEADER_FILL = "1A1A19"
LINK = "256ABF"
FONT = "Tahoma"
THIN = Side(style="thin", color="FFFFFF")


def _put(ws, row: int, col: int, value=None, **font):
    """เขียนค่าลงเซลล์ ข้อความที่ขึ้นต้นด้วย '=' ต้องเป็นข้อความ ไม่ใช่สูตร (กัน formula injection)"""
    cell = ws.cell(row=row, column=col)
    cell.value = value
    if isinstance(value, str) and value.startswith("="):
        cell.data_type = "s"
    if font:
        cell.font = Font(name=FONT, **font)
    return cell


def _link(ws, row: int, col: int, url: str):
    cell = _put(ws, row, col, url)
    if isinstance(url, str) and url.startswith(("https://", "http://")):
        cell.hyperlink = url
        cell.font = Font(name=FONT, color=LINK, underline="single")
    return cell


def _heat(cell, value: float | None):
    cell.border = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    if value is None:
        cell.fill = PatternFill("solid", fgColor=MUTED_FILL)
        cell.font = Font(name=FONT, color="898781")
        return
    i = min(len(RAMP) - 1, max(0, round(value * (len(RAMP) - 1))))
    cell.fill = PatternFill("solid", fgColor=RAMP[i])
    cell.font = Font(name=FONT, color="FFFFFF" if i >= WHITE_TEXT_FROM else INK)
    cell.number_format = "0.0%"


def _header(ws, row: int, values: list[str]):
    for col, v in enumerate(values, start=1):
        c = _put(ws, row, col, v, bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor=HEADER_FILL)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[row].height = 34


def _title(ws, text: str, sub: str):
    _put(ws, 1, 1, text, bold=True, size=14)
    _put(ws, 2, 1, sub, color="52514E")


def _widths(ws, widths: dict[str, float]):
    for col, w in widths.items():
        ws.column_dimensions[col].width = w


def build_workbook(result: dict, lyrics: str, youtube_results: list | None = None) -> BytesIO:
    hm = result["heatmap"]
    pred = result["prediction"]
    names = hm["class_names"]
    n = len(names)
    stamp = datetime.now().strftime("%d/%m/%Y %H:%M")
    wb = Workbook()

    # ---- Sheet 1: heatmap รายคำ
    ws = wb.active
    ws.title = "Heatmap รายคำ"
    _title(ws, "Heatmap รายคำ: คำนี้สัมพันธ์กับยอดวิวระดับไหน (P(ระดับ | คำ) จาก Naive Bayes)",
           f"สร้างเมื่อ {stamp} · แต่ละแถวรวมกัน = 100% · สีเข้ม = สัมพันธ์มาก")
    _header(ws, 4, ["คำ", "ความถี่"] + hm["classes"] + ["แนวโน้ม"])
    for r, w in enumerate(hm["words"], start=5):
        _put(ws, r, 1, w["word"])
        _put(ws, r, 2, w["count"]).alignment = Alignment(horizontal="center")
        for k, s in enumerate(w["scores"]):
            _heat(_put(ws, r, 3 + k, s), s)
        _put(ws, r, 3 + n, names[w["top"]], bold=True)
    if hm["unknown_words"]:
        r = 6 + len(hm["words"])
        _put(ws, r, 1, "คำที่ไม่อยู่ในคลังคำของโมเดล:", italic=True)
        _put(ws, r, 2, ", ".join(hm["unknown_words"]), color="898781")
    _widths(ws, {"A": 22, "B": 9, get_column_letter(3 + n): 12})
    for k in range(n):
        ws.column_dimensions[get_column_letter(3 + k)].width = 20
    ws.freeze_panes = "B5"

    # ---- Sheet 2: heatmap รายบรรทัด
    ws2 = wb.create_sheet("Heatmap รายบรรทัด")
    _title(ws2, "Heatmap รายบรรทัด: ความน่าจะเป็นของระดับยอดวิวเมื่อดูทีละบรรทัด",
           "บรรทัดสีเทา = ไม่มีคำที่โมเดลรู้จัก")
    _header(ws2, 4, ["#", "เนื้อเพลง"] + hm["classes"] + ["แนวโน้ม"])
    for r, line in enumerate(hm["lines"], start=5):
        _put(ws2, r, 1, r - 4).alignment = Alignment(horizontal="center")
        _put(ws2, r, 2, line["text"])
        scores = line["scores"] or [None] * n
        for k, s in enumerate(scores):
            _heat(_put(ws2, r, 3 + k, s if s is not None else "–"), s)
        _put(ws2, r, 3 + n, names[line["top"]] if line["top"] is not None else "–", bold=True)
    _widths(ws2, {"A": 5, "B": 48, get_column_letter(3 + n): 12})
    for k in range(n):
        ws2.column_dimensions[get_column_letter(3 + k)].width = 20
    ws2.freeze_panes = "C5"

    # ---- Sheet 3: สรุปผลทำนาย + เพลงใกล้เคียง + ผลค้นหา YouTube
    ws3 = wb.create_sheet("สรุปผลทำนาย")
    _title(ws3, "ผลทำนายยอดวิว YouTube จากเนื้อเพลง", f"สร้างเมื่อ {stamp}")
    rows = [
        ("ยอดวิวที่คาด", pred["views"], "#,##0"),
        ("ยอดวิวที่คาด (ข้อความ)", pred["views_text"], None),
        ("ระดับที่คาด", f"{pred['level_name']} ({pred['level_range']})", None),
        ("ความมั่นใจ", pred["confidence"], "0.0%"),
        ("คำที่โมเดลรู้จัก", f"{result['tokens']['known']} / {result['tokens']['total']} คำ", None),
    ]
    for r, (label, value, fmt) in enumerate(rows, start=4):
        _put(ws3, r, 1, label, bold=True)
        c = _put(ws3, r, 2, value)
        if fmt:
            c.number_format = fmt

    r = 4 + len(rows) + 1
    _header(ws3, r, ["ระดับยอดวิว", "ความน่าจะเป็น"])
    for p in result["probabilities"]:
        r += 1
        _put(ws3, r, 1, f"{p['name']} ({p['range']})")
        _heat(_put(ws3, r, 2, p["prob"]), p["prob"])

    r += 2
    _header(ws3, r, ["เพลงที่เนื้อหาใกล้เคียง (ชุดข้อมูล)", "ศิลปิน", "ยอดวิว", "ความคล้าย", "ลิงก์"])
    for s in result["similar"]:
        r += 1
        _put(ws3, r, 1, s["title"])
        _put(ws3, r, 2, s["artist"])
        _put(ws3, r, 3, s["views"]).number_format = "#,##0"
        _put(ws3, r, 4, s["similarity"]).number_format = "0.0%"
        if s["url"]:
            _link(ws3, r, 5, s["url"])

    yt = [y for y in (youtube_results or []) if isinstance(y, dict)]
    if yt:
        r += 2
        _header(ws3, r, ["ผลค้นหาบน YouTube (ยอดวิวสด)", "ช่อง", "ยอดวิว", "คล้ายเนื้อเพลง", "ลิงก์"])
        for y in yt:
            r += 1
            _put(ws3, r, 1, str(y.get("title", "")))
            _put(ws3, r, 2, str(y.get("channel", "")))
            views = y.get("views")
            _put(ws3, r, 3, views if isinstance(views, int) else "–").number_format = "#,##0"
            sim = y.get("lyric_similarity")
            _put(ws3, r, 4, sim if isinstance(sim, (int, float)) else "–").number_format = "0.0%"
            _link(ws3, r, 5, str(y.get("url", "")))

    r += 2
    _put(ws3, r, 1, "เนื้อเพลงที่วิเคราะห์", bold=True)
    c = _put(ws3, r + 1, 1, lyrics.strip())
    c.alignment = Alignment(wrap_text=True, vertical="top")
    c.font = Font(name=FONT)
    ws3.merge_cells(start_row=r + 1, start_column=1, end_row=r + 1, end_column=5)
    ws3.row_dimensions[r + 1].height = min(409, 15 * (lyrics.count("\n") + 2))
    _widths(ws3, {"A": 38, "B": 22, "C": 16, "D": 14, "E": 48})

    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf
