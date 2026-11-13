# ใช้ได้กับ Render, Hugging Face Spaces (Docker), Railway, Fly.io ฯลฯ
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=7860 \
    TRUST_PROXY=1

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
# WORKDIR ถูกสร้างโดย root: ต้องให้ user เขียนได้ ไม่งั้น train.py สร้างโฟลเดอร์ models/ ไม่ได้
RUN chown user:user /app
USER user

# เทรนโมเดลตอน build: ใช้ data/songs.csv ถ้ามี ไม่งั้นใช้ข้อมูลจำลอง
RUN python scripts/make_demo_data.py && python train.py

EXPOSE 7860
CMD ["sh", "-c", "exec gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120"]
