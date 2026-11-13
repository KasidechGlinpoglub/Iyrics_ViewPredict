"""อ่านค่าตั้งค่าจาก environment variables และไฟล์ .env (ถ้ามี)

ไฟล์ .env อยู่ที่โฟลเดอร์โปรเจกต์ รูปแบบ KEY=VALUE ทีละบรรทัด
ค่าใน environment จริง (เช่นที่ตั้งบน Render / Hugging Face) มีลำดับความสำคัญสูงกว่า .env
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


def load_env(path: Path = BASE_DIR / ".env") -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


load_env()
