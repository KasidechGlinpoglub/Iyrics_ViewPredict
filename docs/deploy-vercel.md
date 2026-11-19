# Deploy บน Vercel ครั้งแรก

Vercel เปิด Flask จาก `app.py` ที่ราก repo ติดตั้งแพ็กเกจตาม `requirements.txt` และใช้ Python 3.14 ตาม `.python-version` รุ่นที่เปิดบนเว็บใช้โมเดลสามคลาสที่ฝึกจากเพลงจริง 711 เพลงใน `deploy/lyrics_nb.joblib` ไฟล์มีตัวอย่างที่แต่งสำหรับเดโม และมีเวกเตอร์กับข้อมูลประกอบเพื่อแสดงเพลงคล้าย ส่วนเนื้อร้องต้นฉบับและชุดเพลงดิบเก็บไว้ในเครื่อง

## 1. ส่ง commit ที่ตรวจแล้วขึ้น GitHub

เปิด PowerShell ใน worktree ที่ checkout `main` แล้วตรวจ branch และสถานะ

```powershell
git branch --show-current
git status -sb
git fetch origin
git rev-list --left-right --count origin/main...main
```

เลขแรกนับ commit บน GitHub ที่เครื่องยังไม่มี ถ้าเลขแรกมีค่า ให้รวมการเปลี่ยนแปลงของทีมและตรวจอีกครั้ง เมื่อ branch เป็น `main` และเลขแรกเป็น 0 ให้ push เมื่อพร้อมเผยแพร่

```powershell
git push origin main
```

ตรวจบน GitHub ว่ามี `app.py`, `.python-version`, `vercel.json` และ `deploy/lyrics_nb.joblib` ไฟล์ `.env`, `data/songs.csv`, `data/.local/` และ `models/` เก็บในเครื่อง

## 2. สร้างโปรเจกต์ Vercel

1. เข้า [Vercel](https://vercel.com/) แล้วเข้าสู่ระบบด้วย GitHub
2. ใน Dashboard กด **Add New → Project** แล้วกด **Import** ที่ repo `Iyrics_ViewPredict` ถ้า repo ไม่ขึ้น ให้เปิดสิทธิ์ของ Vercel GitHub App สำหรับ repo นี้ หรือขอให้เจ้าของ repo อนุญาต
3. ใช้ **Framework Preset = Flask** และ **Root Directory = ราก repo** ที่มี `app.py` ใช้ค่าอัตโนมัติสำหรับ Build Command, Install Command และ Output Directory
4. ถ้าต้องการค้น YouTube เพิ่ม **Environment Variable** ชื่อ `YOUTUBE_API_KEY` ใส่คีย์ในช่อง Value และเลือก Production ด้วย ถ้าจะทดสอบ Preview ให้เปิดตัวแปรให้ Preview ด้วย การทำนายจากเนื้อร้องรันได้เมื่อยังไม่ได้ตั้งค่าคีย์
5. กด **Deploy** แล้วเปิดลิงก์ `.vercel.app` จากหน้าผลลัพธ์ หากเพิ่งเพิ่ม Environment Variable หลัง deploy ให้เลือก deployment ล่าสุดแล้วกด **Redeploy**

Vercel จะใช้ branch ที่ตั้งเป็น Production Branch ตรวจได้ใน Project Settings ภายหลัง เมื่อผูกกับ `main` การ push ถัดไปจะสร้าง deployment จาก commit ใหม่ตามการตั้งค่าของโปรเจกต์ [คู่มือ Flask](https://vercel.com/docs/frameworks/backend/flask) · [การเชื่อม Git](https://vercel.com/docs/git) · [Environment Variables](https://vercel.com/docs/environment-variables)

## 3. ตรวจเว็บหลัง deploy

- เปิด `/api/info` จากโดเมนที่ได้ ข้อมูลของรุ่นนี้ควรมี `report.n_train = 711`, `report.n_test = 264`, `report.test_correct = 95` และ `is_demo = false`
- เปิดหน้าแรก กด **ลองตัวอย่าง** ควรเห็นเนื้อร้องตัวอย่าง “ฝนที่หน้าต่าง” และผลสามช่วง ลองเปิด Heatmap และดาวน์โหลด Excel กับ PNG
- หากตั้งคีย์ ให้เช็กว่า `youtube.enabled` เป็น `true` แล้วลองกดแท็บ **YouTube** เพื่อดูผลจริง สถานะนี้ยืนยันว่าเว็บเห็นคีย์ ส่วนผลค้นหายืนยันว่า API รับคีย์
- ถ้า build ล้ม ให้เปิด **Deployments → deployment ล่าสุด → Build Logs** ถ้าเปิดเว็บแล้วขึ้น 500 ให้ดู **Runtime Logs** และลอง `/api/info` เพื่อแยกปัญหาโหลดโมเดลออกจากปัญหาหน้าเว็บ

รุ่นนี้เป็นเดโมสำรวจเนื้อร้อง Accuracy บนชุดทดสอบ 264 เพลงคือ 36.0% เทียบ baseline 42.0% [รายงานผล](expanded-dataset.md)

## เตรียมไฟล์ใหม่เมื่อเปลี่ยนโมเดล

ใช้เฉพาะไฟล์ joblib ที่ทีมสร้างและรู้แหล่งที่มา

```powershell
python -m scripts.package_web_model models/lyrics_nb.joblib data/.local/new-web-model.joblib
```

ตรวจผลกับชุดทดสอบและอัปเดตรายงานประเมินก่อนเปลี่ยนไฟล์ใน `deploy/` จากนั้นรัน `python -m unittest discover -s tests -v` ตัวทดสอบค่านับของรุ่นเผยแพร่ต้องตรงกับรายงานของโมเดลที่เลือก
