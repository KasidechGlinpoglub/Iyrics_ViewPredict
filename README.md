# 🎵 Lyrics View Predictor — ทำนายยอดวิว YouTube จากเนื้อเพลง

เว็บแอปที่วางเนื้อเพลงแล้วทำนายยอดวิว YouTube ด้วย **NLP + Multinomial Naive Bayes**
พร้อมค้นเพลงที่เนื้อหาใกล้เคียงบน YouTube แบบ real-time และตาราง Heatmap ของเนื้อเพลง

**🌐 เว็บใช้งานจริง:** https://lyrics-view-predictor.onrender.com
(แผนฟรีของ Render จะหลับเมื่อไม่มีคนใช้ เปิดครั้งแรกอาจรอประมาณ 1 นาที)

| หัวข้อ | |
|---|---|
| [1. Idea](#1-idea) | ปัญหา กลุ่มเป้าหมาย แนวคิด และทำไมต้องใช้ NLP |
| [2. How to use / run / customize](#2-how-to-use) | วิธีใช้งาน วิธีรัน และวิธีปรับแต่ง |
| [3. Data + Results](#3-data--results) | ข้อมูลที่ใช้ วิธีการ และผลการทดสอบ |
| [4. Team members + role](#4-team-members--role) | สมาชิกและหน้าที่ |

---

## 1. Idea

### Pain point (ปัญหา)
- นักแต่งเพลงและศิลปินหน้าใหม่ **ไม่รู้ล่วงหน้าว่าเพลงจะได้รับความนิยมแค่ไหน** ต้องลงทุนทำ MV และโปรโมตก่อนถึงจะรู้ผล
- การหาเพลงแนวเดียวกันมาเทียบต้องค้น YouTube เองทีละเพลง และการค้นด้วยคีย์เวิร์ดมักได้ผลไม่ตรงกับเนื้อหาเนื้อเพลง
- ไม่รู้ว่าคำหรือท่อนไหนสัมพันธ์กับยอดวิว การปรับเนื้อเพลงจึงต้องอาศัยความรู้สึก

### User (กลุ่มเป้าหมาย)
นักแต่งเพลง ศิลปินอิสระ/หน้าใหม่ ค่ายเพลงขนาดเล็กหรือทีม A&R ครีเอเตอร์สายคัฟเวอร์ และผู้สนใจ NLP ภาษาไทย

### Idea (ทำอะไร)
วางเนื้อเพลงแล้วได้ผล 4 อย่าง
1. **ยอดวิวที่คาด** และระดับ **หลักหมื่น / หลักแสน / หลักล้าน** พร้อมความน่าจะเป็น
2. **เพลงที่เนื้อหาใกล้เคียงบน YouTube** พร้อมลิงก์และยอดวิวจริง (ถ้าเพลงนั้นมีอยู่แล้ว จะแสดงยอดวิวจริงของเพลงนั้น)
3. **ตาราง Heatmap** ว่าคำหรือบรรทัดไหนสัมพันธ์กับยอดวิวระดับใด
4. **Export** ผลเป็น Excel และ PNG

### Solution — ดีกว่าเดิมยังไง และทำไมต้องใช้ NLP
- **ดีกว่าเดิม:** วางเนื้อเพลงครั้งเดียวได้ทั้งผลทำนาย เพลงที่คล้ายกัน ยอดวิวจริง และคำที่มีผล ในไม่กี่วินาที แทนการเดาหรือค้นเองทีละเพลง
- **ทำไปทำไม:** ลดความเสี่ยงก่อนลงทุนโปรโมต ช่วยเลือก/ปรับเนื้อเพลง และเห็นตลาดของเพลงแนวเดียวกัน
- **ทำไมต้อง NLP:** ข้อมูลหลักคือข้อความภาษาไทยที่ **ไม่เว้นวรรคระหว่างคำ** และมีภาษาถิ่น/ภาษาอังกฤษปน
  ต้องตัดคำ (PyThaiNLP), แปลงเป็นเวกเตอร์ (TF-IDF), จำแนกด้วย Naive Bayes, วัดความคล้ายด้วย cosine similarity
  และเลือกคำค้นจากความหายากของคำในภาษา ไม่ใช่คีย์เวิร์ดธรรมดา

---

## 2. How to use

### ใช้งานบนเว็บ
1. เปิด https://lyrics-view-predictor.onrender.com
2. วางเนื้อเพลงในช่อง **เนื้อเพลง** ทางซ้าย (หรือกด **สุ่มเนื้อเพลงตัวอย่าง**)
   ป้ายท่อน `[Chorus]`, เครื่องหมาย `(ซ้ำ *)`, `x2` และบรรทัดเครดิตที่ติดมาจากเว็บเนื้อเพลง ระบบตัดออกให้เอง
3. กด **ทำนายยอดวิว** (หรือ `Ctrl + Enter`)
4. อ่านผลทางขวา
   - **ตัวเลขใหญ่:** ยอดวิวที่คาด และระดับ หลักหมื่น/แสน/ล้าน ใต้ตัวเลขจะบอกที่มาของตัวเลข
     (ยอดวิวจริงของเพลงนี้ / ค่าเฉลี่ยของเพลงที่คล้ายกัน / โมเดลจากเนื้อเพลงอย่างเดียว)
   - **เพลงที่เนื้อหาใกล้เคียง:** แท็บ **YouTube (สด)** หรือ **ชุดข้อมูลเทรน** แก้คำค้นแล้วกด **ค้นใหม่** ได้
5. เลื่อนลงดู **ตาราง Heatmap** แบบ **รายคำ** หรือ **รายบรรทัด** (ชี้ที่ช่องเพื่อดูค่า)
6. กด **Export Excel** หรือ **Export PNG** เพื่อบันทึกผล

### How to run (รันในเครื่อง)
ใช้ Python 3.14 ตามเวอร์ชันใน `Dockerfile` คำสั่งนี้เทรนเดโมจากข้อมูลจำลองที่มีใน Git แล้วเปิดเว็บที่ `http://127.0.0.1:5000`

```bash
python -m pip install -r requirements.txt
python train.py --data data/demo_songs.csv
python app.py
```

ถ้ายังไม่ตั้งค่า YouTube API แอปยังทำนายจากโมเดลที่บันทึกไว้ได้ การค้นหาและอัปเดตยอดวิวสดต้องตั้งค่า API key

#### เชื่อมต่อ YouTube Data API
1. [Google Cloud Console](https://console.cloud.google.com/) → สร้างโปรเจกต์ → **APIs & Services → Library** → เปิด **YouTube Data API v3**
2. **Credentials → Create credentials → API key** (แนะนำ Restrict key ให้ใช้ได้เฉพาะ YouTube Data API v3)
3. คัดลอก `.env.example` เป็น `.env` แล้วใส่ `YOUTUBE_API_KEY=...` จากนั้นรัน `python app.py` ใหม่

ใช้ **YouTube Data API v3** ซึ่งเป็น API ทางการของ Google

> ห้ามอัปโหลด `.env` ขึ้น GitHub (อยู่ใน `.gitignore` แล้ว) key ถูกใช้ฝั่งเซิร์ฟเวอร์เท่านั้น

#### Deploy ขึ้นอินเทอร์เน็ต
มี `Dockerfile` (เทรนเดโมจากข้อมูลจำลองตอน build แล้วรันด้วย gunicorn) และ `render.yaml` พร้อมแล้ว
- **Render:** push ขึ้น GitHub → [render.com](https://render.com) → **New → Blueprint** → เลือก repo → ใส่ `YOUTUBE_API_KEY` ในหน้า **Environment** ของ Web Service
- **Hugging Face Spaces:** สร้าง Space แบบ Docker, เพิ่ม `sdk: docker` และ `app_port: 7860` ในหัว README ของ Space และตั้ง secret `YOUTUBE_API_KEY`

### How to customize (ปรับแต่ง)

Google กำหนดโควตาเริ่มต้นให้ `search.list` 100 ครั้ง/วันแยกจากโควตา 10,000 หน่วย/วันของปลายทางทั่วไป ตรวจค่าของโปรเจกต์จริงใน Google Cloud Console ได้ [เอกสารโควตาของ Google](https://developers.google.com/youtube/data/determine_quota_cost)

แอปจำกัดและเก็บผลการเรียก API ดังนี้
- จำกัดการค้นหาไว้ที่ `YT_DAILY_SEARCH_LIMIT` ครั้ง/วัน (ค่าเริ่มต้น 90)
- จำกัดผู้ใช้แต่ละ IP ไว้ที่ `YT_SEARCH_PER_IP` ครั้ง/10 นาที (ค่าเริ่มต้น 20)
- เก็บผลค้นหาไว้ 6 ชั่วโมง และเก็บยอดวิวไว้ 10 นาที (คำค้นเดิมจะไม่เสียโควตาซ้ำ)
- ดึงยอดวิวใช้แค่ 1 หน่วยต่อ 50 วิดีโอ

| อยากเปลี่ยน | ทำที่ไหน |
|---|---|
| **ใช้ข้อมูลเพลงจริงแทนข้อมูลจำลอง** | วางไฟล์ `data/songs.csv` (คอลัมน์ `title, artist, lyrics, views, url`) แล้วรัน `python train.py` — ระบบใช้ไฟล์นี้แทนข้อมูลจำลองอัตโนมัติ |
| **เก็บข้อมูลจริงจาก YouTube** | `python scripts/collect_youtube.py playlist "<ลิงก์ playlist>" -o data/raw_songs.csv` → กรอกเนื้อเพลงในคอลัมน์ `lyrics` → `python scripts/collect_youtube.py fill data/raw_songs.csv -o data/songs.csv` |
| **ระดับยอดวิว** | ค่าเริ่มต้น 3 ระดับคงที่ (`FIXED_EDGES`, `FIXED_NAMES` ใน `core/model.py`) หรือแบ่งตาม quantile: `python train.py --levels quantile --classes 4` |
| **จำนวนครั้งค้น YouTube** | `.env` / Environment: `YT_DAILY_SEARCH_LIMIT` (ต่อวัน, ค่าเริ่มต้น 90), `YT_SEARCH_PER_IP` (ต่อ IP ใน 10 นาที, ค่าเริ่มต้น 20) |
| **วิธีรวมผลโมเดลกับยอดวิวจริง** | `MODEL_WEIGHT`, `FULL_COVERAGE`, `EXACT_MIN_SCORE` ใน `core/model.py` |
| **การเลือกคำค้น/จัดอันดับเพลง YouTube** | ค่าคงที่ต้นไฟล์ `core/matching.py` เช่น `COMMON_FREQ`, `RARE_FLOOR`, `THEME_FREQ`, `HOOK_RANK_WEIGHT` |
| **การทำความสะอาดเนื้อเพลง / stopwords** | `clean_lyrics()` และ `EN_STOPWORDS`, `TH_FILLERS` ใน `core/text.py` |
| **สีและหน้าตา (ธีมม่วง × ทอง)** | ตัวแปรสีใน `:root` ของ `static/style.css`, สี heatmap `RAMP` ใน `static/app.js` และ `core/export.py` |

---

## 3. Data + Results

### Data


ชุดที่ตรวจแล้วมี **711 เพลง จาก 169 ศิลปินใน 65 ช่อง** ใช้ MV ทางการและยอดวิวสะสมที่บันทึกวันที่ **18 พ.ย. 2026** ตามเวลาไทย [เกณฑ์คัดเพลงและนิยามยอดวิว](docs/dataset-protocol.md) · [รายงานชุดข้อมูล](docs/training-dataset.md)

ชุดขยายมี **1,306 เพลง จาก 367 ศิลปินใน 151 ช่อง** ใช้ยอดสะสมที่อ่านวันที่ **19 พ.ย. 2026** ตามเวลาไทย แบ่งตามกลุ่มศิลปินเป็นฝึก **1,042 เพลง** และทดสอบ **264 เพลง** ใช้สามช่วงยอดวิวตามโหมด fixed ให้ส่ง `--split` เมื่อเทรนจากชุดแบ่งที่ล็อกไว้ [วิธีเก็บชุดใหม่และทำผลซ้ำ](docs/expanded-dataset.md) · [รายการแบ่งชุดขยาย](data/songs_expanded_split.csv)

`data/songs.csv` มีเนื้อร้องและยอดวิว จัดเก็บในเครื่องนอก Git ผู้ร่วมทีมที่ต้องใช้ชุด 711 เพลงต้องรับไฟล์นี้จากทีมแยกต่างหาก ไฟล์ประกอบใน Git มีรายการเพลง รายการแบ่งชุด บันทึกการตรวจ และข้อมูลจำลอง [รายการเพลง](data/songs_manifest.csv) · [บันทึกการตรวจ](data/songs_review_audit.csv)

#### เก็บชุดข้อมูลเพิ่มเติม

ตัวอย่างด้านล่างใช้ LRCLIB ช่วยเติมเนื้อร้องสำหรับงานสำรวจ ส่วนชุด 711 เพลงใช้เนื้อร้องจากคำอธิบาย MV ทางการและผ่านการตรวจเพิ่มเติม [ผลรอบทดลอง 30 MV](docs/pilot-30.md) · [วิธีเตรียมชุดปัจจุบัน](docs/training-dataset.md)

1. ดึงรายการอัปโหลดและยอดวิวจากช่อง YouTube ที่เลือก (ต้องมี API key)
   ```bash
   python scripts/collect_youtube.py channels "@genelab" "UC1T-ZwcYVQB6EpqOQD10tmw" "@yuppofficial" "@TangBadVoice" "@safeplanetband" "@timethaiofficial" -o data/raw_songs.csv
   ```
   ระบุช่องด้วย `@handle`, channel ID หรือลิงก์ของช่องได้ คำสั่งอ่านรายการอัปโหลดทุกหน้าและรับคลิปทุกอายุ ใช้ `--limit 50` เมื่อต้องการดูเฉพาะ 50 คลิปแรกต่อช่อง
   รันซ้ำจะเพิ่มเฉพาะ `video_id` ใหม่และรักษาเนื้อร้องกับยอดวิวเดิมของแถวที่มีอยู่ คอลัมน์ `published_at` และ `views_collected_at` จาก API ระบุเวลาด้วย UTC ให้เทียบวันเก็บข้อมูลตามเวลาไทย

   หากมีรายการคลิปที่เลือกไว้ใน playlist แล้ว ให้ใช้คำสั่งเดิม
   ```bash
   python scripts/collect_youtube.py playlist "https://www.youtube.com/playlist?list=..." -o data/raw_songs.csv
   ```
2. คัด MV ทางการและยืนยัน `title` กับ `artist` ก่อนค้นเนื้อร้อง ชื่อคลิปที่ยังมีคำว่า `Official MV` หรือชื่อศิลปินติดมาควรแก้ให้เหลือชื่อเพลง ส่วน `channel` คือชื่อช่องที่เผยแพร่

   ```bash
   python scripts/collect_youtube.py lyrics data/raw_songs.csv
   ```

   คำสั่งค้น [LRCLIB](https://lrclib.net/docs) โดยใช้ชื่อเพลงคู่กับศิลปิน แล้วบันทึก `lyrics`, `lyrics_source` และ `lyrics_status` กลับไปยังไฟล์เดิม ใช้ `-o data/songs_with_lyrics.csv` เพื่อเขียนผลไปยังไฟล์ใหม่ คำสั่งจะเก็บเนื้อร้องที่มีอยู่แล้วไว้

   | `lyrics_status` | ความหมาย |
   |---|---|
   | `matched` | ชื่อเพลงและศิลปินตรงกับรายการเดียวที่มีเนื้อร้อง เก็บ URL ของรายการใน `lyrics_source` |
   | `needs_review` | ขาดชื่อเพลงหรือศิลปิน มีหลายรายการที่ชื่อและศิลปินตรง หรือผลที่ได้คล้ายแต่ข้อมูลยังไม่ตรงทั้งคู่ |
   | `not_found` | ไม่พบผลค้นหา หรือรายการที่ตรงไม่มีเนื้อร้อง |
   | `existing` | มีเนื้อร้องใน CSV อยู่แล้ว |
   | `lookup_error` | ติดต่อ LRCLIB ไม่สำเร็จ สคริปต์บันทึกผลที่ทำได้ก่อนหยุดและสามารถรันใหม่ได้ |

   ตรวจเนื้อร้องกับเวอร์ชัน MV ก่อนใช้เทรน สำหรับเพลงที่ต้องตรวจหรือหาไม่เจอ ให้แก้ชื่อแล้วลองค้นใหม่ หรือกรอกเนื้อร้องและแหล่งที่มาที่ทีมตรวจแล้ว
3. อัปเดตยอดวิวล่าสุดและสร้างไฟล์สำหรับเทรน (แถวที่ยังไม่มีเนื้อเพลงจะถูกข้าม):
   ```bash
   python scripts/collect_youtube.py fill data/raw_songs.csv -o data/songs_with_lyrics.csv
   ```
   ใช้ไฟล์ที่ได้จากขั้นเติมเนื้อร้องเป็นอินพุต คำสั่ง `fill` อัปเดตยอดวิวพร้อมเวลาที่ดึงใหม่ และแจ้งคลิปที่ดึงยอดไม่ได้ก่อนข้ามแถวนั้น ถ้าไม่มี API key จะใช้ yt-dlp แทน (`pip install yt-dlp`)
4. หากต้องการลองเทรนชุดที่เก็บเพิ่มด้วยวิธีเดิม ให้ระบุไฟล์ด้วย `python train.py --data data/songs_with_lyrics.csv`


| ข้อมูล | รายละเอียด |
|---|---|
| **ชุดข้อมูลเทรน (ตอนนี้)** | **ข้อมูลจำลอง 260 เพลง** (`data/demo_songs.csv`) สร้างด้วย `scripts/make_demo_data.py` — เนื้อเพลง ชื่อเพลง และยอดวิวแต่งขึ้นเองใน 7 แนว (อกหัก รักหวาน ลูกทุ่ง ปาร์ตี้ กำลังใจ ชีวิตเมือง สากล) เพราะเนื้อเพลงจริงมีลิขสิทธิ์ แบ่งเป็น หลักหมื่น 86 / หลักแสน 116 / หลักล้าน 58 เพลง |
| **ข้อมูลจริง (รองรับแล้ว)** | ยอดวิวจาก YouTube Data API + เนื้อเพลงที่ผู้ใช้กรอก ผ่าน `scripts/collect_youtube.py` |
| **ข้อมูลสด** | YouTube Data API v3: ผลค้นหา ยอดวิว คำอธิบายคลิป และหัวข้อที่ YouTube วิเคราะห์ (ใช้กรองคลิปที่ไม่ใช่เพลง) |
| **ความถี่คำ** | ภาษาไทย: Thai National Corpus (มากับ PyThaiNLP), ภาษาอังกฤษ: `core/data/en_word_freq.tsv` จาก wordfreq, คำทั่วไปในเพลง: `core/data/lyric_common.txt` (คำที่พบใน ≥ 6% ของ 788 เพลง เช่น บ่ อ้าย ฮัก หัวใจ) |

### Method
| ขั้นตอน | รายละเอียด |
|---|---|
| 1. Preprocessing | ตัดป้ายท่อน/เครดิต/หัวเพลง → normalize → ตัดคำด้วย **PyThaiNLP (newmm)** → ตัด stopwords ไทย/อังกฤษ |
| 2. Label | 3 ระดับ: **หลักหมื่น** (< 1 แสน), **หลักแสน** (1 แสน – 1 ล้าน), **หลักล้าน** (≥ 1 ล้าน) |
| 3. Features | **TF-IDF** ของคำ |
| 4. Model | **Multinomial Naive Bayes** จูน `alpha`, `use_idf`, `min_df` ด้วย GridSearchCV 5-fold (macro-F1) ชุดจริงใช้กลุ่มศิลปินและไฟล์แบ่งที่ล็อก ชุดเดโมแบ่งแบบ stratified ระดับเพลง |
| 5. ยอดวิวที่คาด | โมเดล: `10 ^ Σ P(ระดับ) × median(log10 views ของระดับนั้น)` → รวมกับ YouTube: เจอเพลงเดียวกัน (เนื้อเพลงตรง ≥ 50%) ใช้ยอดวิวจริง ไม่งั้นใช้ค่าเฉลี่ยเรขาคณิตของยอดวิวเพลงที่คล้ายกัน ถ่วงตาม % ความคล้าย |
| 6. เพลงใกล้เคียง (ชุดข้อมูล) | cosine similarity ของ TF-IDF |
| 7. เพลงใกล้เคียง (YouTube) | เลือกคำค้นที่ซ้ำในเนื้อเพลงและหายากในภาษาทั่วไป → ค้น 25 คลิป → ให้คะแนนจากท่อนเพลงในคำอธิบาย / ชื่อเพลง / คำสำคัญ → รวมเวอร์ชัน (MV, Lyric, Cover) → ถ้ายังไม่เจอ ค้นรอบ 2 ด้วยท่อนฮุก |
| 8. Heatmap | รายคำ: `P(ระดับ \| คำ)` จาก `feature_log_prob_` ของ Naive Bayes, รายบรรทัด: `predict_proba` ทีละบรรทัด |

### Results

**เพลงจริง 264 เพลงจาก 88 กลุ่มศิลปินที่กันไว้** โมเดลที่ฝึกจาก 711 เพลงและโมเดลที่ฝึกจาก 1,042 เพลงทายถูกเท่ากัน **95 เพลง (36.0%)** เทียบกับ baseline **111 เพลง (42.0%)** ทั้งสองโมเดลทายเพลงต่ำกว่าแสนวิวถูก 0 จาก 55 เพลง [ผลรายคลาสและข้อจำกัด](docs/expanded-dataset.md)

**โมเดล Naive Bayes** (ข้อมูลจำลอง, test set 52 เพลง — ดูได้ที่ `models/report.json` หลังรัน `train.py`)

| Metric | ค่า |
|---|---|
| Accuracy | **55.8%** (baseline ทายระดับที่พบมากสุด 44.2%) |
| Macro-F1 (test / CV 5-fold) | 0.554 / 0.584 |
| MAE ของ log10(ยอดวิว) | 0.439 (คลาดเคลื่อนประมาณ ×2.7 เท่า) |
| พารามิเตอร์ที่ดีที่สุด | alpha = 0.3, use_idf = False, min_df = 1 |

Confusion matrix (แถว = จริง, คอลัมน์ = ทำนาย) — ไม่มีการทายข้ามระดับไกล (หลักหมื่น ↔ หลักล้าน)

| | หลักหมื่น | หลักแสน | หลักล้าน |
|---|---|---|---|
| **หลักหมื่น** | 8 | 9 | 0 |
| **หลักแสน** | 4 | 15 | 4 |
| **หลักล้าน** | 0 | 6 | 6 |

**การจับคู่เพลงกับ YouTube** — ใช้เนื้อเพลงจากคำอธิบายคลิปเนื้อเพลงเป็นอินพุต แล้ว **ตัดคลิปต้นทางออกจากผลค้นหา**
นับว่าถูกเมื่อชื่อที่โชว์เป็นอันดับ 1 เป็นเพลงเดียวกัน (เวอร์ชันใดก็ได้)

| ชุดทดสอบ | เพลง | YouTube ส่งเพลงนั้นมา | อันดับ 1 ถูก (เมื่อมีในผล) |
|---|---|---|---|
| dev (ใช้ปรับระบบ: ป๊อป ลูกทุ่ง หมอลำ ร็อค สากล) | 30 | 27 (เวอร์ชันแรก 21) | 27/27 |
| **hold-out** (ไม่เคยใช้ปรับระบบ: อกหัก แร็พ เพลงเพราะ เพื่อชีวิต สากล 2024) | 30 | 26 | **23/26** |

**ตัวอย่าง:** วางเนื้อเพลง "ไม่รักดีกว่า – Z9" โมเดลจากเนื้อเพลงอย่างเดียวคาด 2.6 แสน แต่ระบบเจอเพลงนี้บน YouTube
(เนื้อเพลงตรงกัน) จึงแสดงยอดวิวจริง 35.6 ล้าน ระดับหลักล้าน

### Limitations
- การวัดกับชุดจริงรอบนี้ใช้ยอดสะสมของ MV หลายอายุ และโมเดลทายถูกน้อยกว่า baseline
- ยอดวิวขึ้นกับปัจจัยอื่นด้วย เช่น ชื่อเสียงศิลปิน ทำนอง MV การโปรโมต และกระแส เนื้อเพลงอธิบายได้เพียงบางส่วน
- Naive Bayes มองคำแยกกัน (bag-of-words) ไม่เข้าใจลำดับคำหรือความหมายเชิงลึก
- โควตา YouTube API 10,000 หน่วย/วัน (ค้นได้ประมาณ 50–100 ครั้ง)
- การจับคู่ YouTube อาจพลาดเมื่อมีบรรทัดชื่อศิลปินปนในเนื้อเพลง หรือ MV ไม่มีเนื้อเพลงในคำอธิบาย และรองรับเฉพาะภาษาไทย/อังกฤษ

---

## 4. Team members + role

| สมาชิก | บทบาท |
|---|---|
| **Kasidech Glinpoglub** | ทำคนเดียวทุกส่วน: คิดไอเดียและออกแบบระบบ, เตรียมข้อมูล, พัฒนาโมเดล NLP / Naive Bayes, ระบบจับคู่เพลงกับ YouTube, พัฒนาเว็บ (Flask + UI), ทดสอบและวัดผล, deploy |

เครื่องมือช่วยพัฒนา: Claude Code (AI coding assistant)

---

## โครงสร้างไฟล์

```
app.py                      Flask server + API (/api/predict, /api/export/xlsx, /api/youtube/search, /api/youtube/stats, /api/info)
train.py                    เทรนและประเมินโมเดล
core/text.py                ทำความสะอาดเนื้อเพลง / ตัดคำ
core/model.py               Naive Bayes, ยอดวิวที่คาด, รวมผลกับยอดวิวจริง, เพลงใกล้เคียง, heatmap
core/matching.py            เลือกคำค้น / จัดอันดับ / รวมเวอร์ชันเพลงบน YouTube
core/youtube.py             YouTube Data API v3 client (cache, จำกัดโควตา)
core/export.py              ไฟล์ Excel (heatmap 2 ชีต + สรุปผล)
core/config.py              อ่านค่าจาก .env
core/data/                  ความถี่คำภาษาอังกฤษ, รายการคำทั่วไปในเพลง
templates/, static/         หน้าเว็บ (HTML / CSS / JS)
scripts/make_demo_data.py   สร้างข้อมูลจำลอง
scripts/collect_youtube.py  เก็บข้อมูลจากช่องหรือ playlist และค้นเนื้อร้องผ่าน LRCLIB
scripts/lock_training_split.py  สร้างรายการแบ่งชุดจริงตามกลุ่มศิลปิน
data/songs_split.csv        รายการ video_id, split_group และฝั่ง train/test ที่ล็อกแล้ว
Dockerfile, render.yaml     สำหรับ deploy
```

## References และเครดิต
1. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
2. McCallum, A., & Nigam, K. (1998). A comparison of event models for naive Bayes text classification. *AAAI-98 Workshop on Learning for Text Categorization*.
3. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513–523.
4. Dhanaraj, R., & Logan, B. (2005). Automatic prediction of hit songs. *Proceedings of ISMIR 2005*.
5. Singhi, A., & Brown, D. G. (2014). Hit song detection using lyric features alone. *Proceedings of ISMIR 2014*.
6. [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) — ตัดคำภาษาไทย, Thai National Corpus
7. [wordfreq](https://github.com/rspeer/wordfreq) (Robyn Speer) — `core/data/en_word_freq.tsv` สัญญาอนุญาตข้อมูล **CC BY-SA 4.0**
8. [YouTube Data API v3](https://developers.google.com/youtube/v3)
