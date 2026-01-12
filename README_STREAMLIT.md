# 🤖 Askgiraffe - Chatbot UI with Streamlit

ระบบแชทบอทสำหรับให้คำปรึกษาหลักสูตรคณะครุศาสตร์อุตสาหกรรม มจพ. พร้อม UI ที่สวยงามและใช้งานง่าย

## ✨ ฟีเจอร์หลัก

- 💬 **หน้าแชทแบบ Real-time** - ถาม-ตอบได้อย่างต่อเนื่อง
- 🔍 **Hybrid Search** - รวม BM25 (Keyword) + Vector Search (Semantic)
- 🎯 **Cross-Encoder Reranking** - เพิ่มความแม่นยำของผลลัพธ์
- 📚 **แสดงเอกสารอ้างอิง** - ดูที่มาของคำตอบได้
- ⚙️ **ตั้งค่าการค้นหา** - ปรับน้ำหนัก BM25/Vector และจำนวนผลลัพธ์
- 📊 **สถิติการใช้งาน** - ดูจำนวนเอกสารและประวัติการสนทนา
- 🎨 **UI สวยงาม** - ออกแบบด้วย Custom CSS

## 🚀 การติดตั้ง

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

หรือติดตั้งเฉพาะ Streamlit และ OpenAI:

```bash
pip install streamlit openai
```

### 2. ตั้งค่า Environment Variables

สร้างไฟล์ `.env` และใส่ API Key:

```bash
TYPHOON_API_KEY=your_typhoon_api_key_here
```

### 3. ตรวจสอบฐานข้อมูล

ตรวจสอบว่ามีโฟลเดอร์ `chroma_db` และมีข้อมูลอยู่:

```bash
ls -la chroma_db/
```

## 🎮 การใช้งาน

### รันแอปพลิเคชัน Streamlit

```bash
streamlit run app.py
```

หรือกำหนด port เอง:

```bash
streamlit run app.py --server.port 8501
```

### เข้าใช้งานผ่าน Browser

เปิด browser และไปที่:
```
http://localhost:8501
```

## 📖 คู่มือการใช้งาน

### 1. พื้นที่แชท (หน้าหลัก)
- พิมพ์คำถามในช่อง chat input ด้านล่าง
- กด Enter หรือคลิกปุ่มส่ง
- รอAskgiraffeตอบคำถาม (ใช้เวลาประมาณ 5-10 วินาที)

### 2. Sidebar (แถบด้านซ้าย)

#### 📊 ข้อมูลระบบ
- แสดงจำนวนเอกสารในฐานข้อมูล
- แสดงจำนวนข้อความในประวัติการสนทนา

#### 🔍 ตั้งค่าการค้นหา
- **จำนวนเอกสารที่ค้นหา** (3-15): ยิ่งมากยิ่งครอบคลุม แต่ช้ากว่า
- **น้ำหนัก BM25** (0.0-1.0):
  - เพิ่ม = เน้นการค้นหาแบบ Keyword
  - ลด = เน้นการค้นหาแบบ Semantic

#### 👁️ การแสดงผล
- **แสดงเอกสารอ้างอิง**: เปิด/ปิดการแสดงข้อมูลที่ใช้ตอบคำถาม

#### 🗑️ ล้างประวัติการสนทนา
- ลบข้อความทั้งหมดและเริ่มใหม่

### 3. การดูเอกสารอ้างอิง
เมื่อเปิด "แสดงเอกสารอ้างอิง":
- คลิกที่ expander ด้านล่างคำตอบของบอท
- จะแสดง:
  - **Score**: คะแนนความเกี่ยวข้องจาก Hybrid Search
  - **Rerank Score**: คะแนนจาก Cross-Encoder
  - **เนื้อหาเอกสาร**: ข้อมูลที่ใช้ตอบคำถาม

## ⚙️ การตั้งค่าขั้นสูง

### แก้ไขค่าเริ่มต้นในโค้ด

เปิดไฟล์ `app.py` และแก้ไขในส่วน:

```python
# ค่าเริ่มต้นการค้นหา
n_results = st.slider(
    "จำนวนเอกสารที่ค้นหา",
    min_value=3,
    max_value=15,
    value=10,  # <-- แก้ตรงนี้
    ...
)

bm25_weight = st.slider(
    "น้ำหนัก BM25",
    min_value=0.0,
    max_value=1.0,
    value=0.4,  # <-- แก้ตรงนี้
    ...
)
```

### เปิด/ปิด Reranking

แก้ไขใน function `init_chatbot()`:

```python
kb = HybridKnowledgeBase(
    persist_directory="./chroma_db",
    collection_name="chatbot_knowledge",
    use_reranker=True  # <-- เปลี่ยนเป็น False เพื่อปิด
)
```

### เปลี่ยน AI Model

แก้ไขในไฟล์ `chatbot_v03.py` ที่ function `chat()`:

```python
response = self.client.chat.completions.create(
    model="typhoon-v2.1-12b-instruct",  # <-- เปลี่ยน model ตรงนี้
    ...
)
```

**Models ที่ใช้ได้:**
- `typhoon-v2.1-12b-instruct` (เร็ว, แนะนำ)
- `typhoon-v2.5-30b-a3b-instruct` (แม่นยำกว่า แต่ช้ากว่า)

## 🎨 การปรับแต่ง UI

### เปลี่ยนสีธีม

แก้ไข CSS ในไฟล์ `app.py` ส่วน `st.markdown("""<style>...`)`:

```css
.user-message {
    background-color: #DBEAFE;  /* สีพื้นหลังข้อความผู้ใช้ */
    border-left: 4px solid #3B82F6;  /* สีขอบซ้าย */
}

.bot-message {
    background-color: #F0FDF4;  /* สีพื้นหลังข้อความบอท */
    border-left: 4px solid #10B981;  /* สีขอบซ้าย */
}
```

### เปลี่ยน Icon และชื่อ

แก้ไขใน `st.set_page_config()`:

```python
st.set_page_config(
    page_title="ชื่อแอปของคุณ",
    page_icon="🎓",  # <-- เปลี่ยน icon
    ...
)
```

## 🐛 การแก้ไขปัญหา

### ปัญหา: ไม่พบ TYPHOON_API_KEY
**วิธีแก้:**
1. ตรวจสอบว่ามีไฟล์ `.env` อยู่ในโฟลเดอร์เดียวกัน
2. เปิดไฟล์ `.env` และตรวจสอบว่ามี `TYPHOON_API_KEY=...`
3. ลองรันใหม่

### ปัญหา: ไม่พบ ChromaDB collection
**วิธีแก้:**
1. ตรวจสอบว่ามีโฟลเดอร์ `chroma_db` และมีไฟล์ข้างใน
2. ถ้าไม่มี ให้รัน script สร้างฐานข้อมูลก่อน (เช่น `init_vector_db.py`)

### ปัญหา: Streamlit ช้า
**วิธีแก้:**
1. ลดจำนวนเอกสารที่ค้นหาลงเหลือ 5-7
2. ปิด Reranker (`use_reranker=False`)
3. ใช้ model ที่เล็กกว่า

### ปัญหา: ข้อความไม่แสดงหรือ layout พัง
**วิธีแก้:**
1. Clear cache: กด `C` ใน terminal ที่รัน Streamlit
2. Refresh browser (Ctrl+F5 หรือ Cmd+Shift+R)
3. ตรวจสอบ Console ใน browser (F12) ดู error

## 📝 ตัวอย่างคำถาม

- "หลักสูตรวิศวกรรมโยธาและการศึกษาเรียนกี่ปี"
- "มีวิชาอะไรบ้างในปี 1"
- "คุณสมบัติการสมัครเรียนมีอะไรบ้าง"
- "จบแล้วทำงานอะไรได้บ้าง"
- "หลักสูตรคอมพิวเตอร์ศึกษามีอะไรบ้าง"

## 🔧 เทคโนโลยีที่ใช้

- **Frontend**: Streamlit
- **AI Model**: Typhoon (OpenAI-compatible API)
- **Vector Database**: ChromaDB
- **Embedding Model**: paraphrase-multilingual-MiniLM-L12-v2
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Keyword Search**: BM25 (rank-bm25)
- **Thai NLP**: PyThaiNLP

## 📞 ติดต่อสอบถาม

- **เว็บไซต์**: admission.kmutnb.ac.th
- **โทร**: 02-555-2000
- **Facebook**: คณะครุศาสตร์อุตสาหกรรม มจพ.

## 📄 License

© 2025 Faculty of Industrial Education, KMUTNB. All rights reserved.
