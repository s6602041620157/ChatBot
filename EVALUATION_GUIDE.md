# คู่มือการใช้งาน RAG Evaluation Script

## ภาพรวม

สคริปต์นี้ใช้สำหรับประเมินประสิทธิภาพของระบบ RAG (Retrieval-Augmented Generation) โดยใช้ Ragas framework ซึ่งจะวัดค่าต่างๆ เช่น:

- **Context Precision**: ความแม่นยำของ context ที่ดึงมา
- **Context Recall**: ความครอบคลุมของ context
- **Faithfulness**: ความน่าเชื่อถือของคำตอบที่อิงจาก context
- **Answer Relevancy**: ความเกี่ยวข้องของคำตอบกับคำถาม

> ใหม่: ถ้าต้องการประเมินแบบไม่พึ่ง LLM-as-judge ให้ใช้สคริปต์ `evaluate_statistical.py` (ดูหัวข้อ "โหมดสถิติ (ไม่ใช้ LLM)").

## การติดตั้ง

### 1. ติดตั้ง Dependencies

```bash
pip install datasets ragas
```

### 2. ตั้งค่า API Key

สร้างไฟล์ `.env` ในโฟลเดอร์โปรเจกต์:

```bash
GEMINI_API_KEY=your_api_key_here
```

หรือตั้งค่าผ่าน environment variable:

```bash
export GEMINI_API_KEY=your_api_key_here
```

### 3. ทดสอบการตั้งค่า (แนะนำ)

ก่อนรัน evaluation ให้ทดสอบการตั้งค่าก่อน:

```bash
python3 test_evaluation_setup.py
```

สคริปต์นี้จะตรวจสอบ:
- ✅ Environment variables (API keys)
- ✅ Required packages (datasets, ragas)
- ✅ Knowledge base files
- ✅ Component initialization

ถ้าทุกอย่างผ่าน จะแสดงข้อความ "All checks passed!"

## วิธีใช้งาน

### รันสคริปต์

```bash
python3 evaluate_rag.py
```

### โหมดการใช้งาน

เมื่อรันสคริปต์ จะมี 4 โหมดให้เลือก:

#### 1. Quick Evaluation (แนะนำสำหรับการทดสอบเบื้องต้น)
- ใช้การตั้งค่าแบบ Full Stack (Re-ranker + Compression)
- รวดเร็วและให้ผลลัพธ์ทันที
- เหมาะสำหรับการทดสอบเบื้องต้น

#### 2. Compare All Configurations (แนะนำสำหรับการวิเคราะห์เชิงลึก)
- เปรียบเทียบ 3 การตั้งค่า:
  - Baseline (ไม่มี Re-ranker, ไม่มี Compression)
  - Re-ranker Only (มีแค่ Re-ranker)
  - Full Stack (Re-ranker + Compression)
- ใช้เวลานานกว่า แต่ได้ข้อมูลครบถ้วน
- ผลลัพธ์จะบันทึกเป็นไฟล์ JSON ในโฟลเดอร์ `evaluation_results/`

#### 3. Custom Configuration (สำหรับการทดสอบเฉพาะ)
- กำหนดการตั้งค่าเอง
- เลือกว่าจะใช้ Re-ranker หรือไม่
- เลือกว่าจะใช้ Context Compression หรือไม่

#### 4. Exit
- ออกจากโปรแกรม

### โหมดสถิติ (ไม่ใช้ LLM)

หากต้องการประเมินด้วยเมตริกเชิงสถิติโดยไม่เรียก LLM สำหรับการให้คะแนน ใช้:

```bash
python3 evaluate_statistical.py
```

สคริปต์นี้จะคำนวณ:
- answer_exact_match
- answer_f1 (token-level)
- answer_similarity (difflib ratio)
- context_recall
- context_hit_rate

ผลลัพธ์จะสรุปบนหน้าจอและบันทึกไฟล์ JSON ใน `evaluation_results/` (ชื่อไฟล์ขึ้นต้น `deterministic_eval_*.json`)

## การแปลผลลัพธ์

### คะแนนแต่ละ Metric (0.0 - 1.0)

- **0.8 - 1.0**: ดีเยี่ยม - ระบบทำงานได้ดีมาก
- **0.6 - 0.8**: ดี - ระบบทำงานได้ดี แต่อาจปรับปรุงได้
- **0.4 - 0.6**: พอใช้ - ต้องปรับปรุง
- **0.0 - 0.4**: ต้องแก้ไข - มีปัญหาร้ายแรง

### ตัวอย่างผลลัพธ์

```
📊 EVALUATION RESULTS
================================================================================
Context Precision: 0.8542  ← ดีมาก
Context Recall: 0.7321     ← ดี
Faithfulness: 0.9123       ← ดีเยี่ยม
Answer Relevancy: 0.8654   ← ดีเยี่ยม
================================================================================
```

## ผลลัพธ์ที่บันทึก

ผลลัพธ์จะถูกบันทึกในโฟลเดอร์ `evaluation_results/` เป็นไฟล์ JSON:

```
evaluation_results/
├── baseline_20260108_142530.json
├── reranker_only_20260108_142630.json
└── full_stack_20260108_142730.json
```

ตัวอย่างเนื้อหาไฟล์:

```json
{
  "config": "full_stack",
  "timestamp": "20260108_142730",
  "metrics": {
    "context_precision": 0.8542,
    "context_recall": 0.7321,
    "faithfulness": 0.9123,
    "answer_relevancy": 0.8654
  }
}
```

## การปรับแต่ง Test Cases

แก้ไขฟังก์ชัน `create_test_dataset()` ในไฟล์ `evaluate_rag.py`:

```python
def create_test_dataset() -> List[Dict]:
    test_cases = [
        {
            "question": "คำถามของคุณ",
            "ground_truth": "คำตอบที่ถูกต้อง"
        },
        # เพิ่มอีก...
    ]
    return test_cases
```

## การแก้ไขปัญหา

### ปัญหา: ModuleNotFoundError

```bash
pip install datasets ragas
```

### ปัญหา: API Key ไม่พบ

ตรวจสอบว่ามีไฟล์ `.env` และมี `GEMINI_API_KEY` อยู่:

```bash
cat .env
```

### ปัญหา: Import Error

ตรวจสอบว่าไฟล์ `chatbot_v04_keywords.py` อยู่ในโฟลเดอร์เดียวกัน

### ปัญหา: Ragas Evaluation Failed

- ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต
- ตรวจสอบว่า API key ใช้งานได้
- ตรวจสอบว่า test cases มีข้อมูลครบถ้วน

## ข้อมูลเพิ่มเติม

### Requirements
- Python 3.8+
- datasets >= 4.4.2
- ragas >= 0.4.2
- openai (สำหรับ TyphoonChatbot)
- chromadb (สำหรับ vector database)

### Performance Tips
1. ใช้ Quick Evaluation สำหรับการทดสอบเบื้องต้น
2. ใช้ Compare Configurations เมื่อต้องการวิเคราะห์เชิงลึก
3. ปรับจำนวน test cases ให้เหมาะสมกับเวลาที่มี
4. เก็บผลลัพธ์ไว้เปรียบเทียบในอนาคต

## License

สคริปต์นี้เป็นส่วนหนึ่งของโปรเจกต์ RAG Chatbot
