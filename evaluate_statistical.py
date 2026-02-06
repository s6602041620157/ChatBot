#!/usr/bin/env python3
"""
Deterministic/สถิติประเมิน RAG โดยไม่ใช้ LLM-as-judge

สิ่งที่วัด (ไม่เรียก LLM สำหรับการให้คะแนน):
- answer_exact_match  : ตรงกันหลัง normalize
- answer_f1           : F1 ระหว่างโทเคนคำตอบกับ ground truth
- answer_similarity   : difflib ratio ระหว่างคำตอบกับ ground truth
- context_recall      : สัดส่วนโทเคนของ ground truth ที่พบใน context รวม
- context_hit_rate    : สัดส่วนเคสที่ context คลุมเนื้อหาได้ (similarity หรือ recall เกิน threshold)

เอาต์พุต: พิมพ์สรุปบนหน้าจอและบันทึกเป็น JSON ใน evaluation_results/
"""

import json
import os
import re
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from statistics import mean
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from pythainlp.tokenize import word_tokenize

from chatbot_v04_keywords import HybridKnowledgeBase, TyphoonChatbot


# ---------- Helper functions ----------

def normalize_text(text: str) -> str:
    """Lowercase + collapse whitespace for robust matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> List[str]:
    """Thai-friendly tokenization (falls back to simple split)."""
    try:
        return [tok for tok in word_tokenize(normalize_text(text), engine="newmm") if tok.strip()]
    except Exception:
        return [tok for tok in normalize_text(text).split(" ") if tok]


def f1_score(pred: str, ref: str) -> float:
    """Token-level F1 (macro over unique tokens not needed here)."""
    p_tokens = tokenize(pred)
    r_tokens = tokenize(ref)
    if not p_tokens or not r_tokens:
        return 0.0

    common = Counter(p_tokens) & Counter(r_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(p_tokens)
    recall = overlap / len(r_tokens)
    return (2 * precision * recall) / (precision + recall)


def similarity_ratio(a: str, b: str) -> float:
    """Character-level similarity using difflib (0-1)."""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def context_recall(contexts: List[str], ref: str) -> float:
    """Recall of ground-truth tokens covered by concatenated contexts."""
    ref_tokens = tokenize(ref)
    if not ref_tokens:
        return 0.0
    ctx_tokens = tokenize(" ".join(contexts))
    common = Counter(ctx_tokens) & Counter(ref_tokens)
    return sum(common.values()) / len(ref_tokens)


# ---------- Test data ----------

def create_test_dataset() -> List[Dict[str, str]]:
    """Ground-truth QA pairs (เดิมใช้ในสคริปต์ ragas)."""
    return [
        {
            "question": "หลักสูตรวิศวกรรมโยธาและการศึกษาเรียนกี่ปี",
            "ground_truth": "หลักสูตรวิศวกรรมโยธาและการศึกษาใช้เวลาศึกษา 5 ปี",
        },
        {
            "question": "จบหลักสูตรวิศวกรรมโยธาและการศึกษาแล้วได้อะไรบ้าง",
            "ground_truth": "จบแล้วได้ทั้งวุฒิครูและวิศวกร สามารถประกอบอาชีพได้ทั้งสองด้าน",
        },
        {
            "question": "คณะครุศาสตร์อุตสาหกรรมอยู่มหาวิทยาลัยไหน",
            "ground_truth": "คณะครุศาสตร์อุตสาหกรรมอยู่ที่มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (มจพ.)",
        },
        {
            "question": "การรับสมัครนักศึกษาใหม่เปิดรับสมัครเมื่อไหร่",
            "ground_truth": "ข้อมูลการรับสมัครนักศึกษาใหม่จะประกาศในช่วงเดือนพฤศจิกายน-ธันวาคม",
        },
        {
            "question": "ค่าเทอมหลักสูตรวิศวกรรมโยธาและการศึกษาเท่าไหร่",
            "ground_truth": "ค่าเทอมอยู่ที่ประมาณ 17,000-20,000 บาทต่อภาคการศึกษา",
        },
    ]


# ---------- Core evaluation ----------

def evaluate_case(
    question: str,
    ground_truth: str,
    kb: HybridKnowledgeBase,
    bot: TyphoonChatbot,
    n_ctx: int = 5,
    sim_threshold: float = 0.65,
) -> Dict:
    """Run one Q/A + retrieval and compute deterministic metrics."""
    contexts = kb.search_knowledge(question, n_results=n_ctx)
    ctx_texts = [c["text"] for c in contexts]
    answer = bot.chat(question)

    metrics = {
        "answer_exact_match": 1.0 if normalize_text(answer) == normalize_text(ground_truth) else 0.0,
        "answer_f1": f1_score(answer, ground_truth),
        "answer_similarity": similarity_ratio(answer, ground_truth),
        "context_recall": context_recall(ctx_texts, ground_truth),
    }

    ctx_similarity = similarity_ratio(" ".join(ctx_texts), ground_truth) if ctx_texts else 0.0
    metrics["context_hit"] = 1.0 if (ctx_similarity >= sim_threshold or metrics["context_recall"] >= 0.5) else 0.0

    return {
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
        "contexts": ctx_texts,
        "metrics": metrics,
    }


def aggregate_results(results: List[Dict]) -> Dict[str, float]:
    """Mean of each metric across test cases."""
    keys = results[0]["metrics"].keys()
    return {k: mean([r["metrics"][k] for r in results]) for k in keys}


def save_results(results: List[Dict], summary: Dict[str, float]) -> str:
    """Persist detailed + summary metrics to JSON."""
    os.makedirs("evaluation_results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("evaluation_results", f"deterministic_eval_{ts}.json")
    payload = {"summary": summary, "cases": results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# ---------- CLI ----------

def main():
    load_dotenv()
    api_key = os.getenv("TYPHOON_API_KEY")
    if not api_key:
        print("❌ ไม่พบ TYPHOON_API_KEY ใน environment variables")
        return

    print("=" * 80)
    print("🧮 Deterministic RAG Evaluation (no LLM judge)")
    print("=" * 80)

    kb = HybridKnowledgeBase(
        persist_directory="./chroma_db",
        collection_name="chatbot_knowledge",
        use_reranker=True,
        use_keyword_boost=True,
    )
    bot = TyphoonChatbot(api_key, kb, use_compression=False)

    tests = create_test_dataset()
    results = []
    for i, case in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] {case['question']}")
        res = evaluate_case(case["question"], case["ground_truth"], kb, bot)
        results.append(res)
        m = res["metrics"]
        print(f"   - answer_f1: {m['answer_f1']:.3f} | answer_sim: {m['answer_similarity']:.3f}")
        print(f"   - context_recall: {m['context_recall']:.3f} | context_hit: {m['context_hit']:.0f}")

    summary = aggregate_results(results)

    print("\n" + "=" * 80)
    print("📊 SUMMARY (mean over test set)")
    for k, v in summary.items():
        print(f"   {k:20s}: {v:.3f}")
    print("=" * 80)

    out_path = save_results(results, summary)
    print(f"\n💾 Saved to {out_path}")


if __name__ == "__main__":
    main()
