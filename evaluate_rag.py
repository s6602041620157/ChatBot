"""
RAG Evaluation Script using Ragas

This script evaluates the RAG chatbot using various metrics:
- Context Precision: How relevant are the retrieved contexts?
- Context Recall: Does the context contain the answer?
- Faithfulness: Is the answer grounded in the context?
- Answer Relevance: Is the answer relevant to the question?
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from chatbot_v04_keywords import HybridKnowledgeBase, TyphoonChatbot
from ragas_result_utils import METRIC_KEYS, summarize_ragas_result


def create_test_dataset() -> List[Dict]:
    """
    Create a test dataset with question-answer pairs

    Returns:
        List of test cases with questions and ground truth answers
    """
    test_cases = [
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
            "question": "ขอข้อมูลการการเปลี่ยนตอนเรียนหน่อย",
            "ground_truth": "นักศึกษาที่ลงทะเบียนเรียนรายวิชาไว้แล้ว หากประสงค์จะ ขอเปลี่ยนตอนเรียน\nสามารถดำเนินการได้ ภายใน 3 สัปดาห์ นับตั้งแต่วันเปิดภาคการศึกษา สำหรับ ภาคการศึกษาปกติ",
        },
        {
            "question": "ระเบียบการลงทะเบียน",
            "ground_truth": "1.กำหนดวันและวิธีการลงทะเบียน\nมหาวิทยาลัยเป็นผู้กำหนดวันและวิธีการลงทะเบียนในแต่ละภาคเรียน หากนักศึกษาไม่ลงทะเบียนตามกำหนด จะ ไม่มีสิทธิ์เข้าสอบกลางภาคและปลายภาค ในภาคเรียนนั้น",
        },
    ]

    return test_cases


def _get_env_int(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        parsed = int(raw_value)
    except ValueError:
        print(f"⚠️  Invalid {name}={raw_value!r}. Using default: {default}")
        return default

    if parsed < minimum:
        print(f"⚠️  {name} must be >= {minimum}. Using default: {default}")
        return default

    return parsed


def create_run_config() -> Tuple[RunConfig, Dict[str, int]]:
    """Create Ragas RunConfig using env-based overrides."""
    run_config_values = {
        "timeout": _get_env_int("RAGAS_TIMEOUT_SECONDS", 300),
        "max_retries": _get_env_int("RAGAS_MAX_RETRIES", 3, minimum=0),
        "max_wait": _get_env_int("RAGAS_MAX_WAIT_SECONDS", 30),
        "max_workers": _get_env_int("RAGAS_MAX_WORKERS", 4),
    }
    run_config = RunConfig(
        timeout=run_config_values["timeout"],
        max_retries=run_config_values["max_retries"],
        max_wait=run_config_values["max_wait"],
        max_workers=run_config_values["max_workers"],
    )
    return run_config, run_config_values


def _normalize_result_payload(results: Any) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, int]]]:
    """Normalize result payload into scalar metrics plus coverage/raw counts."""
    if isinstance(results, dict) and all(isinstance(results.get(key), (int, float)) for key in METRIC_KEYS):
        metrics = {key: float(results.get(key, 0.0)) for key in METRIC_KEYS}
        coverage = results.get("_coverage", {})
        raw_counts = results.get("_raw_counts", {})
        return metrics, coverage, raw_counts

    summary = summarize_ragas_result(results, metric_keys=METRIC_KEYS)
    return summary["metrics"], summary["coverage"], summary["raw_counts"]


def print_metric_coverage(coverage: Dict[str, Dict[str, float]]) -> None:
    """Print per-metric coverage if present."""
    if not coverage:
        return

    print("\nCoverage (successful rows / total rows):")
    for metric in METRIC_KEYS:
        metric_coverage = coverage.get(metric, {})
        valid = int(metric_coverage.get("valid", 0))
        total = int(metric_coverage.get("total", 0))
        ratio = float(metric_coverage.get("ratio", 0.0))
        print(f"   {metric}: {valid}/{total} ({ratio:.0%})")


def save_results(results: Dict, config_name: str, output_dir: str = "evaluation_results"):
    """
    Save evaluation results to a JSON file

    Args:
        results: Evaluation results dictionary
        config_name: Name of the configuration being evaluated
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    metrics, coverage, raw_counts = _normalize_result_payload(results)

    # Convert results to serializable format
    results_dict = {
        "config": config_name,
        "timestamp": timestamp,
        "metrics": metrics,
        "coverage": coverage,
        "raw_counts": raw_counts,
        "evaluator": results.get("_evaluator") if isinstance(results, dict) else None,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {filepath}")
    return filepath


def _create_gemini_evaluator(gemini_api_key: str) -> Tuple[Any, Any, str]:
    """Create Gemini evaluator."""
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "GEMINI_API_KEY is available, but google-genai is not installed. "
            "Install with: pip install google-genai"
        ) from exc

    google_client = genai.Client(api_key=gemini_api_key)
    model_name = os.getenv("RAGAS_GEMINI_MODEL", "gemini-2.0-flash")
    ragas_llm = llm_factory(
        model_name,
        provider="google",
        client=google_client,
        temperature=0,
    )
    ragas_embeddings = HuggingFaceEmbeddings(
        model=os.getenv(
            "RAGAS_HF_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
    )
    return ragas_llm, ragas_embeddings, f"google/{model_name}"


def create_ragas_evaluator() -> Tuple[Any, Any, str]:
    """
    Create LLM and embeddings for Ragas metrics.

    Priority:
    1) Gemini via google-genai (GEMINI_API_KEY)
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        return _create_gemini_evaluator(gemini_api_key)

    raise ValueError("GEMINI_API_KEY not set. Gemini-only evaluator requires GEMINI_API_KEY.")


def evaluate_rag_system(
    knowledge_base: HybridKnowledgeBase,
    chatbot: TyphoonChatbot,
    test_cases: List[Dict],
    use_reranker: bool = True,
    use_compression: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate RAG system using Ragas metrics

    Args:
        knowledge_base: The hybrid knowledge base
        chatbot: The chatbot instance
        test_cases: List of test questions and ground truth answers
        use_reranker: Whether to use re-ranker
        use_compression: Whether to use context compression

    Returns:
        Normalized evaluation results with scalar metrics + metadata
    """
    print("=" * 80)
    print("🔍 Evaluating RAG System")
    print(f"   Re-ranker: {'✅ Enabled' if use_reranker else '❌ Disabled'}")
    print(f"   Context Compression: {'✅ Enabled' if use_compression else '❌ Disabled'}")
    print("=" * 80)

    # Prepare data for evaluation
    questions: List[str] = []
    answers: List[str] = []
    contexts: List[List[str]] = []
    ground_truths: List[str] = []

    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]

        print(f"\n[{i}/{len(test_cases)}] Processing: {question}")

        try:
            # Get relevant contexts
            relevant_knowledge = knowledge_base.search_knowledge(question, n_results=5)
            context_list = [item["text"] for item in relevant_knowledge]

            # Apply compression if enabled
            if use_compression and context_list:
                try:
                    relevant_knowledge = chatbot.compress_context(question, relevant_knowledge)
                    context_list = [item["text"] for item in relevant_knowledge]
                except Exception as e:
                    print(f"   ⚠️  Warning: Context compression failed: {e}")
                    # Continue with uncompressed context

            # Get chatbot answer
            answer = chatbot.chat(question)

            print(f"   ✅ Answer: {answer[:100]}...")
            print(f"   📄 Contexts: {len(context_list)} items")

            questions.append(question)
            answers.append(answer)
            contexts.append(context_list)
            ground_truths.append(ground_truth)

        except Exception as e:
            print(f"   ❌ Error processing question: {e}")
            # Add placeholder data to maintain dataset consistency
            questions.append(question)
            answers.append("Error: Could not generate answer")
            contexts.append(["Error: Could not retrieve context"])
            ground_truths.append(ground_truth)

    # Create dataset for Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    dataset = Dataset.from_dict(data)

    # Evaluate using Ragas metrics
    print("\n" + "=" * 80)
    print("📊 Running Ragas Evaluation...")
    print("=" * 80)

    try:
        ragas_llm, ragas_embeddings, evaluator_name = create_ragas_evaluator()
        run_config, run_config_values = create_run_config()
        print(f"🧪 Ragas evaluator: {evaluator_name}")
        print(
            "⏱️  RunConfig: "
            f"timeout={run_config_values['timeout']}s, "
            f"retries={run_config_values['max_retries']}, "
            f"max_wait={run_config_values['max_wait']}s, "
            f"workers={run_config_values['max_workers']}"
        )

        raw_result = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
            raise_exceptions=False,
            show_progress=True,
        )

        summary = summarize_ragas_result(raw_result, metric_keys=METRIC_KEYS)
        normalized = dict(summary["metrics"])
        normalized["_coverage"] = summary["coverage"]
        normalized["_raw_counts"] = summary["raw_counts"]
        normalized["_evaluator"] = evaluator_name
        normalized["_run_config"] = run_config_values

        print_metric_coverage(normalized["_coverage"])
        return normalized
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print("Please check your API keys and network connection.")
        raise


def compare_configurations():
    """
    Compare different RAG configurations
    """
    load_dotenv()
    typhoon_api_key = os.getenv("TYPHOON_API_KEY")

    if not typhoon_api_key:
        print("❌ TYPHOON_API_KEY not found in environment variables")
        return

    # Create test dataset
    test_cases = create_test_dataset()
    results_all = []

    try:
        # Configuration 1: No Re-ranker, No Compression
        print("\n" + "=" * 80)
        print("📋 Configuration 1: Baseline (No Re-ranker, No Compression)")
        print("=" * 80)
        kb1 = HybridKnowledgeBase(use_reranker=False)
        bot1 = TyphoonChatbot(typhoon_api_key, kb1, use_compression=False)
        result1 = evaluate_rag_system(kb1, bot1, test_cases, use_reranker=False, use_compression=False)
        save_results(result1, "baseline")
        results_all.append(("Baseline", result1))
    except Exception as e:
        print(f"❌ Configuration 1 failed: {e}")
        result1 = None

    try:
        # Configuration 2: With Re-ranker, No Compression
        print("\n" + "=" * 80)
        print("📋 Configuration 2: With Re-ranker Only")
        print("=" * 80)
        kb2 = HybridKnowledgeBase(use_reranker=True)
        bot2 = TyphoonChatbot(typhoon_api_key, kb2, use_compression=False)
        result2 = evaluate_rag_system(kb2, bot2, test_cases, use_reranker=True, use_compression=False)
        save_results(result2, "reranker_only")
        results_all.append(("Re-ranker Only", result2))
    except Exception as e:
        print(f"❌ Configuration 2 failed: {e}")
        result2 = None

    try:
        # Configuration 3: With Re-ranker and Compression
        print("\n" + "=" * 80)
        print("📋 Configuration 3: Full Stack (Re-ranker + Compression)")
        print("=" * 80)
        kb3 = HybridKnowledgeBase(use_reranker=True)
        bot3 = TyphoonChatbot(typhoon_api_key, kb3, use_compression=True)
        result3 = evaluate_rag_system(kb3, bot3, test_cases, use_reranker=True, use_compression=True)
        save_results(result3, "full_stack")
        results_all.append(("Full Stack", result3))
    except Exception as e:
        print(f"❌ Configuration 3 failed: {e}")
        result3 = None

    if not result1:
        print("\n⚠️  Cannot compare: Baseline configuration failed")
        return

    # Print comparison
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS COMPARISON")
    print("=" * 80)

    print("\n1️⃣  Baseline (No Re-ranker, No Compression):")
    if result1:
        print(f"   Context Precision: {result1['context_precision']:.4f}")
        print(f"   Context Recall: {result1['context_recall']:.4f}")
        print(f"   Faithfulness: {result1['faithfulness']:.4f}")
        print(f"   Answer Relevancy: {result1['answer_relevancy']:.4f}")
        print_metric_coverage(result1.get("_coverage", {}))

    if result2:
        print("\n2️⃣  With Re-ranker Only:")
        print(
            f"   Context Precision: {result2['context_precision']:.4f} "
            f"({'+' if result2['context_precision'] > result1['context_precision'] else ''}"
            f"{(result2['context_precision'] - result1['context_precision']):.4f})"
        )
        print(
            f"   Context Recall: {result2['context_recall']:.4f} "
            f"({'+' if result2['context_recall'] > result1['context_recall'] else ''}"
            f"{(result2['context_recall'] - result1['context_recall']):.4f})"
        )
        print(
            f"   Faithfulness: {result2['faithfulness']:.4f} "
            f"({'+' if result2['faithfulness'] > result1['faithfulness'] else ''}"
            f"{(result2['faithfulness'] - result1['faithfulness']):.4f})"
        )
        print(
            f"   Answer Relevancy: {result2['answer_relevancy']:.4f} "
            f"({'+' if result2['answer_relevancy'] > result1['answer_relevancy'] else ''}"
            f"{(result2['answer_relevancy'] - result1['answer_relevancy']):.4f})"
        )
        print_metric_coverage(result2.get("_coverage", {}))

    if result3:
        print("\n3️⃣  Full Stack (Re-ranker + Compression):")
        print(
            f"   Context Precision: {result3['context_precision']:.4f} "
            f"({'+' if result3['context_precision'] > result1['context_precision'] else ''}"
            f"{(result3['context_precision'] - result1['context_precision']):.4f})"
        )
        print(
            f"   Context Recall: {result3['context_recall']:.4f} "
            f"({'+' if result3['context_recall'] > result1['context_recall'] else ''}"
            f"{(result3['context_recall'] - result1['context_recall']):.4f})"
        )
        print(
            f"   Faithfulness: {result3['faithfulness']:.4f} "
            f"({'+' if result3['faithfulness'] > result1['faithfulness'] else ''}"
            f"{(result3['faithfulness'] - result1['faithfulness']):.4f})"
        )
        print(
            f"   Answer Relevancy: {result3['answer_relevancy']:.4f} "
            f"({'+' if result3['answer_relevancy'] > result1['answer_relevancy'] else ''}"
            f"{(result3['answer_relevancy'] - result1['answer_relevancy']):.4f})"
        )
        print_metric_coverage(result3.get("_coverage", {}))

    print("\n" + "=" * 80)

    return results_all


def main():
    """Main function with improved user interaction"""
    load_dotenv()
    typhoon_api_key = os.getenv("TYPHOON_API_KEY")

    if not typhoon_api_key:
        print("❌ TYPHOON_API_KEY not found in environment variables")
        print("💡 Please create a .env file with: TYPHOON_API_KEY=your_api_key")
        return

    print("=" * 80)
    print("🚀 RAG System Evaluation with Ragas")
    print("=" * 80)
    print("\nChoose evaluation mode:")
    print("1. Quick Evaluation (Full Stack Configuration)")
    print("2. Compare All Configurations (Baseline, Re-ranker, Full Stack)")
    print("3. Custom Configuration")
    print("4. Exit")

    try:
        choice = input("\nEnter your choice (1-4): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n👋 Exiting...")
        return

    test_cases = create_test_dataset()

    try:
        if choice == "1":
            # Quick evaluation with full stack
            print("\n📋 Running Quick Evaluation (Full Stack)...")
            kb = HybridKnowledgeBase(use_reranker=True)
            chatbot = TyphoonChatbot(typhoon_api_key, kb, use_compression=True)
            result = evaluate_rag_system(kb, chatbot, test_cases, use_reranker=True, use_compression=True)

            print("\n" + "=" * 80)
            print("📊 EVALUATION RESULTS")
            print("=" * 80)
            print(f"Context Precision: {result['context_precision']:.4f}")
            print(f"Context Recall: {result['context_recall']:.4f}")
            print(f"Faithfulness: {result['faithfulness']:.4f}")
            print(f"Answer Relevancy: {result['answer_relevancy']:.4f}")
            print_metric_coverage(result.get("_coverage", {}))
            print("=" * 80)

            save_results(result, "quick_eval_full_stack")

        elif choice == "2":
            # Compare all configurations
            print("\n📋 Running Configuration Comparison...")
            print("⚠️  This will take longer as it runs 3 different configurations")
            compare_configurations()

        elif choice == "3":
            # Custom configuration
            print("\n📋 Custom Configuration")
            try:
                use_reranker = input("Use re-ranker? (y/n): ").strip().lower() == "y"
                use_compression = input("Use context compression? (y/n): ").strip().lower() == "y"
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Cancelled...")
                return

            print(f"\n🔧 Configuration: Re-ranker={use_reranker}, Compression={use_compression}")
            kb = HybridKnowledgeBase(use_reranker=use_reranker)
            chatbot = TyphoonChatbot(typhoon_api_key, kb, use_compression=use_compression)
            result = evaluate_rag_system(
                kb,
                chatbot,
                test_cases,
                use_reranker=use_reranker,
                use_compression=use_compression,
            )

            print("\n" + "=" * 80)
            print("📊 EVALUATION RESULTS")
            print("=" * 80)
            print(f"Context Precision: {result['context_precision']:.4f}")
            print(f"Context Recall: {result['context_recall']:.4f}")
            print(f"Faithfulness: {result['faithfulness']:.4f}")
            print(f"Answer Relevancy: {result['answer_relevancy']:.4f}")
            print_metric_coverage(result.get("_coverage", {}))
            print("=" * 80)

            config_name = f"custom_reranker{use_reranker}_compression{use_compression}"
            save_results(result, config_name)

        elif choice == "4":
            print("\n👋 Exiting...")
            return

        else:
            print("\n❌ Invalid choice. Please run again and select 1-4.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during evaluation: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()
