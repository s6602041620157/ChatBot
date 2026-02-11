#!/usr/bin/env python3
"""
Quick test script to verify Gemini-only evaluation setup before running full evaluation.
"""

import os

from dotenv import load_dotenv


def test_setup() -> bool:
    """Test if all required components are available."""

    print("=" * 60)
    print("🔍 Testing RAG Evaluation Setup (Gemini-only)")
    print("=" * 60)

    # Test 1: Environment variables
    print("\n1️⃣  Checking environment variables...")
    load_dotenv()

    typhoon_key = os.getenv("TYPHOON_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if typhoon_key:
        print(f"   ✅ TYPHOON_API_KEY found (length: {len(typhoon_key)} chars)")
    else:
        print("   ❌ TYPHOON_API_KEY not found")
        print("   💡 Create a .env file with: TYPHOON_API_KEY=your_key")
        return False

    if gemini_key:
        print(f"   ✅ GEMINI_API_KEY found (length: {len(gemini_key)} chars)")
    else:
        print("   ❌ GEMINI_API_KEY not found")
        print("   💡 Create a .env file with: GEMINI_API_KEY=your_key")
        return False

    # Test 2: Required imports
    print("\n2️⃣  Checking required imports...")
    try:
        from datasets import Dataset  # noqa: F401
        print("   ✅ datasets package imported")
    except ImportError as e:
        print(f"   ❌ datasets package failed: {e}")
        return False

    try:
        from ragas import evaluate  # noqa: F401
        print("   ✅ ragas package imported")
    except ImportError as e:
        print(f"   ❌ ragas package failed: {e}")
        return False

    try:
        from google import genai  # noqa: F401
        print("   ✅ google-genai package imported")
    except ImportError as e:
        print(f"   ❌ google-genai package failed: {e}")
        return False

    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        print("   ✅ sentence-transformers package imported")
    except ImportError as e:
        print(f"   ❌ sentence-transformers package failed: {e}")
        return False

    try:
        import jsonref  # noqa: F401
        print("   ✅ jsonref package imported")
    except ImportError as e:
        print(f"   ❌ jsonref package failed: {e}")
        return False

    try:
        from chatbot_v04_keywords import HybridKnowledgeBase, TyphoonChatbot  # noqa: F401
        print("   ✅ chatbot modules imported")
    except ImportError as e:
        print(f"   ❌ chatbot modules failed: {e}")
        return False

    # Test 3: Knowledge base files
    print("\n3️⃣  Checking knowledge base files...")
    kb_files = [
        "chroma_db",
        "knowledge_base.json",
    ]

    found_kb = False
    for kb_file in kb_files:
        if os.path.exists(kb_file):
            print(f"   ✅ Found: {kb_file}")
            found_kb = True

    if not found_kb:
        print("   ⚠️  No knowledge base files found")
        print("   💡 You may need to initialize the knowledge base first")

    # Test 4: Gemini client initialization
    print("\n4️⃣  Checking Gemini evaluator readiness...")
    try:
        from google import genai

        _ = genai.Client(api_key=gemini_key)
        print("   ✅ Gemini client initialized")
    except Exception as e:
        print(f"   ❌ Gemini client initialization failed: {e}")
        return False

    # Test 5: Component initialization
    print("\n5️⃣  Testing component initialization...")
    try:
        from chatbot_v04_keywords import HybridKnowledgeBase, TyphoonChatbot

        kb = HybridKnowledgeBase(use_reranker=False)
        print("   ✅ HybridKnowledgeBase initialized")
    except Exception as e:
        print(f"   ❌ HybridKnowledgeBase failed: {e}")
        return False

    try:
        _ = TyphoonChatbot(typhoon_key, kb, use_compression=False)
        print("   ✅ TyphoonChatbot initialized")
    except Exception as e:
        print(f"   ❌ TyphoonChatbot failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ All checks passed! Ready to run evaluation.")
    print("✅ Evaluator path: Gemini-only")
    print("=" * 60)
    print("\n💡 Run: python3 evaluate_rag.py")

    return True


if __name__ == "__main__":
    success = test_setup()
    raise SystemExit(0 if success else 1)
