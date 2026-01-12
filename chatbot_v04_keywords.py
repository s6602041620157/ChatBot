import json
import os
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize

class HybridKnowledgeBase:
    """Class to handle hybrid retrieval (BM25 + Vector Search + Keyword Matching)"""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "chatbot_knowledge",
                 use_reranker: bool = True, use_keyword_boost: bool = True):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.use_reranker = use_reranker
        self.use_keyword_boost = use_keyword_boost

        # Initialize the embedding model
        print("🤖 Loading embedding model...")
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        # Initialize Cross-Encoder for re-ranking
        if self.use_reranker:
            print("🎯 Loading Cross-Encoder for re-ranking...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✅ Re-ranker loaded successfully")

        # Initialize ChromaDB client
        print(f"💾 Connecting to ChromaDB at: {persist_directory}")
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get the collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✅ Connected to collection: {collection_name}")
            print(f"📚 Collection contains {self.collection.count()} documents")
        except Exception as e:
            print(f"❌ Error loading collection: {e}")
            raise

        # Load all documents for BM25
        print("📝 Loading documents for BM25 indexing...")
        self._load_documents_for_bm25()

    def _load_documents_for_bm25(self):
        """Load all documents and create BM25 index"""
        # Get all documents from ChromaDB
        all_docs = self.collection.get(include=['documents', 'metadatas'])
        self.documents = all_docs['documents']
        self.doc_ids = all_docs['ids']
        self.metadatas = all_docs['metadatas']

        # Tokenize documents for BM25 using Thai word tokenizer
        print("🔤 Tokenizing documents for BM25...")
        self.tokenized_docs = [word_tokenize(doc, engine='newmm') for doc in self.documents]

        # Create BM25 index
        print("🔍 Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"✅ BM25 index ready with {len(self.documents)} documents")

    def _extract_query_keywords(self, query: str, top_n: int = 10) -> List[str]:
        """
        สกัด keywords จาก query

        Args:
            query: คำถามของผู้ใช้
            top_n: จำนวน keywords สูงสุดที่ต้องการ

        Returns:
            List ของ keywords
        """
        # ตัดคำ
        words = word_tokenize(query, engine='newmm')

        # กรอง stopwords (เบื้องต้น)
        from pythainlp.corpus import thai_stopwords
        stopwords = thai_stopwords()

        # กรองคำและเก็บเฉพาะคำที่มีความหมาย
        keywords = []
        for word in words:
            word = word.strip()
            if (len(word) >= 2 and
                word not in stopwords and
                not word.isspace() and
                not word.isdigit()):
                keywords.append(word)

        return keywords[:top_n]

    def _calculate_keyword_match_score(self, query_keywords: List[str],
                                       doc_metadata: Dict) -> float:
        """
        คำนวณคะแนนการตรงกันของ keywords

        Args:
            query_keywords: keywords จาก query
            doc_metadata: metadata ของ document

        Returns:
            คะแนนการตรงกัน (0.0 - 1.0)
        """
        if not self.use_keyword_boost or 'keywords' not in doc_metadata:
            return 0.0

        doc_keywords = doc_metadata.get('keywords', [])
        if not doc_keywords or not query_keywords:
            return 0.0

        # นับจำนวน keywords ที่ตรงกัน
        matches = 0
        partial_matches = 0

        for q_kw in query_keywords:
            q_kw_lower = q_kw.lower()

            # Exact match
            if q_kw in doc_keywords or q_kw_lower in [k.lower() for k in doc_keywords]:
                matches += 1
            else:
                # Partial match (substring)
                for d_kw in doc_keywords:
                    if q_kw_lower in d_kw.lower() or d_kw.lower() in q_kw_lower:
                        partial_matches += 0.5
                        break

        # คะแนนรวม (ให้น้ำหนัก exact match มากกว่า)
        total_score = matches + partial_matches

        # Normalize (0.0 - 1.0)
        max_possible = len(query_keywords)
        normalized_score = min(total_score / max_possible, 1.0) if max_possible > 0 else 0.0

        return normalized_score

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform BM25 keyword search

        Args:
            query: User's question
            top_k: Number of top results to return

        Returns:
            List of (doc_index, score) tuples
        """
        # Tokenize query using Thai tokenizer
        tokenized_query = word_tokenize(query, engine='newmm')

        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get top-k results with indices
        top_indices = bm25_scores.argsort()[-top_k:][::-1]
        results = [(idx, bm25_scores[idx]) for idx in top_indices if bm25_scores[idx] > 0]

        return results

    def _vector_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform vector similarity search

        Args:
            query: User's question
            top_k: Number of top results to return

        Returns:
            List of (doc_index, similarity_score) tuples
        """
        # Generate embedding for the query
        query_embedding = self.model.encode([query])

        # Search in vector database
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )

        # Convert to (index, score) format
        # ChromaDB returns distance (lower is better), convert to similarity (higher is better)
        vector_results = []
        if results['documents'][0]:
            for doc_text, distance in zip(results['documents'][0], results['distances'][0]):
                # Find document index
                try:
                    doc_idx = self.documents.index(doc_text)
                    # Convert distance to similarity score (1 - normalized_distance)
                    similarity = 1 / (1 + distance)  # Higher score = more similar
                    vector_results.append((doc_idx, similarity))
                except ValueError:
                    continue

        return vector_results

    def _rerank(self, query: str, documents: List[Dict[str, any]], top_k: int = 5) -> List[Dict[str, any]]:
        """
        Re-rank documents using Cross-Encoder

        Args:
            query: User's question
            documents: List of documents with their scores from hybrid search
            top_k: Number of top results to return after re-ranking

        Returns:
            Re-ranked list of documents
        """
        if not self.use_reranker or not documents:
            return documents[:top_k]

        # Prepare pairs of (query, document) for cross-encoder
        pairs = [(query, doc['text']) for doc in documents]

        # Get re-ranking scores
        rerank_scores = self.reranker.predict(pairs)

        # Add rerank scores to documents
        for doc, score in zip(documents, rerank_scores):
            doc['rerank_score'] = float(score)

        # Sort by rerank score and return top-k
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)[:top_k]

        # Update ranks after re-ranking
        for rank, doc in enumerate(reranked, 1):
            doc['rank'] = rank

        return reranked

    def search_knowledge(self, query: str, n_results: int = 5,
                        bm25_weight: float = 0.3, vector_weight: float = 0.5,
                        keyword_weight: float = 0.2) -> List[Dict[str, str]]:
        """
        Hybrid search combining BM25, vector similarity, and keyword matching

        Args:
            query: User's question
            n_results: Number of top results to return
            bm25_weight: Weight for BM25 scores (default 0.3)
            vector_weight: Weight for vector scores (default 0.5)
            keyword_weight: Weight for keyword matching scores (default 0.2)

        Returns:
            List of relevant knowledge items with text and relevance score
        """
        # สกัด keywords จาก query
        query_keywords = self._extract_query_keywords(query, top_n=15)

        # Perform both searches
        bm25_results = self._bm25_search(query, top_k=15)
        vector_results = self._vector_search(query, top_k=15)

        # Normalize scores for both methods
        def normalize_scores(results: List[Tuple[int, float]]) -> Dict[int, float]:
            if not results:
                return {}
            scores = [score for _, score in results]
            min_score = min(scores)
            max_score = max(scores)

            if max_score == min_score:
                return {idx: 1.0 for idx, _ in results}

            return {
                idx: (score - min_score) / (max_score - min_score)
                for idx, score in results
            }

        bm25_normalized = normalize_scores(bm25_results)
        vector_normalized = normalize_scores(vector_results)

        # Calculate keyword matching scores
        keyword_scores = {}
        all_indices = set(bm25_normalized.keys()) | set(vector_normalized.keys())

        for idx in all_indices:
            if idx < len(self.metadatas):
                keyword_score = self._calculate_keyword_match_score(
                    query_keywords,
                    self.metadatas[idx]
                )
                keyword_scores[idx] = keyword_score

        # Combine scores using weighted sum
        combined_scores = {}

        for idx in all_indices:
            bm25_score = bm25_normalized.get(idx, 0.0)
            vector_score = vector_normalized.get(idx, 0.0)
            keyword_score = keyword_scores.get(idx, 0.0)

            # Weighted combination
            combined_scores[idx] = (
                bm25_weight * bm25_score +
                vector_weight * vector_score +
                keyword_weight * keyword_score
            )

        # Sort by combined score and get more results for re-ranking
        top_k_for_rerank = min(n_results * 3, 15)  # Get 3x results for re-ranking
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_for_rerank]

        # Format results
        relevant_knowledge = []
        for rank, (idx, score) in enumerate(sorted_results, 1):
            relevant_knowledge.append({
                'text': self.documents[idx],
                'score': score,
                'rank': rank,
                'bm25_score': bm25_normalized.get(idx, 0.0),
                'vector_score': vector_normalized.get(idx, 0.0),
                'keyword_score': keyword_scores.get(idx, 0.0),
                'metadata': self.metadatas[idx] if idx < len(self.metadatas) else {},
                'matched_keywords': self._get_matched_keywords(query_keywords, self.metadatas[idx]) if idx < len(self.metadatas) else []
            })

        # Apply re-ranking if enabled
        if self.use_reranker and relevant_knowledge:
            relevant_knowledge = self._rerank(query, relevant_knowledge, top_k=n_results)
        else:
            relevant_knowledge = relevant_knowledge[:n_results]

        return relevant_knowledge

    def _get_matched_keywords(self, query_keywords: List[str], metadata: Dict) -> List[str]:
        """หา keywords ที่ตรงกันระหว่าง query และ document"""
        if 'keywords' not in metadata:
            return []

        doc_keywords = metadata.get('keywords', [])
        matched = []

        for q_kw in query_keywords:
            for d_kw in doc_keywords:
                if q_kw.lower() == d_kw.lower() or q_kw.lower() in d_kw.lower() or d_kw.lower() in q_kw.lower():
                    if d_kw not in matched:
                        matched.append(d_kw)

        return matched[:5]  # Top 5 matched keywords

    def get_context_string(self, relevant_items: List[Dict[str, str]]) -> str:
        """Convert relevant knowledge items to context string with keyword information"""
        if not relevant_items:
            return ""

        context = "ข้อมูลที่เกี่ยวข้องจากฐานความรู้:\n\n"
        for item in relevant_items:
            context += f"{item['rank']}. {item['text']}\n"

            # เพิ่มข้อมูล keywords ที่ตรงกัน (ถ้ามี)
            if item.get('matched_keywords'):
                context += f"   [Keywords: {', '.join(item['matched_keywords'][:3])}]\n"

            context += "\n"

        return context


class TyphoonChatbot:
    """Main chatbot class using Typhoon API - with keyword-enhanced search"""

    def __init__(self, api_key: str, knowledge_base: HybridKnowledgeBase,
                 use_compression: bool = True):
        self.api_key = api_key
        self.knowledge_base = knowledge_base
        self.use_compression = use_compression
        self.setup_typhoon()
        self.conversation_history = []

    def setup_typhoon(self):
        """Initialize Typhoon API"""
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.opentyphoon.ai/v1"
            )
            print("✅ เชื่อมต่อ Typhoon API สำเร็จ!")
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ Typhoon API: {e}")
            raise

    def expand_query(self, query: str) -> str:
        """
        Expand user query to improve search results for natural language questions (TH focus).
        แก้/เพิ่มจากเดิมโดยอิงหัวข้อจริงในหลักสูตร CED/TCT/TM
        """
        q = query.strip()
        ql = q.lower()

        # 🎯 ตรวจจับคำถามแบบง่าย: ถ้าถามแค่ "กี่ปี" และไม่มีคำถามซับซ้อนอื่น ให้ขยายน้อยมาก
        is_simple_duration_question = (
            any(w in ql for w in ['กี่ปี', 'นานเท่าไหร่', 'ใช้เวลา']) and
            not any(w in ql for w in ['วิชา', 'รายวิชา', 'ภาคเรียน', 'แผน', 'อาจารย์', 'ค่า'])
        )

        if is_simple_duration_question:
            # คำถามง่าย เพิ่มแค่คำที่เกี่ยวข้องกับระยะเวลาเท่านั้น
            q += " ระยะเวลา ปี ปีการศึกษา ระยะเวลาเรียน ระยะเวลาตลอดหลักสูตร ระยะเวลาการศึกษา"
            return q  # ⚠️ Return ทันทีไม่ต้องขยายเพิ่ม

        # 1) เจตนาทั่วไป/ภาษาพูด
        if any(w in ql for w in ['อยาก', 'ต้องการ', 'สนใจ']):
            if 'รู้' in ql or 'ทราบ' in ql:
                q += " ข้อมูล รายละเอียด"

        # 2) คำถามปลายเปิด/วิธีทำ
        if any(w in ql for w in ['ทำไง', 'ยังไง']):
            q += " วิธีการ ขั้นตอน"
        if 'มี' in ql and 'ไหม' in ql:
            q += " มีหรือไม่"
        if any(w in ql for w in ['คืออะไร', 'คือ อะไร', 'คือ?', 'คือ ']):
            q += " ความหมาย คำจำกัดความ นิยาม"
        if any(w in ql for w in ['ต่างกัน', 'เปรียบเทียบ', 'เทียบ']):
            q += " ความแตกต่าง เปรียบเทียบ เกณฑ์"

        # 3) หัวข้อรับสมัคร/คุณสมบัติ/เอกสาร/เทียบโอน
        if any(w in ql for w in ['สมัคร', 'รับเข้า', 'คุณสมบัติ', 'เทียบโอน', 'โอนหน่วยกิต']):
            q += " การรับสมัคร คุณสมบัติ เอกสาร การเทียบโอนหน่วยกิต เงื่อนไขการลงทะเบียน"

        # 4) หลักสูตร/หน่วยกิต/โครงสร้างรายวิชา
        if any(w in ql for w in ['หน่วยกิต', 'โครงสร้าง', 'หลักสูตร', 'หมวดวิชา']):
            q += " โครงสร้างหลักสูตร จำนวนหน่วยกิต หมวดวิชาศึกษาทั่วไป กลุ่มวิชาการศึกษา กลุ่มวิชาเทคโนโลยีคอมพิวเตอร์ กลุ่มวิชาชีพ รายวิชา รายวิชาแกน รายวิชาเลือกเสรี แผนการเรียน"

        # 5) ปีการศึกษา/ระยะเวลา/ภาคเรียน/ฤดูร้อน
        if any(w in ql for w in ['กี่ปี', 'นานเท่าไหร่', 'ใช้เวลา']):
            # ถ้าถามแค่เรื่องระยะเวลา ขยายแค่นิดเดียว
            if any(w in ql for w in ['กี่ปี', 'นานเท่าไหร่', 'ใช้เวลา']) and not any(w in ql for w in ['วิชา', 'รายวิชา', 'แผน', 'โครงสร้าง']):
                q += " ระยะเวลาเรียน ปีการศึกษา ระยะเวลาการศึกษา ระยะเวลาตลอดหลักสูตร"
            else:
                q += " ระยะเวลาเรียน ปีการศึกษา ระบบทวิภาค ปฏิทินการศึกษา ภาคฤดูร้อน ระยะเวลาแต่ละภาค"
        elif any(w in ql for w in ['ภาคเรียน', 'เปิดเทอม', 'ฤดูร้อน']):
            q += " ระยะเวลาเรียน ปีการศึกษา ระบบทวิภาค ปฏิทินการศึกษา ภาคฤดูร้อน ระยะเวลาแต่ละภาค"

        # 6) เนื้อหาวิชา/แผนรายปี/ผลลัพธ์การเรียนรู้
        if any(w in ql for w in ['วิชา', 'รายวิชา', 'เนื้อหา', 'แผนการเรียน', 'ปีที่']):
            q += " รายวิชา หมวดวิชา โครงสร้างหลักสูตร ผลลัพธ์การเรียนรู้รายปี สมรรถนะผู้สำเร็จการศึกษา"

        # 7) ฝึกงาน/ฝึกสอน/ภาคปฏิบัติ/ชั่วโมง
        if any(w in ql for w in ['ฝึกงาน', 'สหกิจ', 'ฝึกสอน', 'ปฏิบัติการสอน', 'ชั่วโมง']):
            q += " ฝึกงาน S/U ชั่วโมงปฏิบัติการสอน ฝึกอบรม ระเบียบฝึกประสบการณ์ คุณสมบัติก่อนฝึก"

        # 8) ภาษาในการสอน
        if any(w in ql for w in ['ภาษา', 'english', 'อังกฤษ', 'สองภาษา', 'bilingual']):
            q += " การจัดการเรียนการสอนภาษาไทยและภาษาอังกฤษ เอกสาร/ตำราภาษาไทยและอังกฤษ"

        # 9) อาชีพ/จบแล้วทำอะไรได้บ้าง/วุฒิ/วิชาชีพ
        if any(w in ql for w in ['จบแล้ว', 'จบไป', 'สำเร็จการศึกษา', 'ทำงานอะไร', 'อาชีพ', 'วุฒิ']):
            q += " ปริญญา วุฒิการศึกษา อาชีพที่ประกอบได้ นักพัฒนาโปรแกรม นักวิชาการคอมพิวเตอร์ นักวิเคราะห์ระบบ ครูช่าง วิทยากรฝึกอบรม ผู้ดูแลระบบคอมพิวเตอร์ ใบประกอบวิชาชีพที่เกี่ยวข้อง"

        # 10) หน่วยงาน/คณาจารย์/สถานที่เรียน/ความร่วมมือ
        if any(w in ql for w in ['อาจารย์', 'ผู้รับผิดชอบ', 'อาจารย์ผู้สอน', 'สถานที่เรียน', 'เรียนที่ไหน', 'ความร่วมมือ', 'พันธมิตร']):
            q += " ชื่ออาจารย์ผู้รับผิดชอบหลักสูตร คุณวุฒิการศึกษา สถานที่จัดการเรียนการสอน ความร่วมมือกับสถาบัน/บริษัท"

        # 11) มาตรฐานวิชาชีพ/หน่วยงานกำกับ (เช่น คุรุสภา/สภาวิศวกร)
        if any(w in ql for w in ['คุรุสภา', 'ครูวิศวกรรม', 'สภาวิศวกร', 'วศ.']):
            q += " เกณฑ์มาตรฐานวิชาชีพครู มคอ มาตรฐานสภาวิศวกร คุณสมบัติวิชาชีพและจรรยาบรรณ"

        # 12) เฉพาะสาขา (โยธาและการศึกษา / คอมพิวเตอร์ศึกษา)
        if any(w in ql for w in ['โยธา', 'วิศวกรรมโยธา', 'civil']):
            q += " หลักสูตร 5 ปี วิศวกรรมโยธาและการศึกษา STEM ผลลัพธ์การเรียนรู้ งานโครงสร้างพื้นฐาน ความร่วมมือภาคอุตสาหกรรม"

        # ⚠️ พิเศษ: ถ้าถามเรื่อง "เทียบโอน" + "กี่ปี" ไม่ต้องเพิ่มรายละเอียดวิชา เพราะต้องการแค่ระยะเวลา
        if any(w in ql for w in ['เทียบโอน', 'ปวส']) and any(w in ql for w in ['กี่ปี', 'ระยะเวลา', 'นานเท่าไหร่']):
            q += " หลักสูตรเทียบโอน ปวส 3 ปี 4 ปี ระยะเวลาตลอดหลักสูตร"
        elif any(w in ql for w in ['คอมพิวเตอร์ศึกษา', 'เทคโนโลยีคอมพิวเตอร์', 'ced', 'tct', 'computer']):
            q += " รายวิชากลุ่มวิชาเทคโนโลยีคอมพิวเตอร์ ผลลัพธ์การเรียนรู้รายปี IoT เครือข่าย ระบบปฏิบัติการ วิทยาการข้อมูล เว็บและมือถือ"

        # 13) คำถามเรื่องค่าใช้จ่าย (ถ้ามีบริบทคำว่า 'ค่า' ร่วมกับคำเฉพาะ)
        if 'ค่า' in ql and any(w in ql for w in ['ใช้จ่าย', 'เทอม', 'เล่าเรียน', 'ลงทะเบียน', 'ค่าธรรมเนียม']):
            q += " ค่าธรรมเนียม ค่าลงทะเบียน ค่าใช้จ่ายประมาณการ"

        # คืนค่า
        return q

    def compress_context(self, query: str, contexts: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Compress context by filtering only relevant parts using LLM
        """
        if not self.use_compression or not contexts:
            return contexts

        compressed_contexts = []

        for ctx in contexts:
            # Create compression prompt
            compression_prompt = f"""กรุณาวิเคราะห์ว่าข้อมูลต่อไปนี้เกี่ยวข้องกับคำถามหรือไม่

คำถาม: {query}

ข้อมูล: {ctx['text']}

ตอบเฉพาะ "YES" ถ้าข้อมูลนี้เกี่ยวข้องกับคำถาม หรือ "NO" ถ้าไม่เกี่ยวข้อง
จากนั้นถ้าตอบ YES ให้สกัดเฉพาะส่วนที่เกี่ยวข้องออกมา (สรุปสั้นๆ ภายใน 2-3 ประโยค)

รูปแบบคำตอบ:
[YES/NO]
[ข้อความที่สกัดออกมา (ถ้า YES)]"""

            try:
                response = self.client.chat.completions.create(
                    model="typhoon-v2.5-30b-a3b-instruct",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that analyzes text relevance. Answer in Thai."},
                        {"role": "user", "content": compression_prompt}
                    ],
                    max_tokens=256,
                    temperature=0.3
                )
                response_text = response.choices[0].message.content.strip()

                # Check if context is relevant
                if response_text.startswith("YES"):
                    # Extract the compressed text
                    lines = response_text.split('\n', 1)
                    compressed_text = lines[1].strip() if len(lines) > 1 else ctx['text']

                    # Keep the context with compressed text
                    compressed_ctx = ctx.copy()
                    compressed_ctx['text'] = compressed_text
                    compressed_ctx['original_text'] = ctx['text']
                    compressed_ctx['compressed'] = True
                    compressed_contexts.append(compressed_ctx)

            except Exception as e:
                # If compression fails, keep original
                print(f"⚠️ Compression failed for a context: {e}")
                compressed_contexts.append(ctx)

        return compressed_contexts if compressed_contexts else contexts[:2]

    def create_prompt(self, user_question: str) -> str:
        """Create a comprehensive prompt with context"""
        # Expand query for better search
        expanded_query = self.expand_query(user_question)

        # Search for relevant knowledge with more results
        relevant_knowledge = self.knowledge_base.search_knowledge(expanded_query, n_results=10)

        # Apply context compression if enabled (ปิดไว้เพื่อความรวดเร็ว)
        # if self.use_compression:
        #     relevant_knowledge = self.compress_context(user_question, relevant_knowledge)

        context = self.knowledge_base.get_context_string(relevant_knowledge)

        prompt = f"""คุณคือ "Askgiraffe" ผู้ช่วยให้คำปรึกษาเกี่ยวกับหลักสูตรและการรับสมัครนักศึกษา
ของคณะครุศาสตร์อุตสาหกรรม มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (มจพ.)

**กฎการตอบคำถาม (STRICT RAG - ต้องปฏิบัติตาม):**
1. ✅ ตอบได้เฉพาะข้อมูลที่มีใน "ข้อมูลอ้างอิง" ด้านล่างเท่านั้น
2. ✅ อ่านข้อมูลอ้างอิงทั้งหมดอย่างละเอียด ก่อนตอบคำถาม
3. ✅ รวบรวมข้อมูลที่เกี่ยวข้องจากหลายๆ ข้อมูลอ้างอิงมาตอบให้ครบถ้วน
4. ❌ ห้ามเดาหรือสร้างข้อมูลที่ไม่มีในฐานความรู้
5. ❌ ห้ามใช้ความรู้ทั่วไปหรือข้อมูลภายนอกมาตอบ
6. ✅ ถ้าไม่มีข้อมูลหรือข้อมูลไม่เพียงพอ ให้บอกตรงๆ ว่า "ขออภัยค่ะ ข้อมูลนี้ไม่มีในระบบ" แล้วแนะนำช่องทางติดต่อ

**บุคลิกภาพและสไตล์การตอบ:**
- ตอบด้วยภาษาที่เป็นกันเอง เหมือนพี่ให้คำแนะนำน้อง
- ใช้คำว่า "ครับ/ค่ะ" ตามความเหมาะสม
- อธิบายให้เข้าใจง่าย ตรงประเด็น กระชับ
- ถ้ามีข้อมูลหลายส่วนที่เกี่ยวข้อง ให้รวบรวมมาตอบให้ครบ

**ข้อมูลอ้างอิง (ตอบได้เฉพาะจากข้อมูลนี้เท่านั้น):**
{context if context else "(ไม่พบข้อมูลที่เกี่ยวข้องในฐานความรู้)"}

**คำถามจากผู้ใช้:** {user_question}

**การตรวจสอบก่อนตอบ:**
- อ่านข้อมูลอ้างอิงทั้งหมด 10 ข้อ อย่างละเอียด
- หาข้อมูลที่ตอบคำถามได้ทั้งหมด แล้วรวบรวมมาตอบให้ครบถ้วน
- ถ้าไม่มีข้อมูล ให้บอกตรงๆ และแนะนำช่องทางติดต่อ

**ช่องทางติดต่อสำหรับข้อมูลที่ไม่มีในระบบ:**
- เว็บไซต์: admission.kmutnb.ac.th
- โทร: 02-555-2000 (สำนักงานคณะ)
- Facebook: คณะครุศาสตร์อุตสาหกรรม มจพ.

**คำตอบของคุณ (ต้องอิงจากข้อมูลอ้างอิงเท่านั้น):**"""

        return prompt

    def chat(self, user_input: str, max_history_turns: int = 3) -> str:
        """
        Process user input and return chatbot response with conversation memory

        Args:
            user_input: User's question
            max_history_turns: Maximum number of previous conversation turns to include (default: 3)
        """
        try:
            # Create prompt with knowledge base context
            prompt = self.create_prompt(user_input)

            # Build messages with conversation history
            messages = [
                {"role": "system", "content": "You are a helpful assistant. You must answer only in Thai."}
            ]

            # Add recent conversation history (limited to max_history_turns)
            recent_history = self.conversation_history[-max_history_turns:] if self.conversation_history else []
            for conv in recent_history:
                messages.append({"role": "user", "content": conv["user"]})
                messages.append({"role": "assistant", "content": conv["bot"]})

            # Add current question with RAG context
            messages.append({"role": "user", "content": prompt})

            # Generate response using Typhoon
            response = self.client.chat.completions.create(
                model="typhoon-v2.1-12b-instruct",
                messages=messages,
                max_tokens=16000,
                temperature=0.6
            )

            if response.choices[0].message.content:
                # Store conversation history
                self.conversation_history.append({
                    "user": user_input,
                    "bot": response.choices[0].message.content
                })
                return response.choices[0].message.content
            else:
                return "ขออภัย ไม่สามารถสร้างคำตอบได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"

        except Exception as e:
            print(f"เกิดข้อผิดพลาด: {e}")
            return "ขออภัย เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง"

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


def main():
    """Main function to run the chatbot"""

    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = os.getenv('TYPHOON_API_KEY')
    if not api_key:
        print("❌ ไม่พบ TYPHOON_API_KEY ใน environment variables")
        return

    try:
        # Initialize vector knowledge base and chatbot
        print("=" * 80)
        print("🔧 Initializing chatbot with KEYWORD-ENHANCED hybrid retrieval...")
        print("=" * 80)
        kb = HybridKnowledgeBase(
            persist_directory="./chroma_db",
            collection_name="chatbot_knowledge",
            use_keyword_boost=True  # เปิดใช้งาน keyword boosting
        )
        chatbot = TyphoonChatbot(api_key, kb)

        print("=" * 80)
        print("🤖 ยินดีต้อนรับสู่ระบบแชทบอทคณะครุศาสตร์อุตสาหกรรม (v4 - Keyword Enhanced)")
        print("    มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ")
        print("=" * 80)
        print("📚 ระบบ: Hybrid Retrieval + Keyword Matching")
        print("🔍 เทคโนโลยี: BM25 (keyword) + Vector (semantic) + Keyword Boost + Cross-Encoder")
        print("🌪️  AI Model: Typhoon v2.1-12b-instruct")
        print("-" * 80)
        print("📝 คุณสามารถถามคำถามเกี่ยวกับหลักสูตรวิศวกรรมโยธาและการศึกษาได้")
        print("💡 พิมพ์ 'quit', 'exit', หรือ 'ออก' เพื่อปิดโปรแกรม")
        print("🔄 พิมพ์ 'clear' เพื่อล้างประวัติการสนทนา")
        print("📜 พิมพ์ 'history' เพื่อดูประวัติการสนทนา")
        print("-" * 80)

        while True:
            # Get user input
            user_input = input("\n🙋 คุณ: ").strip()

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'ออก', '']:
                print("\n👋 ขอบคุณที่ใช้บริการ! สวัสดีครับ/ค่ะ")
                break

            # Check for clear command
            if user_input.lower() in ['clear', 'ล้าง']:
                chatbot.clear_history()
                print("\n🧹 ล้างประวัติการสนทนาเรียบร้อยแล้ว")
                continue

            # Check for history command
            if user_input.lower() in ['history', 'ประวัติ']:
                history = chatbot.get_conversation_history()
                if history:
                    print("\n📜 ประวัติการสนทนา:")
                    print("-" * 40)
                    for i, conv in enumerate(history, 1):
                        print(f"[{i}] คุณ: {conv['user']}")
                        print(f"[{i}] บอท: {conv['bot'][:100]}...")
                        print("-" * 40)
                else:
                    print("\n📭 ยังไม่มีประวัติการสนทนา")
                continue

            # Process the question
            print("\n🤔 กำลังคิด...")
            response = chatbot.chat(user_input)
            print(f"\n🤖 บอท: {response}")

    except KeyboardInterrupt:
        print("\n\n👋 ขอบคุณที่ใช้บริการ! สวัสดีครับ/ค่ะ")
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")


if __name__ == "__main__":
    main()
