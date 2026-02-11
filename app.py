import streamlit as st
import os
from dotenv import load_dotenv
from chatbot_v04_keywords import HybridKnowledgeBase, TyphoonChatbot
import time
import base64

# Load environment variables
load_dotenv()

# Load background image and encode to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Get base64 encoded images
bg_image = get_base64_image("gb.jpg")
bot_logo = get_base64_image("Askgiraffe.png")
hamburger_icon = get_base64_image("hamburger.svg")

# Page configuration
st.set_page_config(
    page_title="Askgiraffe - ผู้ช่วยหลักสูตรคณะครุศาสตร์อุตสาหกรรม",
    page_icon="Askgiraffe.png",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS - Modern Green & White Theme with Kanit Font
st.markdown(f"""
<style>
    /* Import Kanit Font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&display=swap');

    /* Global Font Setting */
    * {{
        font-family: 'Kanit', sans-serif !important;
    }}

    /* พื้นหลังหน้าหลัก - Image Background */
    .stApp {{
        background-image: url('data:image/jpeg;base64,{bg_image}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.85);
        z-index: -1;
    }}

    /* พื้นหลัง Main content area */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    /* Sidebar Background - Darker Green */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #A7F3D0 0%, #D1FAE5 100%);
    }}

    [data-testid="stSidebar"] > div:first-child {{
        background: linear-gradient(180deg, #A7F3D0 0%, #D1FAE5 100%);
    }}

    /* Enable Sidebar Toggle Button */
    button[kind="header"] {{
        display: block !important;
        color: #047857 !important;
        padding: 0.5rem !important;
        border-radius: 0.5rem !important;
        transition: all 0.3s ease !important;
    }}

    /* Sidebar Toggle Button Styling */
    button[kind="header"]:hover {{
        background-color: rgba(16, 185, 129, 0.2) !important;
        transform: scale(1.1);
    }}

    /* Replace default icon with custom hamburger icon */
    button[kind="header"] svg {{
        display: none !important;
    }}

    button[kind="header"]::before {{
        content: '';
        display: inline-block;
        width: 24px;
        height: 24px;
        background-image: url('data:image/svg+xml;base64,{hamburger_icon}');
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
    }}

    /* Header Styling with Shadow */
    .main-header {{
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(5, 150, 105, 0.1);
        letter-spacing: -0.5px;
    }}

    .sub-header {{
        font-size: 1.3rem;
        font-weight: 600;
        shadow: 1px 1px 3px rgba(4, 120, 87, 0.1);
        color: #047857;
        text-align: center;
        margin-bottom: 2.5rem;
        letter-spacing: 0.3px;
    }}

    /* Chat Message Containers with Modern Shadow */
    .chat-message {{
        padding: 1.75rem;
        border-radius: 1rem;
        margin-bottom: 1.25rem;
        display: flex;
        flex-direction: column;
        color: #1F2937;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }}

    .chat-message:hover {{
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
        transform: translateY(-2px);
    }}

    /* User Message - Clean White Design - Right Aligned */
    .user-message {{
        background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
        border-right: 5px solid #10B981;
        border: 2px solid #D1FAE5;
        color: #1F2937;
        margin-left: 15%;
        align-items: flex-end;
        text-align: right;
    }}

    /* Bot Message - Soft Green Background - Left Aligned */
    .bot-message {{
        background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 100%);
        border-left: 5px solid #059669;
        border: 1px solid #D1FAE5;
        color: #1F2937;
        margin-right: 15%;
        align-items: flex-start;
        text-align: left;
    }}

    /* Message Labels */
    .message-label {{
        font-weight: 600;
        margin-bottom: 0.75rem;
        font-size: 1rem;
        letter-spacing: 0.3px;
    }}

    .user-label {{
        color: #10B981;
    }}

    .bot-label {{
        color: #059669;
    }}

    /* Context Box */
    .context-box {{
        background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%);
        padding: 1.25rem;
        border-radius: 0.75rem;
        border-left: 4px solid #10B981;
        margin-top: 0.75rem;
        font-size: 0.9rem;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.1);
    }}

    /* Stat Cards - Modern with Gradient */
    .stat-card {{
        background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
        padding: 1.25rem;
        border-radius: 1rem;
        border: 2px solid #10B981;
        text-align: center;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.15);
        transition: all 0.3s ease;
    }}

    .stat-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(16, 185, 129, 0.25);
    }}

    .stat-number {{
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    .stat-label {{
        font-size: 0.95rem;
        font-weight: 400;
        color: #047857;
        margin-top: 0.5rem;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: #059669 !important;
        font-weight: 600 !important;
        padding: 0.5rem 0;
        letter-spacing: 0.3px;
    }}

    /* Slider Styling - Modern Green */
    .stSlider > div > div > div > div {{
        background: linear-gradient(90deg, #10B981 0%, #059669 100%) !important;
    }}

    .stSlider > div > div > div {{
        background-color: #D1FAE5 !important;
    }}

    /* Button Styling - Modern with Gradient */
    .stButton > button {{
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.85rem 1.75rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3) !important;
        letter-spacing: 0.5px;
    }}

    /* Ensure button text is white */
    .stButton > button * {{
        color: #FFFFFF !important;
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(5, 150, 105, 0.4) !important;
    }}

    .stButton > button:active {{
        transform: translateY(-1px);
    }}

    /* Checkbox Styling */
    .stCheckbox {{
        padding: 0.75rem 0;
    }}

    .stCheckbox > label {{
        font-weight: 500 !important;
        font-size: 1rem !important;
        color: #047857 !important;
    }}

    /* Info/Success/Warning boxes */
    .stAlert {{
        background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 100%) !important;
        border: 2px solid #10B981 !important;
        border-radius: 0.75rem !important;
        color: #047857 !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15) !important;
    }}

    /* Divider Styling */
    [data-testid="stSidebar"] hr {{
        border-color: #D1FAE5 !important;
        border-width: 2px !important;
        margin: 1.5rem 0 !important;
        opacity: 0.6;
    }}

    /* Section Headers with Modern Background */
    [data-testid="stSidebar"] h3 {{
        background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%);
        padding: 1rem 1.25rem !important;
        border-radius: 0.75rem;
        margin-bottom: 1.25rem !important;
        border-left: 5px solid #10B981;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.1);
    }}

    /* Expander Styling - Modern Design */
    .streamlit-expanderHeader {{
        background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%) !important;
        border-radius: 0.75rem !important;
        border: 2px solid #D1FAE5 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        color: #047857 !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
    }}

    .streamlit-expanderHeader:hover {{
        background: linear-gradient(135deg, #D1FAE5 0%, #ECFDF5 100%) !important;
        border-color: #10B981 !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15) !important;
    }}

    /* Chat Input Styling - Modern with Shadow */
    .stChatInput > div {{
        border: 3px solid #10B981 !important;
        border-radius: 1rem !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2) !important;
        transition: all 0.3s ease !important;
    }}

    .stChatInput > div:focus-within {{
        border-color: #059669 !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3) !important;
        transform: translateY(-2px);
    }}

    .stChatInput input {{
        font-size: 1.05rem !important;
        font-weight: 400 !important;
    }}

    /* Markdown text in sidebar */
    [data-testid="stSidebar"] p {{
        color: #1F2937;
        line-height: 1.8;
        font-weight: 400;
    }}

    [data-testid="stSidebar"] strong {{
        color: #047857;
        font-weight: 600;
    }}

    /* Help text */
    .stSlider [data-testid="stMarkdownContainer"] p {{
        font-size: 0.9rem;
        color: #6B7280;
        font-weight: 400;
    }}

    /* Main content h3 styling */
    .main h3 {{
        color: #059669 !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
        margin-bottom: 1rem !important;
        letter-spacing: 0.3px;
    }}

    /* Scrollbar Styling */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}

    ::-webkit-scrollbar-track {{
        background: #F0FDF4;
        border-radius: 10px;
    }}

    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        border-radius: 10px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }}

    /* Welcome Mode */
    .welcome-stage {{
        min-height: 48vh;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }}

    .welcome-card {{
        width: 100%;
        max-width: 820px;
        padding: 2.25rem 2rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 253, 244, 0.95) 100%);
        border: 2px solid #A7F3D0;
        box-shadow: 0 10px 26px rgba(5, 150, 105, 0.16);
        text-align: center;
    }}

    .welcome-title {{
        color: #047857;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        line-height: 1.35;
    }}

    .welcome-subtitle {{
        color: #1F2937;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.8;
        margin: 0;
    }}

    .welcome-input-caption {{
        text-align: center;
        color: #047857;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 0.65rem;
    }}

    @media (max-width: 768px) {{
        .welcome-stage {{
            min-height: 40vh;
            margin-top: 1rem;
        }}

        .welcome-card {{
            padding: 1.5rem 1.25rem;
            border-radius: 1rem;
        }}

        .welcome-title {{
            font-size: 1.5rem;
        }}

        .welcome-subtitle {{
            font-size: 0.98rem;
            line-height: 1.7;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def init_chatbot():
    """Initialize the chatbot (cached to avoid reloading)"""
    api_key = os.getenv('TYPHOON_API_KEY')
    if not api_key:
        st.error("❌ ไม่พบ TYPHOON_API_KEY ใน environment variables")
        st.stop()

    with st.spinner('🔧 กำลังโหลดระบบ Chatbot...'):
        kb = HybridKnowledgeBase(
            persist_directory="./chroma_db",
            collection_name="chatbot_knowledge",
            use_reranker=True,
            use_keyword_boost=True
        )
        chatbot = TyphoonChatbot(api_key, kb, use_compression=False)

    return chatbot

# Initialize chatbot
chatbot = init_chatbot()

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'show_context' not in st.session_state:
    st.session_state.show_context = False

if 'pending_user_input' not in st.session_state:
    st.session_state.pending_user_input = None


def process_user_input(user_input: str):
    """Store user message immediately, then trigger response generation on next rerun."""
    if not user_input:
        return

    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Defer model generation to next rerun so user question appears instantly
    st.session_state.pending_user_input = user_input
    st.rerun()


def process_pending_response():
    """Generate bot response for a queued user message."""
    pending_input = st.session_state.pending_user_input
    if not pending_input:
        return

    # Generate response
    with st.spinner('🤔 กำลังคิดคำตอบ...'):
        response = chatbot.chat(pending_input)
        response_sources = chatbot.get_last_response_sources()

    # Add bot response to chat
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": response_sources,
    })

    st.session_state.pending_user_input = None
    st.rerun()

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🤖 Askgiraffe")
    st.markdown('<p style="color: #047857; font-size: 0.95rem;">ผู้ช่วยให้คำปรึกษาหลักสูตร<br>คณะครุศาสตร์อุตสาหกรรม มจพ.</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Clear history button
    if st.button("🗑️ ล้างประวัติการสนทนา", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_user_input = None
        chatbot.clear_history()
        st.rerun()

    st.session_state.show_context = st.checkbox(
        "📚 แสดงข้อมูลอ้างอิง",
        value=st.session_state.show_context,
        help="แสดงแหล่งข้อมูลที่ใช้ตอบจริง (vector หรือ QA fallback)",
    )

    st.markdown("---")

    # Contact information
    st.markdown("### 📞 ติดต่อสอบถาม")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%); padding: 1.25rem; border-radius: 0.75rem; border: 2px solid #10B981; box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15);">
        <span style="color: #1F2937; line-height: 2;">
        📧 <a href="http://admission.kmutnb.ac.th" style="color: #059669; text-decoration: none; font-weight: 500;">admission.kmutnb.ac.th</a><br>
        ☎️ <span style="font-weight: 500;">02-555-2000</span><br>
        📘 <span style="font-weight: 500;">คณะครุศาสตร์อุตสาหกรรม</span>
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Footer - Copyright
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.85rem; padding: 1.5rem 0.5rem; border-top: 2px solid #D1FAE5; margin-top: 2rem;">
        <strong style="color: #047857; font-size: 0.9rem;">พัฒนาโดย</strong><br>
        <span style="color: #1F2937; line-height: 1.8;">คณะครุศาสตร์อุตสาหกรรม<br>มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ</span><br><br>
        <span style="font-size: 0.8rem; color: #9CA3AF;">© 2025 KMUTNB. All rights reserved.</span>
    </div>
    """, unsafe_allow_html=True)

is_welcome_mode = len(st.session_state.messages) == 0
is_waiting_response = st.session_state.pending_user_input is not None

if is_welcome_mode:
    st.markdown(
        """
        <div class="welcome-stage">
            <div class="welcome-card">
                <div class="welcome-title">สวัสดีครับ 👋 ยินดีต้อนรับสู่ Askgiraffe</div>
                <p class="welcome-subtitle">
                    ผมพร้อมช่วยตอบคำถามเกี่ยวกับหลักสูตรคณะครุศาสตร์อุตสาหกรรม มจพ.<br>
                    พิมพ์คำถามของคุณได้เลย แล้วผมจะช่วยค้นหาคำตอบให้อย่างรวดเร็ว
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, center_col, _ = st.columns([1, 2.5, 1])
    with center_col:
        st.markdown('<div class="welcome-input-caption">เริ่มถามคำถามแรกของคุณได้ที่นี่</div>', unsafe_allow_html=True)
        welcome_input = st.chat_input(
            "✨ พิมพ์คำถามของคุณที่นี่... (กด Enter เพื่อส่ง)",
            key="chat_input_welcome",
            disabled=is_waiting_response,
        )
    process_user_input(welcome_input)
else:
    # Header
    st.markdown(f'<div class="main-header"><img src="data:image/png;base64,{bot_logo}" style="width: 200px; height: 200px; vertical-align: middle; margin-right: 15px;"> Askgiraffe</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ผู้ช่วยให้คำปรึกษาหลักสูตร คณะครุศาสตร์อุตสาหกรรม มจพ.</div>', unsafe_allow_html=True)

    # Main chat area
    st.markdown('<h3 style="color: #000000; font-weight: 600; font-size: 1.5rem; margin-bottom: 1rem;">💬 พื้นที่แชท</h3>', unsafe_allow_html=True)

    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f'''
            <div class="chat-message user-message">
                <div class="message-label user-label">🙋 คุณ:</div>
                <div>{message["content"]}</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="chat-message bot-message">
                <div class="message-label bot-label"><img src="data:image/png;base64,{bot_logo}" style="width: 28px; height: 28px; vertical-align: middle; margin-right: 8px;"> Askgiraffe:</div>
                <div>{message["content"]}</div>
            </div>
            ''', unsafe_allow_html=True)

            # Show context if enabled and available
            if st.session_state.show_context and message.get("sources"):
                sources = message["sources"]
                with st.expander(f"📚 เอกสารอ้างอิง ({len(sources)} รายการ)", expanded=False):
                    for source in sources:
                        source_type = source.get("source_type", "vector")
                        if source_type == "qa_fallback":
                            st.markdown(
                                f"""**QA_FALLBACK**
                                - หมวดหมู่: {source.get('category', '')}
                                - เอกสาร: {source.get('source_document', '')}
                                - คะแนนรวม: {source.get('combined_score', 0.0):.3f}
                                - คำถาม: {source.get('question', '')}
                                - คำตอบ: {source.get('answer', '')}
                                """
                            )
                        else:
                            score_info = f"🎯 Score: {source.get('score', 0):.3f}"
                            if 'rerank_score' in source:
                                score_info += f" | Rerank: {source.get('rerank_score', 0):.3f}"
                            score_info += f" | Raw Vector: {source.get('raw_vector_similarity', 0):.3f}"

                            st.markdown(
                                f"""**รายการที่ {source.get('rank', 0)}** | {score_info}

                                {source.get('text', '')}
                                """
                            )
                        st.markdown("---")

    # Chat input with enhanced styling
    chat_input = st.chat_input(
        "✨ พิมพ์คำถามของคุณที่นี่... (กด Enter เพื่อส่ง)",
        key="chat_input_bottom",
        disabled=is_waiting_response,
    )
    process_user_input(chat_input)
    process_pending_response()
