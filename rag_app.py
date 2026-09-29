import os
import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Nigeria Healthcare RAG",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .main {
        padding-top: 1rem;
    }

    /* Header */
    .app-header {
        padding: 1rem 0 1.5rem 0;
    }

    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }

    .app-subtitle {
        color: #64748b;
        font-size: 1rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
    }

    /* IMPORTANT:
       Do NOT style .stChatInput directly.
       Streamlit controls the chat input positioning.
    */

    /* Keep the chat area from being hidden behind the input */
    [data-testid="stChatMessageContainer"] {
        padding-bottom: 7rem;
    }

    /* Improve chat input visibility */
    [data-testid="stChatInput"] {
        background-color: white;
    }

    /* Source boxes */
    .source-box {
        background-color: #f8fafc;
        border-left: 4px solid #16a34a;
        padding: 0.7rem 1rem;
        margin-top: 0.5rem;
        border-radius: 5px;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
embedding_api_key = os.getenv("EMBEDDING_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is missing from your .env file.")
    st.stop()

if not embedding_api_key:
    st.error("EMBEDDING_API_KEY is missing from your .env file.")
    st.stop()


# ============================================================
# PROVIDER CONFIG
# ============================================================

CHAT_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL_NAME = "openai/gpt-oss-20b"

EMBEDDING_BASE_URL = "https://qwen-embed.publicaai.com/v1"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

CHROMA_PATH = "Countrybrief-JAB_store"


# ============================================================
# INITIALIZE EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        api_key=embedding_api_key,
        base_url=EMBEDDING_BASE_URL,
    )


# ============================================================
# INITIALIZE VECTOR DATABASE
# ============================================================

@st.cache_resource
def load_vectorstore():

    embeddings = load_embeddings()

    vectorstore = Chroma(
        collection_name="countrybrief",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    return vectorstore


# ============================================================
# INITIALIZE RETRIEVER
# ============================================================

@st.cache_resource
def load_retriever():

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 10,
        },
    )

    return retriever


# ============================================================
# INITIALIZE CHAT MODEL
# ============================================================

@st.cache_resource
def load_chat_model():

    return ChatOpenAI(
        api_key=api_key,
        base_url=CHAT_BASE_URL,
        model=CHAT_MODEL_NAME,
        temperature=0,
    )


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a knowledgeable health consultant providing insights
about the healthcare system in Nigeria.

Answer the user's question using ONLY the provided context.

The context comes from a country brief covering topics such as:

- Country overview
- Selected health indicators
- Developments in healthcare prior to political independence in 1960
- Characteristics of healthcare services during the colonial period
- Legal beginnings of the healthcare system after independence
- Description of the current healthcare system from 2016–2023

Instructions:

1. Give a clear and comprehensive answer.
2. Use information from the supplied context.
3. Do not invent facts that are not supported by the context.
4. If the context does not contain enough information to answer,
   clearly say that the information is not available in the
   retrieved documents.
5. Structure longer answers using headings or bullet points.
6. When appropriate, mention relevant dates and historical periods.

Context:
{context}

Question:
{question}
"""
)


# ============================================================
# CREATE RAG CHAIN
# ============================================================

@st.cache_resource
def load_chain():

    chat_model = load_chat_model()

    return prompt | chat_model | StrOutputParser()


# ============================================================
# RAG FUNCTION
# ============================================================

def ask_rag(question):

    retriever = load_retriever()
    chain = load_chain()

    # Retrieve relevant documents
    documents = retriever.invoke(question)

    # Make sure we actually have context
    if not documents:
        return (
            "I could not find relevant information in the "
            "Country Brief knowledge base.",
            [],
        )

    # Combine document content
    context = "\n\n--- DOCUMENT ---\n\n".join(
        doc.page_content for doc in documents
    )

    # Generate answer
    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return answer, documents


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🇳🇬 Nigeria Healthcare RAG")

    st.markdown(
        """
        This chatbot uses **Retrieval-Augmented Generation (RAG)**
        to answer questions about Nigeria's healthcare system.
        """
    )

    st.divider()

    st.markdown("### Configuration")

    st.write("**Chat model**")
    st.code(CHAT_MODEL_NAME)

    st.write("**Embedding model**")
    st.code(EMBEDDING_MODEL_NAME)

    st.write("**Retriever**")
    st.code("MMR")

    st.write("**Retrieved documents**")
    st.code("5")

    st.divider()

    st.markdown("### Topics")

    st.markdown(
        """
        - 🇳🇬 Country overview
        - 📊 Health indicators
        - 🏥 Healthcare history
        - 🏛️ Colonial healthcare
        - ⚖️ Healthcare legislation
        - 📈 Healthcare system 2016–2023
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 👋 I'm your Nigeria Healthcare Assistant.\n\n"
                    "I can answer questions about Nigeria's healthcare "
                    "system using the Country Brief knowledge base."
                ),
            }
        ]

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">

        <div class="app-title">
            🇳🇬 Nigeria Healthcare Assistant
        </div>

        <div class="app-subtitle">
            Ask questions about Nigeria's healthcare system
            using your Country Brief knowledge base.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INITIALIZE CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm your Nigeria Healthcare Assistant.\n\n"
                "I can answer questions about Nigeria's healthcare "
                "system using the information in the Country Brief "
                "knowledge base.\n\n"
                "Try asking:\n\n"
                "- What is the healthcare system like in Nigeria?\n"
                "- How did healthcare develop after independence?\n"
                "- What were the characteristics of colonial healthcare?\n"
                "- What health indicators are discussed in the document?"
            ),
        }
    ]


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display sources
        if message["role"] == "assistant" and message.get("sources"):

            sources = message["sources"]

            with st.expander(
                f"📚 Retrieved sources ({len(sources)})"
            ):

                for i, source in enumerate(sources, 1):

                    metadata = source.metadata or {}

                    page = metadata.get("page", "N/A")

                    content = source.page_content or ""

                    if len(content) > 500:
                        content = content[:500] + "..."

                    st.markdown(
                        f"**Source {i} — Page {page}**"
                    )

                    st.caption(content)


# ============================================================
# CHAT INPUT
# ============================================================

# IMPORTANT:
# st.chat_input() should be at the top level of the Streamlit app.
# Do not place it inside a sidebar, column, form, or container.

question = st.chat_input(
    "Ask a question about Nigeria's healthcare system..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "🔎 Searching the knowledge base..."
            ):

                answer, documents = ask_rag(question)

            # Display answer
            st.markdown(answer)

            # Display sources
            if documents:

                with st.expander(
                    f"📚 Retrieved sources ({len(documents)})"
                ):

                    for i, doc in enumerate(documents, 1):

                        metadata = doc.metadata or {}

                        page = metadata.get(
                            "page",
                            "N/A",
                        )

                        content = doc.page_content or ""

                        if len(content) > 500:
                            content = content[:500] + "..."

                        st.markdown(
                            f"**Source {i} — Page {page}**"
                        )

                        st.caption(content)

            # Save assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": documents,
                }
            )

        except Exception as e:

            error_message = (
                "Sorry, I encountered an error while "
                "processing your question."
            )

            st.error(error_message)

            st.exception(e)
