import os
from dotenv import load_dotenv

# LangSmith-এর জন্য Key Import
from langsmith import traceable

# হালকা ও দ্রুতগতির Loaders এবং Embeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS

# LLM, Prompts & Parsers
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# .env ফাইল থেকে API Token ও LangSmith Keys লোড করা
load_dotenv()

# ==========================================
# ১. LangSmith Environment Config
# ==========================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "02-traceable-full-pipeline"
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"

PDF_PATH = "islr.pdf"

# ==========================================
# ২. Explicitly Traced Setup Steps (@traceable)
# ==========================================

@traceable(name="load_pdf")
def load_pdf(path: str):
    print("Fast parsing PDF with PyMuPDF...")
    loader = PyMuPDFLoader(path)
    return loader.load()

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    print("Generating Embeddings via Hugging Face Endpoint API...")
    # ভারী Local Transformer-এর বদলে Serverless Cloud Embeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    vs = FAISS.from_documents(splits, embeddings)
    return vs

# পুরো Data Ingestion & Indexing প্রসেসটিকে ১টি Parent Span-এ ট্রেস করার জন্য
@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"'{pdf_path}' ফাইলটি পাওয়া যায়নি! প্রজেক্ট ফোল্ডারে PDF ফাইলটি যোগ করুন।")
    
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs

# ==========================================
# ৩. Setup Execution & Retriever Init
# ==========================================
vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ==========================================
# ৪. LCEL Pipeline Setup (Qwen 2.5 Coder LLM)
# ==========================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    base_url="https://router.huggingface.co/v1",
    temperature=0.2,
)

parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

# Final RAG Chain
chain = parallel | prompt | llm | parser

# ==========================================
# ৫. Interactive Query Execution
# ==========================================
print("\nPDF RAG system ready. Type your question (or Ctrl+C to exit).")

try:
    while True:
        q = input("\nQ: ").strip()
        if not q:
            continue
        
        # LangSmith-এ রানটিকে সহজে চেনার জন্য কাস্টম কনফিগারেশন
        config = {
            "run_name": "pdf_rag_query",
            "tags": ["rag", "pdf-chat", "qwen2.5-coder"],
            "metadata": {"pdf_source": PDF_PATH}
        }
        
        ans = chain.invoke(q, config=config)
        print("\nA:", ans)

except KeyboardInterrupt:
    print("\nExiting RAG system.")