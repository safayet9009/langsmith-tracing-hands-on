import os
from dotenv import load_dotenv

# Document Loader, Splitter, VectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Embeddings, LLM and Parsers
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Environment variables load করা
load_dotenv()

# ==========================================
# ১. LangSmith Enabled Configuration
# ==========================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langsmith-rag-demo"

PDF_PATH = "islr.pdf"

# ==========================================
# ২. PDF Load and Document Splitting
# ==========================================
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"'{PDF_PATH}' ফাইলটি পাওয়া যায়নি!")

print("Parsing PDF with PyMuPDF...")
loader = PyMuPDFLoader(PDF_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# ==========================================
# ৩. Cloud Embeddings & VectorStore Setup
# ==========================================
print("Setting up Cloud-based Embeddings...")
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ==========================================
# ৪. Prompt, LLM and Parser Setup
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

# ==========================================
# ৫. Standard RAG Chain Construction
# ==========================================
parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

# ফাইনাল LCEL RAG Chain
chain = parallel | prompt | llm | parser

# ==========================================
# ৬. Interactive Question-Answering Loop
# ==========================================
print("\nPDF RAG System Ready. Type your question (or Ctrl+C to exit).")

try:
    while True:
        q = input("\nQ: ")
        if not q.strip():
            continue
        
        # স্বয়ংক্রিয়ভাবে এটি LangSmith-এ ট্রেস পাঠাবে
        ans = chain.invoke(q.strip())
        print("\nA:", ans)

except KeyboardInterrupt:
    print("\nExiting RAG System.")