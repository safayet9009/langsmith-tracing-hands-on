import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# .env ফাইল থেকে HUGGINGFACEHUB_API_TOKEN লোড করা
load_dotenv()

# ১. প্রম্পট টেমপ্লেট ডিফাইন করা
prompt = PromptTemplate.from_template("{question}")

# ২. LLM মডেল কনফিগারেশন (Hugging Face Router & Qwen 2.5 Coder)
model = ChatOpenAI(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    base_url="https://router.huggingface.co/v1",
    temperature=0.7,
)

# ৩. আউটপুট পার্সার (সরাসরি স্ট্রাকচার্ড টেক্সট স্ট্রিং ফেরত পাওয়ার জন্য)
parser = StrOutputParser()

# ৪. LCEL Chain তৈরি (prompt -> model -> parser)
chain = prompt | model | parser

# ৫. চেইন ইনভোক করা এবং ফলাফল প্রিন্ট করা
result = chain.invoke({"question": "What is the capital of Peru?"})
print(result)