import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ==========================================
# ১. LangSmith Config & Environment
# ==========================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langsmith-sequential-chain-demo"

# Force synchronous trace submission (preserves all steps)
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false" 

# ==========================================
# ২. Prompt Templates & Output Parser
# ==========================================
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
).with_config({"run_name": "ReportPromptBuilder"})

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
).with_config({"run_name": "SummaryPromptBuilder"})

parser = StrOutputParser().with_config({"run_name": "TextOutputParser"})

# ==========================================
# ৩. LLM Model Setup
# ==========================================
model = ChatOpenAI(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    base_url="https://router.huggingface.co/v1",
    temperature=0.7,
).with_config({"run_name": "Qwen32B_LLM"})

# ==========================================
# ৪. Sub-Chains for Explicit Tracing
# ==========================================
# Step 1 Sub-chain
step1_chain = (prompt1 | model | parser).with_config({"run_name": "Step1_Report_Generator"})

# Step 2 Sub-chain (Map Step 1 output string to 'text' key expected by prompt2)
step2_chain = (
    (lambda text_input: {"text": text_input}) 
    | prompt2 
    | model 
    | parser
).with_config({"run_name": "Step2_Summary_Generator"})

# Master Sequential Chain
chain = (step1_chain | step2_chain).with_config({"run_name": "Sequential_Master_Chain"})

# ==========================================
# ৫. Execution
# ==========================================
config = {
    "tags": ["masterclass", "qwen-2.5", "detailed-tracing"],
    "metadata": {
        "user_id": "safayet",
        "environment": "development"
    }
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)