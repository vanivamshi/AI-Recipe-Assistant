# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1HeO1deH7uWefN-Zq3K_mL7AJhrqa0YrR

Download Kaggle dataset

Code:
AI Recipe Assistant using RAG (Retrieval-Augmented Generation)
- Dataset: 64k recipes from Kaggle
- Embeddings: Google Gemini
- Vector Store: FAISS
- LLM: Google Gemini
- UI: Streamlit
"""

# ── 1. DOWNLOAD DATASET ────────────────────────────────────────────────────────
import kagglehub
path = kagglehub.dataset_download("prashantsingh001/recipes-dataset-64k-dishes")
print("Dataset downloaded to:", path)

# ── 2. INSTALL DEPENDENCIES ────────────────────────────────────────────────────
!pip install -q langchain langchain-community langchain-text-splitters
!pip install -q langchain-google-genai langchain-core
!pip install -q faiss-cpu pandas streamlit pyngrok

# ── 3. LOAD DATASET ────────────────────────────────────────────────────────────
import pandas as pd
df = pd.read_csv('/content/1_Recipe_csv.csv')
print(f"Loaded {len(df)} recipes")
print("Columns:", df.columns.tolist())
df.head()

# ── 4. CONVERT RECIPES TO DOCUMENTS ───────────────────────────────────────────
# Each recipe is wrapped in a LangChain Document for processing
from langchain_core.documents import Document

documents = []
for _, row in df.iterrows():
    text = f"""Recipe Name: {row['recipe_title']}
Ingredients: {row['ingredients']}
Instructions: {row['directions']}"""
    documents.append(Document(page_content=text))

print(f"Created {len(documents)} documents")

# ── 5. SPLIT DOCUMENTS INTO CHUNKS ─────────────────────────────────────────────
# Large recipes are split into smaller overlapping chunks for better retrieval
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,    # max characters per chunk
    chunk_overlap=200   # overlap to preserve context between chunks
)
docs = splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks")

# ── 6. CREATE EMBEDDINGS ───────────────────────────────────────────────────────
# Convert text chunks into vector representations using Gemini embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

GEMINI_API_KEY = "your-gemini-api-key"  # 🔑 Replace with your key from aistudio.google.com

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2-preview",
    google_api_key=GEMINI_API_KEY
)

# ── 7. BUILD VECTOR STORE ──────────────────────────────────────────────────────
# Store embeddings in FAISS for fast similarity search
# Note: Using 500 docs due to free tier rate limit (100 requests/min)
# For full 64k recipes, upgrade to Gemini paid tier
from langchain_community.vectorstores import FAISS

docs_subset = docs[:500]
vectorstore = FAISS.from_documents(docs_subset, embeddings)
vectorstore.save_local("recipe_index")
print(f"✓ Vector store saved with {len(docs_subset)} recipe chunks")

# ── 8. CHECK AVAILABLE GEMINI MODELS ──────────────────────────────────────────
# Useful to verify which models your API key can access
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)

print("Available models for text generation:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(" -", m.name)

# ── 9. BUILD RAG PIPELINE ──────────────────────────────────────────────────────
# Retriever fetches top 5 most relevant recipe chunks for each query
# LLM then generates a formatted answer based on retrieved context
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   # use gemini-1.5-pro for higher quality responses
    google_api_key=GEMINI_API_KEY,
    temperature=0.3             # lower = more factual, higher = more creative
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful recipe assistant. Use the following recipes to answer the question.
Always include the full ingredients list and complete step-by-step instructions.

Context: {context}

Question: {question}

Format your answer as:
**Recipe Name:** ...
**Ingredients:** (full list)
**Instructions:** (numbered steps)
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── 10. TEST QUERIES ───────────────────────────────────────────────────────────
test_queries = [
    "Give me a vegetarian pasta recipe",
    "What recipes can I make with garlic?",
    "Quick breakfast recipes",
    "Low calorie dinner ideas",
]

response = qa_chain.invoke(test_queries[0])
print(response)

# ── 11. LAUNCH STREAMLIT UI ────────────────────────────────────────────────────
# Runs the recipe assistant as a web app accessible via ngrok tunnel
import subprocess, time
from pyngrok import ngrok

NGROK_DOMAIN = "your-ngrok-domain.ngrok-free.app"  # 🔑 From dashboard.ngrok.com/domains

subprocess.Popen([
    "streamlit", "run", "app.py",
    "--server.port=8501",
    "--server.enableCORS=false",
    "--server.enableXsrfProtection=false"
])
time.sleep(5)

public_url = ngrok.connect(addr=8501, proto="http", domain=NGROK_DOMAIN)
print(f"✓ App live at: https://{NGROK_DOMAIN}")
