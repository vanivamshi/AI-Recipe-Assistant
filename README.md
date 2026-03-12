# AI-Recipe-Assistant

A RAG (Retrieval-Augmented Generation) app that lets you search and discover recipes using natural language. Built with LangChain and Google Gemini.

---

## What it does

Ask questions like:
- *"Give me a vegetarian pasta recipe"*
- *"What can I make with garlic and chicken?"*
- *"Quick low calorie dinner ideas"*

The app retrieves relevant recipes from a 64k recipe dataset and generates a formatted answer with ingredients and instructions.

---

## How it works (RAG Pipeline)

```
User Query
    ↓
LangChain Retriever  →  searches FAISS vector store
    ↓
Top 5 relevant recipe chunks retrieved
    ↓
LangChain Prompt  →  context + question passed to LLM
    ↓
Gemini LLM  →  generates formatted recipe answer
    ↓
Streamlit UI displays result
```

**LangChain is used for:**
- `Document` — wraps each recipe as a structured document
- `RecursiveCharacterTextSplitter` — splits long recipes into chunks
- `GoogleGenerativeAIEmbeddings` — converts text to vectors
- `FAISS` vector store integration — stores and retrieves embeddings
- `ChatPromptTemplate` — structures the prompt sent to the LLM
- `RunnablePassthrough` + pipe (`|`) — chains retriever → prompt → LLM → output
- `StrOutputParser` — parses the LLM response as plain text

---

## Tech Stack

| Component | Tool |
|---|---|
| Framework | LangChain |
| LLM | Google Gemini (`gemini-1.5-flash`) |
| Embeddings | Google Gemini (`gemini-embedding-2-preview`) |
| Vector Store | FAISS |
| UI | Streamlit |
| Dataset | [64k Recipes - Kaggle](https://www.kaggle.com/datasets/prashantsingh001/recipes-dataset-64k-dishes) |
| Tunnel | ngrok |

---

## Setup

### 1. Clone the repo
```bash
1. Download the file
Download cook_book.py from this repository.
```

### 2. Install dependencies
```bash
pip install langchain langchain-community langchain-core
pip install langchain-text-splitters langchain-google-genai
pip install faiss-cpu streamlit pyngrok pandas
```

### 3. Add your API keys
In `cook_book.py` and the notebook, replace:
```python
GEMINI_API_KEY = "your-gemini-api-key"   # https://aistudio.google.com
NGROK_DOMAIN   = "your-domain.ngrok-free.app"  # https://dashboard.ngrok.com
```

### 4. Run the notebook
Open `langchain_cook_book.ipynb` in Google Colab and run all cells. This will:
- Download the dataset
- Build the vector store (`recipe_index/`)
- Launch the Streamlit app via ngrok

### 5. Or run locally
```bash
streamlit run cook_book.py
```
---
```
- ngrok free account: [dashboard.ngrok.com](https://dashboard.ngrok.com)
