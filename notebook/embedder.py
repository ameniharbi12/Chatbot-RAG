# embedder.py
import os
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# Config
DATA_FOLDER = os.getenv("DATA_FOLDER", "C:\\Chatbot-RAG\\data")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en")
MAX_TOKENS = 512
STRIDE = 256  # overlap between chunks

# -------------------------------
# Load tokenizer and model
print("Loading embedding model:", EMBEDDING_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model.eval()

# -------------------------------
# File reading
def read_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except:
        return ""

def read_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except:
        return ""

def read_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except:
        return ""

def read_file(file_path: str) -> str:
    ext = file_path.lower().split(".")[-1]
    if ext == "txt":
        return read_txt(file_path)
    elif ext == "pdf":
        return read_pdf(file_path)
    elif ext in ("docx", "doc"):
        return read_docx(file_path)
    elif ext == "xml":
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except:
            return ""
    else:
        return ""

# -------------------------------
# Chunking
def chunk_text(text: str, max_length: int = MAX_TOKENS, stride: int = STRIDE) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i+max_length]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        if i + max_length >= len(tokens):
            break
    return chunks

# -------------------------------
# Embedding
def embed_chunk(chunk: str) -> list[float]:
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy().tolist()
    return embedding

# -------------------------------
# DB helper
def save_embedding(corpus: str, embedding: list[float], cursor) -> None:
    cursor.execute(
        "INSERT INTO embeddings (corpus, embedding) VALUES (%s, %s)",
        (corpus, embedding)
    )
