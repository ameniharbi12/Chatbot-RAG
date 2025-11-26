
"""
This script is a complete pipeline for creating a Retrieval-Augmented Generation (RAG) system
using local documents and a large language model (LLM) from Google Gemini. It performs the
following key functions:

1. **Package Imports**
   - Imports necessary packages for NLP, embeddings, database interaction, PDF/DOCX reading, 
     and LLM interaction.
   - Required packages include:
     transformers, torch, psycopg, google-genai, numpy, PyPDF2, python-docx.

2. **Configuration / Variables**
   - `data_folder`: Directory where all documents (PDF, DOCX, TXT, XML) are stored.
   - `db_connection_str`: PostgreSQL connection string (replace with actual credentials).
   - `embedding_model_name`: The pre-trained embedding model used to convert text into vectors.

3. **Initialize Embedding Model**
   - Loads a tokenizer and embedding model from Hugging Face.
   - The embeddings are used to represent document text in vector space for similarity search.

4. **File Reading Helpers**
   - `read_txt`, `read_pdf`, `read_docx`: Read text from TXT, PDF, DOCX files.
   - `read_file`: Detects file extension and reads content accordingly; skips unreadable files.

5. **Embedding & Database Functions**
   - `calculate_embeddings(text)`: Tokenizes and embeds text into a 1024-dimensional vector.
     - Uses truncation to limit input to `max_length=512` tokens.
     - Uses mean pooling over token embeddings.
   - `save_embedding(corpus, embedding, cursor)`: Inserts the text and embedding into the PostgreSQL
     database using the `vector` extension for similarity search.

6. **LLM Interaction**
   - `generate_answer(prompt)`: Uses Google Gemini API to generate an answer to a prompt.
   - API key must be replaced with your own in `client = genai.Client(api_key="YOUR_GEMINI_API_KEY")`.
   - Controls response length with `max_output_tokens` and creativity with `temperature`.

7. **Main Execution**
   - Connects to PostgreSQL, ensures the `vector` extension exists.
   - Creates `embeddings` table (id, corpus, embedding vector).
   - Iterates over all files in `data_folder`:
     - Reads each file, skipping empty or unreadable ones.
     - Calculates embeddings and stores them in the database.
   - Commits all inserts at the end.

8. **Example LLM Query**
   - Demonstrates how to call the `generate_answer` function with a user question.
   - Prints the answer returned by Gemini.

9. **Prompt Engineering Techniques**
   - Several strategies can be applied when querying the LLM to improve output quality:
     1. **Zero-shot prompting**: Directly ask a question with no examples.
     2. **One-shot prompting**: Provide a single example before asking the question.
     3. **Few-shot prompting**: Provide multiple examples of question-answer pairs.
     4. **Few-shot with explanations**: Include reasoning in the examples.
     5. **Chain-of-thought (CoT)**: Ask the model to reason step-by-step before answering.
     6. **Chain-of-thought with examples**: Combine CoT reasoning with few-shot examples.
     7. **Instructional / Role-playing**: Ask the model to take a specific role (e.g., teacher, expert).
     8. **Self-critique / Reflection**: Ask the model to evaluate and improve its own answer.
     9. **Step-by-step reasoning**: Force the model to outline steps explicitly before final answer.
     10. **Creative / Engaging**: Encourage imaginative or story-like responses.
     11. **Compare & Rank**: Ask the model to evaluate multiple options or solutions.
     12. **Hypothetical reasoning**: Pose “what if” scenarios for problem solving.
     13. **Error detection / debugging prompts**: Ask the model to find mistakes in a reasoning process.
     14. **Constraint-based reasoning**: Include limits or rules the answer must follow.
     15. **Multi-turn / progressive prompting**: Break a complex task into multiple sub-prompts.

   - These techniques can be combined to create more accurate, creative, or structured responses.


**Notes / Recommendations:**
- For large documents (like PDFs with thousands of tokens), this script may need **text chunking** to
  avoid token length errors with the embedding model.
- Always replace the placeholder API key with a valid Gemini API key.
- Ensure PostgreSQL has the `vector` extension enabled for embeddings similarity search.
- Can be modularized by separating `embedder`, `database`, and `generator` logic into separate files.
"""


import os
import psycopg
from psycopg.rows import dict_row
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from google import genai
from google.genai import types
from PyPDF2 import PdfReader
import docx

# -------------------------------
# Variables
data_folder = "C:\\Chatbot-RAG\\data"
db_connection_str = "dbname='' user='' password='' host='localhost' port=''"
embedding_model_name = "BAAI/bge-large-en"

# -------------------------------
# Initialize embedding model
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

# -------------------------------
# Helper functions to read files
def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="cp1252") as f:
        return f.read()

def read_pdf(file_path: str) -> str:
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def read_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def read_file(file_path: str) -> str:
    ext = file_path.lower().split(".")[-1]
    if ext == "txt":
        return read_txt(file_path)
    elif ext == "pdf":
        return read_pdf(file_path)
    elif ext in ["docx", "doc"]:
        try:
            return read_docx(file_path)
        except Exception:
            print(f"Skipping unreadable file: {file_path}")
            return ""
    elif ext == "xml":
        with open(file_path, "r", encoding="cp1252") as f:
            return f.read()
    else:
        return ""

# -------------------------------
# Functions for embeddings & DB
def calculate_embeddings(text: str) -> list[float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding[0].cpu().numpy().tolist()

def save_embedding(corpus: str, embedding: list[float], cursor) -> None:
    cursor.execute(
        "INSERT INTO embeddings (corpus, embedding) VALUES (%s, %s)",
        (corpus, embedding)
    )

def generate_answer(prompt: str) -> str:
    # Replace with your actual API key
    client = genai.Client(api_key="YOUR_GEMINI_API_KEY")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=300
        )
    )
    return response.text

# -------------------------------
# Main execution
with psycopg.connect(db_connection_str) as conn:
    conn.autocommit = True
    with conn.cursor(row_factory=dict_row) as cur:
        # Drop old table if exists
        cur.execute("DROP TABLE IF EXISTS embeddings;")
        # Ensure vector extension exists
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Create embeddings table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                corpus TEXT,
                embedding VECTOR(1024)
            );
        """)

        # Process all files in the data folder
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                file_path = os.path.join(root, file)
                text = read_file(file_path)
                if not text.strip():
                    continue
                embedding = calculate_embeddings(text)
                print("Saving:", file, "...")
                print("Embedding length:", len(embedding))
                save_embedding(text, embedding, cur)

        conn.commit()

# -------------------------------
# Example usage
user_question = "Comment je peux ameliorer mon anglais cet été ?"
answer = generate_answer(user_question)
print("\n=== GEMINI ANSWER ===")
print(answer)


# -------------------------------
# Example dictionary of prompt types for testing
prompt_strategies = {
    "zero_shot": "Réponds à cette question de manière claire : Comment ma fille peut améliorer son anglais cet été ?",
    "one_shot": "Q: Comment apprendre le français rapidement ? A: Lire, écouter, pratiquer.\nMaintenant réponds : Comment ma fille peut améliorer son anglais cet été ?",
    "few_shot": "Q: Comment apprendre le français rapidement ? A: Lire, écouter, pratiquer.\nQ: Comment apprendre à coder efficacement ? A: Tutoriels, projets, pratique.\nMaintenant réponds : Comment ma fille peut améliorer son anglais cet été ?",
    "few_shot_with_explanations": "Q: Comment apprendre le français ? A: Lire (vocabulaire), écouter (compréhension), pratiquer (fluency).\nQ: Comment apprendre à coder ? A: Tutoriels (guidé), projets (pratique), pratique (maîtrise).\nRéponds à la question suivante avec explications : Comment ma fille peut améliorer son anglais cet été ?",
    "chain_of_thought": "Réfléchis étape par étape avant de répondre : Comment ma fille peut améliorer son anglais cet été ?",
    "instructional_role": "Tu es un professeur d'anglais. Donne des conseils détaillés à un parent pour améliorer l'anglais de sa fille cet été.",
    "self_critique": "Réponds à la question suivante, puis critique ta réponse pour la rendre plus complète : Comment ma fille peut améliorer son anglais cet été ?",
    "step_by_step": "Résous ce problème étape par étape et explique chaque étape avant de donner la réponse finale : Comment ma fille peut améliorer son anglais cet été ?",
    "creative": "Réponds de manière créative et motivante à : Comment ma fille peut améliorer son anglais cet été ?"
}

for name, prompt in prompt_strategies.items():
     print(f"=== {name} ===")
     response = generate_answer(prompt)
     print(response)