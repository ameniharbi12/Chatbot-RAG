# generator.py
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from retriever import retrieve_relevant_chunks
from embedder import embed_chunk

load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
client = genai.Client(api_key=GENAI_API_KEY)

def generate_answer(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=300
        )
    )
    return response.text

def answer_question(question: str, top_k: int = 5) -> str:
    chunks = retrieve_relevant_chunks(question, top_k)
    context = "\n".join(chunks)
    prompt = f"Réponds à la question en utilisant le contexte suivant:\n{context}\nQuestion: {question}"
    return generate_answer(prompt)

# -------------------------------
# Example usage
if __name__ == "__main__":
    user_question = "Comment je peut améliorer mon anglais cet été ?"
    answer = answer_question(user_question)
    print("\n=== GEMINI ANSWER ===")
    print(answer)
