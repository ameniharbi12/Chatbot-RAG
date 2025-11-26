# prompt_engineering.py
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# -------------------------------
# Initialize Gemini client
client = genai.Client(api_key=GENAI_API_KEY)

# -------------------------------
# Function to query LLM
def query_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 300) -> str:
    """
    Send a prompt to Gemini LLM and return the generated text.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    )
    return response.text

# -------------------------------
# Example question
question = "Comment ma fille peut améliorer son anglais cet été ?"

# -------------------------------
# Define prompts using different strategies
prompts = {
    "zero_shot": f"Réponds à cette question de manière claire : {question}",

    "one_shot": f"Réponds à cette question de manière concise : {question}",

    "few_shot": f"""Voici des exemples de réponses aux questions similaires :
1. Q: Comment améliorer son français rapidement ? A: Lire des livres, écouter des podcasts, pratiquer chaque jour.
2. Q: Comment apprendre à coder efficacement ? A: Faire des projets pratiques, suivre des tutoriels, pratiquer régulièrement.

Maintenant réponds à cette question : {question}""",

    "few_shot_with_explanations": f"""Exemples avec raisonnement :
1. Q: Comment améliorer son français rapidement ? A: Lire des livres (cela augmente le vocabulaire), écouter des podcasts (améliore l'écoute), pratiquer chaque jour (renforce l'apprentissage).
2. Q: Comment apprendre à coder efficacement ? A: Faire des projets pratiques (applique les concepts), suivre des tutoriels (apprentissage guidé), pratiquer régulièrement (perfectionnement).

Maintenant explique et réponds à cette question : {question}""",

    "chain_of_thought": f"""Réfléchis étape par étape avant de répondre à la question suivante.
Question: {question}
Étapes:""",

    "chain_of_thought_with_example": f"""Exemple de raisonnement pas-à-pas pour une question similaire :
Q: Comment améliorer son français rapidement ?
Étapes:
1. Identifier les compétences à améliorer : lecture, écoute, expression.
2. Trouver des ressources adaptées : livres, podcasts, vidéos.
3. Pratiquer quotidiennement et suivre les progrès.
Maintenant, fais pareil pour cette question : {question}""",

    "instructional_role": f"Tu es un professeur d'anglais expérimenté. Donne des conseils détaillés à un parent pour que sa fille améliore son anglais cet été.\nQuestion: {question}",

    "creative": f"Réponds de manière créative et engageante à cette question pour motiver un enfant : {question}",

    "self_critique": f"""Réponds à la question suivante et critique ensuite ta propre réponse pour la rendre plus précise et complète.
Question: {question}""",

    "step_by_step": f"Résous ce problème étape par étape et explique chaque étape avant de donner la réponse finale.\nQuestion: {question}"
}

# -------------------------------
# Test all prompt strategies
if __name__ == "__main__":
    for strategy, prompt in prompts.items():
        print(f"\n=== Strategy: {strategy} ===")
        print("Prompt:\n", prompt)
        print("\nResponse:")
        try:
            output = query_llm(prompt)
            print(output)
        except Exception as e:
            print("Error:", e)
