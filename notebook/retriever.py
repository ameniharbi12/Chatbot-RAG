# retriever.py
import psycopg
from psycopg.rows import dict_row
from embedder import embed_chunk  # make sure embed_chunk is imported

DB_CONNECTION_STR = "dbname='rag_chatbot' user='postgres' password='' host='localhost' port='5432'"

def retrieve_relevant_chunks(question: str, top_k: int = 5):
    """
    Retrieve top-k chunks from PostgreSQL embeddings using pgvector.
    """
    question_emb = embed_chunk(question)
    with psycopg.connect(DB_CONNECTION_STR) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT corpus 
                FROM embeddings
                ORDER BY embedding <#> %s::vector
                LIMIT %s
            """, (question_emb, top_k))
            return [row["corpus"] for row in cur.fetchall()]
