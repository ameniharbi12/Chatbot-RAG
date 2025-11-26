# db.py
import os
import sys
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from embedder import DATA_FOLDER, read_file, chunk_text, embed_chunk, save_embedding, tokenizer

load_dotenv()
DB_CONNECTION_STR = os.getenv("DB_CONNECTION_STR")

if not DB_CONNECTION_STR:
    print("Please set DB_CONNECTION_STR in your .env")
    sys.exit(1)

def ensure_table_and_insert_all():
    with psycopg.connect(DB_CONNECTION_STR) as conn:
        conn.autocommit = True
        with conn.cursor(row_factory=dict_row) as cur:
            # Ensure pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("pgvector extension ensured.")

            # Drop and create embeddings table
            cur.execute("DROP TABLE IF EXISTS embeddings;")
            cur.execute("""
                CREATE TABLE embeddings (
                    id SERIAL PRIMARY KEY,
                    corpus TEXT,
                    embedding VECTOR(1024)
                );
            """)
            print("Embeddings table created.")

            # Process all files
            for root, dirs, files in os.walk(DATA_FOLDER):
                for file in files:
                    file_path = os.path.join(root, file)
                    print("Processing:", file_path)

                    text = read_file(file_path)
                    if not text.strip():
                        print("Skipping empty/unreadable file:", file_path)
                        continue

                    # ---------------------------
                    # Split text into safe-size chunks
                    chunks = chunk_text(text, max_length=512, stride=256)
                    print(f"Split into {len(chunks)} chunks")

                    for i, chunk in enumerate(chunks):
                        token_count = len(tokenizer.encode(chunk))
                        print(f"  Chunk {i+1}: {token_count} tokens")
                        embedding = embed_chunk(chunk)
                        save_embedding(chunk, embedding, cur)

                    print(f"Saved {len(chunks)} chunks for {file}")

            print("All files processed successfully.")

if __name__ == "__main__":
    ensure_table_and_insert_all()
