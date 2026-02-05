# ------------------ IMPORTS ------------------
import os
import uuid
import shutil
import io
import pymupdf
from PIL import Image
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from openai import OpenAI


# ------------------ ENV ------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")

client_llm = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key='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',
)


# ------------------ CONSTANTS ------------------
COLLECTION_NAME = "col_1"
QDRANT_URL = "http://localhost:6333"
IMAGE_DIR = "images"


# ------------------ MODEL ------------------
model = SentenceTransformer("clip-ViT-B-32")


# ------------------ DB SETUP ------------------
def initialise_db():
    client = QdrantClient(QDRANT_URL)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=512,
                distance=Distance.COSINE,
            ),
        )
    return client


def reset_storage():
    client = QdrantClient(QDRANT_URL)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    shutil.rmtree(IMAGE_DIR, ignore_errors=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)


# ------------------ INDEXING ------------------
def populate_db(doc_path: str, client: QdrantClient):
    doc = pymupdf.open(doc_path)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # -------- TEXT --------
    for page_no, page in enumerate(doc):
        text = page.get_text().strip()
        if not text:
            continue

        embedding = model.encode(text).tolist()

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "type": "text",
                        "page_no": page_no,
                        "text": text,
                    },
                )
            ],
        )

    # -------- IMAGES --------
    for page_no, page in enumerate(doc):
        for img in page.get_images(full=True):
            xref = img[0]
            base = doc.extract_image(xref)

            image_bytes = base["image"]
            ext = base["ext"]

            filename = f"{uuid.uuid4()}.{ext}"
            filepath = os.path.join(IMAGE_DIR, filename)

            with open(filepath, "wb") as f:
                f.write(image_bytes)

            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            embedding = model.encode(pil_image).tolist()

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "type": "image",
                            "page_no": page_no,
                            "filename": filename,
                        },
                    )
                ],
            )


# ------------------ RETRIEVAL ------------------
def process_query(query: str, top_k: int = 20):
    client = QdrantClient(QDRANT_URL)
    query_embedding = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
    )

    return results.points


# ------------------ RAG LLM ------------------
def generate_answer(query: str, retrieved_points):
    """
    Feeds retrieved TEXT chunks to the LLM and generates an answer
    """

    context_chunks = [
        item.payload["text"]
        for item in retrieved_points
        if item.payload["type"] == "text"
    ]

    if not context_chunks:
        return "No relevant text found in the document."

    context = "\n\n".join(context_chunks[:5])

    prompt = f"""
You are a helpful assistant answering questions from a document.

Context:
{context}

Question:
{query}

Answer using only the provided context.
"""

    response = client_llm.chat.completions.create(
        model="stepfun/step-3.5-flash:free",
        messages=[
            {"role": "system", "content": "You are a document question answering assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
