import base64
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pymupdf
import requests
from PIL import Image
from io import BytesIO

API_URL = "https://6a22-34-32-212-13.ngrok-free.app/embed"
PDF_PATH = Path("/home/harandra/Downloads/Airbus-A330-MSFS-Manual.pdf")
CACHE_PATH = PDF_PATH.with_suffix(".rag_cache.pkl")

TEXT_BATCH_SIZE = 16
IMAGE_BATCH_SIZE = 2
TOP_K = 5


# -------------------- Embedding API --------------------

def embed_texts(texts, is_query=False, batch_size=TEXT_BATCH_SIZE):
    vectors = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        response = requests.post(
            API_URL,
            json={"texts": batch, "is_query": is_query},
            timeout=(20, 300),
        )
        response.raise_for_status()

        vectors.extend(response.json()["embeddings"])
        print(
            f"\rEmbedding text: {min(start + len(batch), len(texts))}"
            f"/{len(texts)}",
            end="",
            flush=True,
        )

    if texts:
        print()

    return np.asarray(vectors, dtype=np.float32)


def embed_images(images_b64, batch_size=IMAGE_BATCH_SIZE):
    vectors = []

    for start in range(0, len(images_b64), batch_size):
        batch = images_b64[start:start + batch_size]

        response = requests.post(
            API_URL,
            json={"images": batch},
            timeout=(20, 300),
        )
        response.raise_for_status()

        vectors.extend(response.json()["embeddings"])
        print(
            f"\rEmbedding images: {min(start + len(batch), len(images_b64))}"
            f"/{len(images_b64)}",
            end="",
            flush=True,
        )

    if images_b64:
        print()

    return np.asarray(vectors, dtype=np.float32)


# -------------------- Text chunking --------------------

def chunk_text(text, size=800, overlap=150):
    step = size - overlap

    return [
        text[i:i + size].strip()
        for i in range(0, len(text), step)
        if text[i:i + size].strip()
    ]


# -------------------- PDF cache helpers --------------------

def pdf_fingerprint(pdf_path):
    """Create a cache key from the PDF path, size, and modification time."""
    stat = pdf_path.stat()
    value = f"{pdf_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_cache():
    if not CACHE_PATH.exists():
        return None

    try:
        with CACHE_PATH.open("rb") as f:
            cached = pickle.load(f)

        if cached.get("fingerprint") == pdf_fingerprint(PDF_PATH):
            print(f"Using cached embeddings: {CACHE_PATH}")
            return cached

        print("PDF changed; rebuilding embeddings.")
    except Exception as e:
        print(f"Could not load cache ({e}); rebuilding.")

    return None


def save_cache(items, vectors):
    cache = {
        "fingerprint": pdf_fingerprint(PDF_PATH),
        "items": items,
        "vectors": vectors,
    }

    with CACHE_PATH.open("wb") as f:
        pickle.dump(cache, f)

    print(f"Saved embedding cache: {CACHE_PATH}")


# -------------------- Extract PDF text and page images --------------------

def build_index():
    text_items = []
    image_items = []

    print("Extracting PDF text and rendering pages...")

    with pymupdf.open(PDF_PATH) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            page_text = page.get_text("text").strip()

            for chunk in chunk_text(page_text):
                if len(chunk) > 50:
                    text_items.append({
                        "kind": "text",
                        "page": page_num,
                        "content": chunk,
                    })

            # Render pages for visual retrieval (tables, diagrams, etc.)
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(1.2, 1.2),
                alpha=False,
            )

            image_b64 = base64.b64encode(
                pix.tobytes("png")
            ).decode("utf-8")

            image_items.append({
                "kind": "image",
                "page": page_num,
                "content": image_b64,
            })

            print(
                f"\rProcessed page {page_num}/{len(pdf)}",
                end="",
                flush=True,
            )

    print()

    print(f"Text chunks: {len(text_items)}")
    print(f"Page images: {len(image_items)}")

    text_vectors = (
        embed_texts([item["content"] for item in text_items])
        if text_items
        else np.empty((0, 768), dtype=np.float32)
    )

    image_vectors = (
        embed_images([item["content"] for item in image_items])
        if image_items
        else np.empty((0, 768), dtype=np.float32)
    )

    all_items = text_items + image_items

    vector_groups = [
        vectors
        for vectors in (text_vectors, image_vectors)
        if len(vectors)
    ]

    all_vectors = np.vstack(vector_groups)

    print("All embeddings:", all_vectors.shape)

    save_cache(all_items, all_vectors)
    return all_items, all_vectors


# -------------------- Load or build index --------------------

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

cached = load_cache()

if cached:
    all_items = cached["items"]
    all_vectors = cached["vectors"]
else:
    all_items, all_vectors = build_index()


# -------------------- Ask and retrieve --------------------

while True:
    question = input("\nAsk a question about the PDF (or type 'exit'): ").strip()

    if question.lower() in {"exit", "quit"}:
        break

    if not question:
        print("Please enter a question.")
        continue

    query_vector = embed_texts([question], is_query=True)[0]

    # Embeddings are normalized by the API, so dot product is cosine similarity.
    scores = all_vectors @ query_vector
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    print(f"\nQuestion: {question}")
    print("\nRetrieved results:\n")

    for rank, idx in enumerate(top_indices, start=1):
        item = all_items[idx]

        print(
            f"{rank}. Similarity: {scores[idx]:.3f} | "
            f"Type: {item['kind']} | Page: {item['page']}"
        )

        if item["kind"] == "text":
            print(item["content"])
        else:
            print(f"Page image retrieved: page {item['page']}")

            # Display the retrieved page image in a window.
            image_bytes = base64.b64decode(item["content"])
            image = Image.open(BytesIO(image_bytes))
            image.show()

        print("-" * 60)