import base64
import hashlib
import os
import uuid
from pathlib import Path
from typing import Any
import uvicorn
import anthropic
import numpy as np
import pymupdf
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# ---------------- Configuration ----------------

EMBEDDING_API_URL = os.getenv(
    "EMBEDDING_API_URL",
    "https://YOUR-NGROK-URL.ngrok-free.app/embed",
)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")

UPLOAD_DIR = Path("uploaded_pdfs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TEXT_BATCH_SIZE = 16
IMAGE_BATCH_SIZE = 2
TOP_K = 6
MAX_IMAGES_TO_SEND = 2
MAX_PDF_SIZE_MB = 100

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is missing from .env")

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

app = FastAPI(title="PDF RAG Chatbot API")

# For local development. Restrict this to your frontend origin in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory document sessions:
# document_id -> {"items": [...], "vectors": np.ndarray, "filename": str}
documents: dict[str, dict[str, Any]] = {}


# ---------------- Embedding API ----------------

def call_embedding_api(payload: dict) -> list[list[float]]:
    try:
        response = requests.post(
            EMBEDDING_API_URL,
            json=payload,
            timeout=(20, 300),
        )
        response.raise_for_status()
        result = response.json()
        return result["embeddings"]
    except requests.RequestException as exc:
        raise RuntimeError(f"Embedding API request failed: {exc}") from exc
    except (KeyError, ValueError) as exc:
        raise RuntimeError("Embedding API returned an invalid response.") from exc


def embed_texts(
    texts: list[str],
    *,
    is_query: bool = False,
    batch_size: int = TEXT_BATCH_SIZE,
) -> np.ndarray:
    vectors = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(
            call_embedding_api({
                "texts": batch,
                "is_query": is_query,
            })
        )

    if not vectors:
        return np.empty((0, 768), dtype=np.float32)

    result = np.asarray(vectors, dtype=np.float32)
    return normalize_vectors(result)


def embed_images(
    images_b64: list[str],
    batch_size: int = IMAGE_BATCH_SIZE,
) -> np.ndarray:
    vectors = []

    for start in range(0, len(images_b64), batch_size):
        batch = images_b64[start : start + batch_size]
        vectors.extend(
            call_embedding_api({"images": batch})
        )

    if not vectors:
        return np.empty((0, 768), dtype=np.float32)

    result = np.asarray(vectors, dtype=np.float32)
    return normalize_vectors(result)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# ---------------- PDF extraction ----------------

def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    step = size - overlap
    return [
        text[i : i + size].strip()
        for i in range(0, len(text), step)
        if text[i : i + size].strip()
    ]


def build_index(pdf_path: Path) -> tuple[list[dict], np.ndarray]:
    text_items = []
    image_items = []

    with pymupdf.open(pdf_path) as pdf:
        total_pages = len(pdf)

        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text").strip()

            for chunk in chunk_text(page_text):
                if len(chunk) > 50:
                    text_items.append({
                        "kind": "text",
                        "page": page_number,
                        "content": chunk,
                    })

            # Render pages for charts, tables, diagrams, and other visual content.
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(1.2, 1.2),
                alpha=False,
            )
            image_b64 = base64.b64encode(
                pix.tobytes("png")
            ).decode("utf-8")

            image_items.append({
                "kind": "image",
                "page": page_number,
                "content": image_b64,
            })

            print(f"Rendered page {page_number}/{total_pages}")

    text_vectors = embed_texts(
        [item["content"] for item in text_items]
    )
    image_vectors = embed_images(
        [item["content"] for item in image_items]
    )

    items = text_items + image_items
    vector_parts = [
        vectors
        for vectors in (text_vectors, image_vectors)
        if len(vectors) > 0
    ]

    if not vector_parts:
        raise ValueError("No text or images could be extracted from this PDF.")

    all_vectors = np.vstack(vector_parts).astype(np.float32)

    if len(items) != len(all_vectors):
        raise RuntimeError("Embedding count does not match extracted items.")

    return items, all_vectors


# ---------------- Retrieval ----------------

def retrieve(
    question: str,
    items: list[dict],
    vectors: np.ndarray,
    top_k: int = TOP_K,
) -> list[dict]:
    query_vector = embed_texts([question], is_query=True)[0]

    # Vectors are normalized, so dot product is cosine similarity.
    scores = vectors @ query_vector
    indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for index in indices:
        item = items[int(index)].copy()
        item["score"] = float(scores[index])
        results.append(item)

    return results


# ---------------- Claude answer generation ----------------

def answer_with_claude(question: str, retrieved_items: list[dict]) -> str:
    content = []

    text_results = [
        item for item in retrieved_items
        if item["kind"] == "text"
    ]

    if text_results:
        excerpts = [
            f"[PDF page {item['page']}]\n{item['content']}"
            for item in text_results
        ]

        content.append({
            "type": "text",
            "text": (
                "Retrieved text excerpts from the uploaded PDF:\n\n"
                + "\n\n---\n\n".join(excerpts)
            ),
        })

    image_results = [
        item for item in retrieved_items
        if item["kind"] == "image"
    ][:MAX_IMAGES_TO_SEND]

    for item in image_results:
        content.append({
            "type": "text",
            "text": f"Retrieved PDF page image: page {item['page']}.",
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": item["content"],
            },
        })

    content.append({
    "type": "text",
    "text": (
        f"Question: {question}\n\n"
        "Answer using only the supplied PDF excerpts and page images. "
        "Read values from tables or images when needed. "
        "For charts, describe only what is visibly supported. "
        "Do not invent numeric values, units, labels, or chart endpoints. "
        "Do not infer units such as million or billion unless the PDF states them. "
        "If a value or unit is not clearly shown, say it is not specified. "
        "If the evidence does not contain the answer, say you could not "
        "find it in the retrieved material. Do not guess. "
        "Include the supporting PDF page number(s)."
    ),
})

    response = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": content}],
    )

    return "\n".join(
        block.text
        for block in response.content
        if block.type == "text"
    )


# ---------------- API schemas ----------------

class AskRequest(BaseModel):
    document_id: str
    question: str


# ---------------- API routes ----------------

@app.get("/")
def root():
    return {"status": "ok", "service": "PDF RAG Chatbot"}


@app.get("/health")
def health():
    return {"status": "ready"}


@app.post("/file_reciever")
async def upload_pdf(file: UploadFile = File(...)):
    # Keep this endpoint spelling because your frontend already uses it.
    filename = file.filename or "uploaded.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    document_id = str(uuid.uuid4())
    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"

    try:
        size = 0

        with pdf_path.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                size += len(chunk)
                if size > MAX_PDF_SIZE_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF exceeds the {MAX_PDF_SIZE_MB} MB limit.",
                    )

                output.write(chunk)

        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")

        # Validate that PyMuPDF can open the uploaded file.
        try:
            with pymupdf.open(pdf_path) as pdf:
                if len(pdf) == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="The PDF contains no pages.",
                    )
        except pymupdf.FileDataError as exc:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid PDF.",
            ) from exc

        print(f"Indexing uploaded PDF: {filename}")
        items, vectors = build_index(pdf_path)

        documents[document_id] = {
            "items": items,
            "vectors": vectors,
            "filename": filename,
        }

        return {
            "document_id": document_id,
            "filename": filename,
            "text_chunks": sum(item["kind"] == "text" for item in items),
            "page_images": sum(item["kind"] == "image" for item in items),
            "message": "PDF uploaded and indexed successfully.",
        }

    except HTTPException:
        pdf_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        print(f"PDF processing failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not process PDF: {exc}",
        ) from exc
    finally:
        await file.close()


@app.post("/ask")
def ask_question(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    document = documents.get(request.document_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found. Please upload the PDF again.",
        )

    try:
        retrieved_items = retrieve(
            question,
            document["items"],
            document["vectors"],
        )

        answer = answer_with_claude(question, retrieved_items)

        source_pages = sorted({
            item["page"] for item in retrieved_items
        })

        return {
            "answer": answer,
            "source_pages": source_pages,
        }

    except anthropic.APIError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Claude API error: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not answer question: {exc}",
        ) from exc

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)