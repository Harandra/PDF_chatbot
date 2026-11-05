%%writefile /content/embedding_api.py
import base64
import binascii
import io
import logging
import os
import threading
from contextlib import asynccontextmanager

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer


# ---------------- Configuration ----------------

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-2")
DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")
PORT = int(os.getenv("PORT", "6885"))

MAX_TEXTS = 32
MAX_IMAGES = 8
MAX_TEXT_CHARS = 20_000
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
EMBEDDING_BATCH_SIZE = 8

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding-api")
inference_lock = threading.Lock()


# ---------------- Request / response models ----------------

class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[str] = Field(default_factory=list, max_length=MAX_TEXTS)
    images: list[str] = Field(default_factory=list, max_length=MAX_IMAGES)
    is_query: bool = False


class EmbedResponse(BaseModel):
    model: str
    dimension: int
    embeddings: list[list[float]]


# ---------------- Model lifecycle ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable. Enable a GPU runtime "
            "or set EMBEDDING_DEVICE=cpu."
        )

    logger.info("Loading %s on %s", MODEL_NAME, DEVICE)

    app.state.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    app.state.dimension = app.state.model.get_sentence_embedding_dimension()

    logger.info(
        "Model ready: device=%s dimension=%s",
        app.state.model.device,
        app.state.dimension,
    )

    yield

    if hasattr(app.state, "model"):
        del app.state.model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="EmbeddingGemma API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------- Helpers ----------------

def validate_request(req: EmbedRequest):
    if not req.texts and not req.images:
        raise HTTPException(400, "Provide at least one text or image.")

    for i, text in enumerate(req.texts):
        if not text.strip():
            raise HTTPException(400, f"Text at index {i} is empty.")

        if len(text) > MAX_TEXT_CHARS:
            raise HTTPException(
                413,
                f"Text at index {i} exceeds {MAX_TEXT_CHARS} characters.",
            )


def decode_image(image_b64: str) -> Image.Image:
    try:
        if image_b64.startswith("data:"):
            _, image_b64 = image_b64.split(",", 1)

        image_bytes = base64.b64decode(image_b64, validate=True)

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Image file is too large.")

        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Image dimensions are too large.")
            source.verify()

        with Image.open(io.BytesIO(image_bytes)) as source:
            return source.convert("RGB")

    except HTTPException:
        raise
    except (
        binascii.Error,
        ValueError,
        UnidentifiedImageError,
        OSError,
    ) as exc:
        raise HTTPException(400, "Invalid image or image format.") from exc


def normalize_vectors(vectors) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    if not np.isfinite(vectors).all():
        raise RuntimeError("Model returned non-finite embedding values.")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


# ---------------- Endpoints ----------------

@app.get("/health/live")
def liveness():
    return {"status": "alive"}


@app.get("/health/ready")
def readiness():
    model = getattr(app.state, "model", None)

    if model is None:
        raise HTTPException(503, "Model is not ready.")

    return {
        "status": "ready",
        "model": MODEL_NAME,
        "device": str(model.device),
        "dimension": app.state.dimension,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    validate_request(req)

    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(503, "Model is not ready.")

    vectors = []

    try:
        with inference_lock, torch.inference_mode():
            # Text embeddings
            for start in range(0, len(req.texts), EMBEDDING_BATCH_SIZE):
                batch = req.texts[start:start + EMBEDDING_BATCH_SIZE]

                if req.is_query:
                    result = model.encode_query(
                        batch,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                else:
                    result = model.encode_document(
                        batch,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )

                vectors.extend(normalize_vectors(result).tolist())

            # Image embeddings
            for image_b64 in req.images:
                image = decode_image(image_b64)

                result = model.encode(
                    {"image": image},
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

                vectors.extend(normalize_vectors(result).tolist())

        return EmbedResponse(
            model=MODEL_NAME,
            dimension=app.state.dimension,
            embeddings=vectors,
        )

    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.exception("GPU out of memory")
        raise HTTPException(
            503,
            "GPU memory exhausted. Try a smaller batch.",
        ) from exc
    except Exception as exc:
        logger.exception("Embedding request failed")
        raise HTTPException(
            500,
            "Embedding request failed. Check the server logs.",
        ) from exc


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        workers=1,
        log_level="info",
    )