"""Convert text into embeddings (vector representations)
so later used for similarity search
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Multilingual model (works with Arabic + English)
_MODEL_NAME = "intfloat/multilingual-e5-base"
_model = None


def get_model() -> SentenceTransformer:
    """Load the model once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of texts into embeddings.
    shape returned will be (num_texts, embedding_dim)
    """
    model = get_model()

    # E5 model expects "passage:" prefix for documents
    passages = [f"passage: {t}" for t in texts]

    vectors = model.encode(passages, normalize_embeddings=True)
    return np.array(vectors)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query."""
    model = get_model()

    # E5 model expects "query:" prefix
    q = f"query: {query}"
    vec = model.encode([q], normalize_embeddings=True)

    return np.array(vec)[0]

