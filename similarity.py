from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(query_features: np.ndarray, image_features: List[np.ndarray]) -> np.ndarray:
    """Compute cosine similarity between query and all candidate image features."""
    if len(image_features) == 0:
        return np.array([], dtype=np.float32)

    q = query_features.reshape(1, -1)
    m = np.vstack(image_features)
    sims = cosine_similarity(q, m)[0]
    return sims.astype(np.float32)


def get_top_k_similar(similarities: np.ndarray, k: int) -> List[Tuple[int, float]]:
    """Return list of (index, similarity) for top-k similarities."""
    if similarities.size == 0 or k <= 0:
        return []
    top_indices = np.argsort(similarities)[::-1][:k]
    return [(int(i), float(similarities[i])) for i in top_indices]
