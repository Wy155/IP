import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_top_k(query_feature, features, paths, k=5):

    scores = []

    for i, f in enumerate(features):

        score = cosine_similarity(query_feature, f)
        scores.append((score, paths[i]))

    scores.sort(reverse=True)

    return scores[:k]