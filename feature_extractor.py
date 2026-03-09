import streamlit as st
import numpy as np
from PIL import Image
import requests
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


# ─────────────────────────────────────────────
# ImageNet transform (used only for ResNet)
# ─────────────────────────────────────────────

def get_image_transform():
    """ImageNet preprocessing for ResNet."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ─────────────────────────────────────────────
# Deep learning models (ResNet-50)
# ─────────────────────────────────────────────

@st.cache_resource
def load_feature_extractor() -> nn.Module:
    """Load pre-trained ResNet50 and drop final FC layer to use as feature extractor."""
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])  # up to avgpool
    feature_extractor.eval()
    return feature_extractor


@st.cache_resource
def load_classifier() -> nn.Module:
    """Load pre-trained ResNet50 classifier."""
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()
    return model


@st.cache_data
def load_imagenet_labels() -> List[str]:
    """Load ImageNet class labels."""
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text.strip().split("\n")
    except Exception:
        return [f"class_{i}" for i in range(1000)]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


# ─────────────────────────────────────────────
# Training-free feature extractors
# ─────────────────────────────────────────────

def extract_color_histogram(image: Image.Image, bins: int = 64) -> np.ndarray:
    """
    RGB Color Histogram.
    Concatenates per-channel histograms → (bins*3,) vector.
    """
    img = _ensure_rgb(image)
    arr = np.array(img)
    hist_parts = []
    for ch in range(3):
        h, _ = np.histogram(arr[:, :, ch], bins=bins, range=(0, 256))
        hist_parts.append(h.astype(np.float32))
    feats = np.concatenate(hist_parts)
    return _l2_normalize(feats)


def extract_hsv_histogram(image: Image.Image, h_bins: int = 18, s_bins: int = 8, v_bins: int = 8) -> np.ndarray:
    """
    HSV Color Histogram.
    Captures hue, saturation and brightness distributions separately.
    """
    img = _ensure_rgb(image).convert("HSV")
    arr = np.array(img)
    h_hist, _ = np.histogram(arr[:, :, 0], bins=h_bins, range=(0, 256))
    s_hist, _ = np.histogram(arr[:, :, 1], bins=s_bins, range=(0, 256))
    v_hist, _ = np.histogram(arr[:, :, 2], bins=v_bins, range=(0, 256))
    feats = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    return _l2_normalize(feats)


def extract_glcm(image: Image.Image, distances: List[int] = None, angles: List[float] = None) -> np.ndarray:
    """
    Gray-Level Co-occurrence Matrix (GLCM) – texture features.
    Extracts: contrast, dissimilarity, homogeneity, energy, correlation, ASM
    for each (distance × angle) combination.
    Requires scikit-image.
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        st.error("scikit-image is required for GLCM. Run: pip install scikit-image")
        return np.zeros(24, dtype=np.float32)

    if distances is None:
        distances = [1, 3]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    img_gray = np.array(_ensure_rgb(image).convert("L"))
    # Reduce to 64 levels for speed
    img_gray = (img_gray // 4).astype(np.uint8)

    glcm = graycomatrix(img_gray, distances=distances, angles=angles,
                        levels=64, symmetric=True, normed=True)

    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feats = []
    for prop in props:
        val = graycoprops(glcm, prop).flatten()
        feats.extend(val.tolist())

    feats = np.array(feats, dtype=np.float32)
    return _l2_normalize(feats)


def extract_hog(image: Image.Image, pixels_per_cell: int = 16) -> np.ndarray:
    """
    Histogram of Oriented Gradients (HOG) – shape & edge features.
    Requires scikit-image.
    """
    try:
        from skimage.feature import hog
    except ImportError:
        st.error("scikit-image is required for HOG. Run: pip install scikit-image")
        return np.zeros(256, dtype=np.float32)

    img_gray = np.array(_ensure_rgb(image).resize((128, 128)).convert("L"))
    feats = hog(
        img_gray,
        orientations=8,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(2, 2),
        feature_vector=True,
    )
    feats = feats.astype(np.float32)
    return _l2_normalize(feats)


def extract_lbp(image: Image.Image, radius: int = 3, n_points: int = None) -> np.ndarray:
    """
    Local Binary Pattern (LBP) – micro-texture features.
    Requires scikit-image.
    """
    try:
        from skimage.feature import local_binary_pattern
    except ImportError:
        st.error("scikit-image is required for LBP. Run: pip install scikit-image")
        return np.zeros(256, dtype=np.float32)

    if n_points is None:
        n_points = 8 * radius

    img_gray = np.array(_ensure_rgb(image).resize((128, 128)).convert("L"))
    lbp = local_binary_pattern(img_gray, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    feats = hist.astype(np.float32)
    return _l2_normalize(feats)


def extract_gabor(image: Image.Image) -> np.ndarray:
    """
    Gabor filter bank – frequency & orientation texture features.
    Uses 4 frequencies × 6 orientations = 24 filters; returns mean + std per filter (48-d).
    Requires scikit-image.
    """
    try:
        from skimage.filters import gabor_kernel
        from scipy.ndimage import convolve
    except ImportError:
        st.error("scikit-image and scipy are required for Gabor. Run: pip install scikit-image scipy")
        return np.zeros(48, dtype=np.float32)

    img_gray = np.array(_ensure_rgb(image).resize((128, 128)).convert("L"), dtype=np.float32) / 255.0

    frequencies = [0.1, 0.2, 0.3, 0.4]
    thetas = [k * np.pi / 6 for k in range(6)]

    feats = []
    for freq in frequencies:
        for theta in thetas:
            kernel = np.real(gabor_kernel(freq, theta=theta))
            filtered = convolve(img_gray, kernel)
            feats.append(filtered.mean())
            feats.append(filtered.std())

    feats = np.array(feats, dtype=np.float32)
    return _l2_normalize(feats)


def extract_combined(image: Image.Image, model: nn.Module) -> np.ndarray:
    """
    Combined: Color Histogram + HOG + ResNet deep features (concatenated & normalised).
    """
    color = extract_color_histogram(image)
    hog_f = extract_hog(image)
    deep = extract_features_resnet(image, model)
    feats = np.concatenate([color, hog_f, deep])
    return _l2_normalize(feats)


# ─────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────

METHODS = {
    "ResNet-50 (Deep)": "resnet",
    "Color Histogram (RGB)": "color_hist",
    "Color Histogram (HSV)": "hsv_hist",
    "GLCM (Texture)": "glcm",
    "HOG (Shape/Edges)": "hog",
    "LBP (Micro-Texture)": "lbp",
    "Gabor Filter Bank": "gabor",
    "Combined (Color + HOG + ResNet)": "combined",
}


def extract_features(
    image: Image.Image,
    model: nn.Module = None,
    method: str = "resnet",
) -> np.ndarray:
    """
    Dispatch to the chosen feature extraction method.

    Parameters
    ----------
    image  : PIL Image
    model  : ResNet model (required only for 'resnet' and 'combined')
    method : key from METHODS dict values
    """
    if method == "resnet":
        if model is None:
            raise ValueError("ResNet model must be provided for method='resnet'")
        return extract_features_resnet(image, model)
    elif method == "color_hist":
        return extract_color_histogram(image)
    elif method == "hsv_hist":
        return extract_hsv_histogram(image)
    elif method == "glcm":
        return extract_glcm(image)
    elif method == "hog":
        return extract_hog(image)
    elif method == "lbp":
        return extract_lbp(image)
    elif method == "gabor":
        return extract_gabor(image)
    elif method == "combined":
        if model is None:
            raise ValueError("ResNet model must be provided for method='combined'")
        return extract_combined(image, model)
    else:
        raise ValueError(f"Unknown method: {method}")


def extract_features_resnet(image: Image.Image, model: nn.Module) -> np.ndarray:
    """Extract L2-normalized deep feature vector from image using ResNet-50."""
    transform = get_image_transform()
    image = _ensure_rgb(image)
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        feats = model(img_tensor)  # [1,2048,1,1]

    feats = feats.squeeze().detach().cpu().numpy().astype(np.float32)  # [2048]
    return _l2_normalize(feats)


# ─────────────────────────────────────────────
# Classification (always uses ResNet)
# ─────────────────────────────────────────────

def classify_image(
    image: Image.Image,
    model: nn.Module,
    labels: List[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Classify image and return top-k (label, probability)."""
    transform = get_image_transform()
    image = _ensure_rgb(image)
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)

    top_probs, top_indices = torch.topk(probs, top_k)

    results: List[Tuple[str, float]] = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        i = idx.item()
        label = labels[i] if i < len(labels) else f"class_{i}"
        results.append((label, float(prob.item())))

    return results
