import streamlit as st
import numpy as np
from PIL import Image
import requests
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


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


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def extract_features(image: Image.Image, model: nn.Module) -> np.ndarray:
    """Extract L2-normalized feature vector from image."""
    transform = get_image_transform()
    image = _ensure_rgb(image)

    img_tensor = transform(image).unsqueeze(0)  # [1,3,224,224]

    with torch.no_grad():
        feats = model(img_tensor)  # [1,2048,1,1] for resnet50 avgpool output

    feats = feats.squeeze().detach().cpu().numpy().astype(np.float32)  # [2048]
    norm = np.linalg.norm(feats)
    if norm > 0:
        feats = feats / norm
    return feats


def classify_image(
    image: Image.Image,
    model: nn.Module,
    labels: List[str],
    top_k: int = 5
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
