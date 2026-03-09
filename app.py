import time
from typing import List, Tuple

import streamlit as st
import numpy as np
from PIL import Image

from feature_extractor import (
    METHODS,
    load_feature_extractor,
    load_classifier,
    load_imagenet_labels,
    extract_features,
    classify_image,
)
from google_search import search_images_google, download_image
from similarity import compute_similarity, get_top_k_similar


# ─────────────────────────────────────────────────
# Method descriptions shown in the sidebar
# ─────────────────────────────────────────────────
METHOD_INFO = {
    "resnet":     ("🤖 Deep learning (ResNet-50 ImageNet)", "2 048-d", "High – requires PyTorch"),
    "color_hist": ("🎨 RGB intensity distribution per channel", "192-d", "Very fast – no extra deps"),
    "hsv_hist":   ("🌈 Hue/Saturation/Value distribution", "34-d",  "Very fast – no extra deps"),
    "glcm":       ("🔲 Co-occurrence texture statistics", "48-d",   "Moderate – needs scikit-image"),
    "hog":        ("📐 Edge & gradient orientation histograms", "~1 764-d", "Fast – needs scikit-image"),
    "lbp":        ("🔵 Micro-texture local binary patterns", "26-d",  "Fast – needs scikit-image"),
    "gabor":      ("〰️ Frequency/orientation filter responses", "48-d",  "Moderate – needs scikit-image + scipy"),
    "combined":   ("🔗 Color Hist + HOG + ResNet (concatenated)", "varies", "Slow – needs all deps"),
}

NEEDS_RESNET = {"resnet", "combined"}


def inject_css():
    st.markdown(
        """
        <style>
          .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
          }
          .pipeline-step {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
            margin-bottom: 1rem;
          }
          .similarity-score {
            font-size: 1.2rem;
            font-weight: bold;
            color: #667eea;
          }
          .method-badge {
            background: #f0f2ff;
            border-left: 4px solid #667eea;
            padding: 0.4rem 0.8rem;
            border-radius: 4px;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
          }
          .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_feature_heatmap(features: np.ndarray, title: str = "Extracted Features"):
    """Optional visualization; requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        st.caption("Install matplotlib to see feature heatmap: pip install matplotlib")
        return

    st.subheader("🧠 " + title)
    size = features.size
    side = int(np.ceil(np.sqrt(min(size, 256))))
    padded = np.zeros(side * side, dtype=np.float32)
    padded[:min(size, side * side)] = features[:min(size, side * side)]
    grid = padded.reshape(side, side)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_title(f"Feature Vector ({size:,} dims)")
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.set_page_config(page_title="Hybrid CBIR", page_icon="🔍", layout="wide")
    inject_css()

    st.markdown('<h1 class="main-header">🔍 Hybrid CBIR</h1>', unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Configuration")

    # Feature extraction method selector
    st.sidebar.subheader("🧪 Feature Extraction Method")
    method_label = st.sidebar.selectbox(
        "Select method",
        options=list(METHODS.keys()),
        index=0,
        help="Choose how image features are extracted for similarity comparison.",
    )
    method_key = METHODS[method_label]

    # Show method info card
    if method_key in METHOD_INFO:
        desc, dims, cost = METHOD_INFO[method_key]
        st.sidebar.markdown(
            f"<div class='method-badge'>"
            f"<b>{desc}</b><br>"
            f"📏 Dimensions: {dims}<br>"
            f"⚡ Cost: {cost}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    top_k = st.sidebar.slider("Top K Results", min_value=1, max_value=20, value=5)
    num_search_images = st.sidebar.slider("Number of Search Images", min_value=5, max_value=50, value=20)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 Google API (Optional)")
    api_key = st.sidebar.text_input("API Key", type="password", help="Google Custom Search API Key")
    cx = st.sidebar.text_input("Search Engine ID", help="Custom Search Engine ID")

    # ── Session state ─────────────────────────────────────────────────────────
    if "results" not in st.session_state:
        st.session_state.results = {}

    # ── Step 1: Upload ────────────────────────────────────────────────────────
    st.markdown('<p class="pipeline-step">Step 1: Upload Image</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload an image to find similar images",
    )

    if uploaded_file is None:
        st.info("👆 Please upload an image to start the similarity pipeline")
        st.markdown("### 📋 Pipeline Steps:")
        for s in [
            "1️⃣ **Upload Image** – Select an image to analyze",
            "2️⃣ **Feature Extraction** – Extract features using your chosen method",
            "3️⃣ **Image Classification** – Generate keywords from ResNet predictions",
            "4️⃣ **Google Image Search** – Search for similar images online",
            "5️⃣ **Download Images** – Fetch images from search results",
            "6️⃣ **Feature Extraction** – Extract features from all downloaded images",
            "7️⃣ **Similarity Comparison** – Compute cosine similarity scores",
            "8️⃣ **Show Top K** – Display the most similar images",
        ]:
            st.markdown(s)
        return

    query_image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(query_image, caption="Uploaded Image", use_container_width=True)

    st.info(f"🧪 Feature method: **{method_label}**", icon="ℹ️")

    # ── Run Pipeline ──────────────────────────────────────────────────────────
    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        st.session_state.results = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Conditionally load ResNet
        feature_extractor = None
        classifier = None
        labels = None

        needs_resnet = method_key in NEEDS_RESNET

        if needs_resnet:
            status_text.text("Loading ResNet-50 model…")
            feature_extractor = load_feature_extractor()

        # Always load classifier for keyword generation (Step 3)
        status_text.text("Loading classifier for keyword generation…")
        classifier = load_classifier()
        labels = load_imagenet_labels()

        # Step 2: Feature extraction on query image
        status_text.text(f"Step 2: Extracting features ({method_label})…")
        progress_bar.progress(10)

        query_features = extract_features(query_image, model=feature_extractor, method=method_key)
        st.session_state.results["query_features"] = query_features
        st.session_state.results["method_label"] = method_label
        progress_bar.progress(20)

        # Step 3: Classification for search keywords
        status_text.text("Step 3: Classifying image to generate keywords…")
        progress_bar.progress(30)

        classifications = classify_image(query_image, classifier, labels, top_k=5)
        st.session_state.results["classifications"] = classifications
        progress_bar.progress(40)

        keywords = [cls[0] for cls in classifications[:3]]
        search_query = " ".join(keywords)
        st.session_state.results["search_query"] = search_query

        # Step 4: Search
        status_text.text(f"Step 4: Searching images for '{search_query}'…")
        progress_bar.progress(50)
        image_urls = search_images_google(
            search_query,
            num_images=num_search_images,
            api_key=api_key or None,
            cx=cx or None,
        )
        st.session_state.results["image_urls"] = image_urls
        progress_bar.progress(60)

        # Step 5: Download
        status_text.text("Step 5: Downloading images…")
        progress_bar.progress(65)

        downloaded_images: List[Tuple[str, Image.Image]] = []
        download_progress = st.progress(0.0)

        for i, url in enumerate(image_urls):
            img = download_image(url)
            if img is not None:
                downloaded_images.append((url, img))
            download_progress.progress((i + 1) / max(1, len(image_urls)))

        download_progress.empty()
        st.session_state.results["downloaded_images"] = downloaded_images
        progress_bar.progress(75)

        # Step 6: Features for downloaded images
        status_text.text(f"Step 6: Extracting features from downloaded images ({method_label})…")
        image_features: List[np.ndarray] = []
        feature_progress = st.progress(0.0)

        for i, (_, img) in enumerate(downloaded_images):
            feats = extract_features(img, model=feature_extractor, method=method_key)
            image_features.append(feats)
            feature_progress.progress((i + 1) / max(1, len(downloaded_images)))

        feature_progress.empty()
        st.session_state.results["image_features"] = image_features
        progress_bar.progress(85)

        # Step 7: Similarity
        status_text.text("Step 7: Computing cosine similarity…")
        progress_bar.progress(90)

        similarities = compute_similarity(query_features, image_features)
        st.session_state.results["similarities"] = similarities
        progress_bar.progress(95)

        # Step 8: Top K
        status_text.text("Step 8: Ranking and selecting Top K images…")
        top_k_results = get_top_k_similar(similarities, min(top_k, len(similarities)))
        st.session_state.results["top_k_results"] = top_k_results

        progress_bar.progress(100)
        status_text.text("✅ Pipeline completed!")
        time.sleep(0.6)
        status_text.empty()
        progress_bar.empty()

    # ── Display Results ───────────────────────────────────────────────────────
    results = st.session_state.results

    if "classifications" in results:
        st.markdown("---")
        st.markdown(
            '<p class="pipeline-step">Step 2-3: Feature Extraction & Classification</p>',
            unsafe_allow_html=True,
        )

        colA, colB = st.columns(2)
        with colA:
            show_feature_heatmap(
                results["query_features"],
                title=f"Features – {results.get('method_label', '')}",
            )

        with colB:
            st.subheader("🏷️ Image Classification (ResNet keywords)")
            for label, prob in results["classifications"]:
                st.write(f"**{label}**")
                st.progress(prob)
                st.caption(f"Confidence: {prob * 100:.2f}%")
            st.info(f"🔍 Search Query: **{results['search_query']}**")

    if "downloaded_images" in results:
        st.markdown("---")
        st.markdown(
            '<p class="pipeline-step">Step 4-5: Image Search & Download</p>',
            unsafe_allow_html=True,
        )

        downloaded_images = results["downloaded_images"]
        st.write(f"Downloaded {len(downloaded_images)} images")

        cols = st.columns(5)
        for i, (_, img) in enumerate(downloaded_images[:20]):
            with cols[i % 5]:
                st.image(img, use_container_width=True, caption=f"Image {i + 1}")

    if "top_k_results" in results:
        st.markdown("---")
        st.markdown(
            '<p class="pipeline-step">Step 6-8: Similarity Comparison & Top K Results</p>',
            unsafe_allow_html=True,
        )

        top_k_results = results["top_k_results"]
        downloaded_images = results.get("downloaded_images", [])

        st.subheader(
            f"🏆 Top {len(top_k_results)} Most Similar Images  "
            f"<small style='color:#888;font-size:0.8rem'>({results.get('method_label','')})</small>",
        )

        if not top_k_results:
            st.warning("No similarity results to display (no images downloaded / no features).")
            return

        cols = st.columns(min(len(top_k_results), 5))
        for i, (idx, sim) in enumerate(top_k_results):
            with cols[i % 5]:
                if idx < len(downloaded_images):
                    _, img = downloaded_images[idx]
                    st.image(img, use_container_width=True)
                    st.markdown(
                        f'<p class="similarity-score">🎯 {sim * 100:.1f}%</p>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Rank #{i + 1}")

        # Optional detailed table
        try:
            import pandas as pd

            st.markdown("---")
            st.subheader("📊 Detailed Similarity Scores")

            table = []
            for rank, (idx, sim) in enumerate(top_k_results, 1):
                table.append({
                    "Rank": rank,
                    "Image Index": idx + 1,
                    "Similarity Score": f"{sim * 100:.2f}%",
                    "Cosine Similarity": f"{sim:.4f}",
                })
            st.dataframe(pd.DataFrame(table), use_container_width=True)
        except Exception:
            st.caption("Install pandas to see the detailed table: pip install pandas")


if __name__ == "__main__":
    main()
