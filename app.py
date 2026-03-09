import streamlit as st
import cv2
import numpy as np
from feature_extractor import extract_features
from similarity import get_top_k
from google_search import search_images, download_images

st.title("CBIR Image Retrieval System")

# Step 1 : Upload query image
uploaded = st.file_uploader("Upload Query Image", type=["jpg","png"])

# Step 2 : Keyword input
keyword = st.text_input("Search keyword")

if uploaded and keyword:

    # read query image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    query_img = cv2.imdecode(file_bytes, 1)

    st.image(query_img, caption="Query Image")

    # extract query feature
    query_feature = extract_features(query_img)

    # Step 3 Google search
    links = search_images(keyword)

    # Step 4 download images
    paths = download_images(links)

    features = []
    valid_paths = []

    # Step 5 extract features
    for p in paths:

        img = cv2.imread(p)

        if img is None:
            continue

        f = extract_features(img)

        features.append(f)
        valid_paths.append(p)

    # Step 6 similarity
    results = get_top_k(query_feature, features, valid_paths, k=5)

    st.subheader("Top Similar Images")

    for score, path in results:

        st.image(path)
        st.write("Similarity:", round(score,3))