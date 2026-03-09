import requests
import os

API_KEY = "YOUR_API_KEY"
CX = "YOUR_SEARCH_ENGINE_ID"

def search_images(keyword):

    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "q": keyword,
        "cx": CX,
        "searchType": "image",
        "num": 10,
        "key": API_KEY
    }

    res = requests.get(url, params=params).json()

    links = [item["link"] for item in res["items"]]

    return links


def download_images(links):

    paths = []

    for i, link in enumerate(links):

        img_data = requests.get(link).content
        path = f"temp_images/img_{i}.jpg"

        with open(path, "wb") as f:
            f.write(img_data)

        paths.append(path)

    return paths