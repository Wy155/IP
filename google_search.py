import hashlib
from io import BytesIO
from typing import List, Optional

import requests
from PIL import Image


def search_images_google(
    query: str,
    num_images: int = 20,
    api_key: str | None = None,
    cx: str | None = None
) -> List[str]:
    """
    Search for images using Google Custom Search API if (api_key, cx) provided.
    Otherwise returns fallback URLs.
    """
    if api_key and cx:
        try:
            from googleapiclient.discovery import build

            service = build("customsearch", "v1", developerKey=api_key)

            image_urls: List[str] = []
            for start in range(1, num_images + 1, 10):
                result = service.cse().list(
                    q=query,
                    cx=cx,
                    searchType="image",
                    num=min(10, num_images - len(image_urls)),
                    start=start,
                ).execute()

                if "items" in result:
                    for item in result["items"]:
                        image_urls.append(item["link"])
                        if len(image_urls) >= num_images:
                            break

            if image_urls:
                return image_urls
        except Exception:
            # Fall back below
            pass

    return search_images_fallback(query, num_images)


def search_images_fallback(query: str, num_images: int = 20) -> List[str]:
    """
    Fallback: deterministic Picsum URLs using an md5 seed of query.
    """
    base_url = "https://picsum.photos/seed"
    qhash = hashlib.md5(query.encode("utf-8")).hexdigest()[:12]

    urls: List[str] = []
    for i in range(num_images):
        seed = f"{qhash}_{i}"
        urls.append(f"{base_url}/{seed}/400/400")
    return urls


def download_image(url: str, timeout: int = 10) -> Optional[Image.Image]:
    """Download image from URL and return PIL Image, or None on failure."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content))
        return img
    except Exception:
        return None
