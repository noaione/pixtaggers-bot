import base64
from io import BytesIO
from typing import AsyncGenerator, Literal, TypedDict

import httpx
from tqdm import tqdm


class SimplePost(TypedDict):
    id: int
    version: str
    image_url: str
    thumbnail_url: str
    tags: list[str]
    safety: str
    kind: Literal["image", "video", "animation"]


class SimpleTag(TypedDict):
    names: list[str]  # Name + aliases
    category: str
    version: str
    usages: int


class SzurubooruClient:
    def __init__(self, base_url: str, username: str, token: str):
        """
        Initializes the client with the base URL and API credentials.
        """
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api"

        # Szurubooru authenticates via a base64 encoded "username:token" string
        auth_string = f"{username}:{token}".encode("utf-8")
        encoded_auth = base64.b64encode(auth_string).decode("utf-8")

        self.session = httpx.AsyncClient(timeout=30.0)
        self.session.headers.update({
            "Authorization": f"Token {encoded_auth}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    async def close(self):
        await self.session.aclose()

    async def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Internal helper for API requests."""
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        response = await self.session.request(method, url, **kwargs)
        if response.status_code >= 400:
            raise Exception(f"API request failed: {response.status_code} {response.text}")
        return response.json()

    async def get_current_tags(self) -> list[str]:
        """Fetches the list of all tags currently in the system."""
        offset = 0
        limit = 100
        all_tags = []
        while True:
            params = {"offset": offset, "limit": limit}
            data = await self._request("GET", "tags", params=params)
            tags = data.get("results", [])
            if not tags:
                break

            for tag in tags:
                all_tags.extend(tag["names"])  # Add all names and aliases to the set of current tags
            offset += limit
        return all_tags

    async def create_tag(self, tag_name: str, category: str) -> dict:
        """Creates a new tag in the system."""
        payload = {"names": [tag_name], "category": category}
        return await self._request("POST", "tags", json=payload)

    async def iter_posts(self, query: str = "", limit: int = 50) -> AsyncGenerator[SimplePost, None]:
        """
        Iterates through existing posts, yielding the ID, version, and image URL.
        """
        offset = 0
        while True:
            params = {"query": query, "offset": offset, "limit": limit}

            data = await self._request("GET", "posts", params=params)
            posts = data.get("results", [])

            if not posts:
                break

            for post in posts:
                yield {
                    "id": post["id"],
                    "version": post["version"],
                    # Szurubooru stores the relative media path in 'contentUrl'
                    "image_url": f"{self.base_url}/{post['contentUrl'].lstrip('/')}",
                    "thumbnail_url": f"{self.base_url}/{post['thumbnailUrl'].lstrip('/')}",
                    "tags": [t["names"][0] for t in post["tags"]],
                    "safety": post["safety"],
                    "kind": post["type"],
                }

            offset += limit

    async def download_image(self, url: str) -> bytes:
        """
        Downloads an image from the given URL.
        """
        async with self.session.stream("GET", url) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0)) or None
            chunks: list[bytes] = []
            with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading") as bar:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    chunks.append(chunk)
                    bar.update(len(chunk))
            return b"".join(chunks)

    async def get_post(self, post_id: int) -> SimplePost:
        """
        Fetches the details of a specific post by ID.
        """
        data = await self._request("GET", f"post/{post_id}")
        return {
            "id": data["id"],
            "version": data["version"],
            "image_url": f"{self.base_url}/{data['contentUrl'].lstrip('/')}",
            "thumbnail_url": f"{self.base_url}/{data['thumbnailUrl'].lstrip('/')}",
            "tags": [t["names"][0] for t in data["tags"]],
            "safety": data["safety"],
            "kind": data["type"],
        }

    async def update_post(
        self, post_id: int, version: str, tags: list[str] | None = None, safety: str | None = None
    ) -> SimplePost:
        """
        Updates the tags and/or rating (safety) of a specific post.
        Requires the current 'version' of the post for optimistic concurrency control.
        """
        payload: dict[str, int | list[str] | str] = {"version": version}

        if tags is not None:
            payload["tags"] = tags

        # Safety usually accepts 'safe', 'sketchy', or 'unsafe'
        if safety is not None:
            payload["safety"] = safety

        data = await self._request("PUT", f"post/{post_id}", json=payload)
        return {
            "id": data["id"],
            "version": data["version"],
            "image_url": f"{self.base_url}/{data['contentUrl'].lstrip('/')}",
            "thumbnail_url": f"{self.base_url}/{data['thumbnailUrl'].lstrip('/')}",
            "tags": [t["names"][0] for t in data["tags"]],
            "safety": data["safety"],
            "kind": data["type"],
        }

    async def iter_tags(self, query: str = "", limit: int = 100) -> AsyncGenerator[SimpleTag, None]:
        """
        Iterates through tags in the booru.
        """
        offset = 0
        while True:
            params = {"query": query, "offset": offset, "limit": limit}

            data = await self._request("GET", "tags", params=params)
            tags = data.get("results", [])

            if not tags:
                break

            for tag in tags:
                yield tag

            offset += limit

    async def delete_tag(self, tag_name: str, version: str) -> None:
        """
        Deletes a tag. The API requires the tag's current version
        and enforces that the tag has 0 usages.
        """
        payload = {"version": version}
        await self._request("DELETE", f"tag/{tag_name}", json=payload)

    async def update_thumbnail(self, post_id: int, version: str, thumbnail_data: bytes) -> None:
        """
        Updates the thumbnail URL of a specific post.
        Requires the current 'version' of the post for optimistic concurrency control.
        """
        buffer_io = BytesIO(thumbnail_data)
        token_resp = await self._request(
            "POST", "uploads", files={"content": (None, buffer_io, "image/png")}, headers={"Content-Type": None}
        )
        payload = {"version": version, "thumbnailToken": token_resp["token"]}
        await self._request("PUT", f"post/{post_id}", json=payload)

    async def batch_create_tags(self, tags: list[str], category: str):
        if not tags:
            return
        for tag in tags:
            try:
                await self.create_tag(tag, category)
                print(f"Created tag '{tag}' in category '{category}'.")
            except Exception as e:
                print(f"Error creating tag '{tag}': {e}")
