import asyncio
import base64
import json
import time
from io import BytesIO
from pathlib import Path
from typing import AsyncGenerator, Literal, TypedDict

import curl_cffi
import httpx

BACKOFF_BASE = 0.75  # Base backoff time in seconds for retries
USER_AGENT = "pixtaggers-bot/0.1.0"
HttpMethod = Literal[
    "GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "TRACE", "PATCH", "QUERY"
]


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
    def __init__(
        self,
        base_url: str,
        username: str,
        token: str,
        tag_cache_path: str | Path | None = ".cache/szurubooru-tags.json",
        http_impersonate: str | None = None,
    ):
        """
        Initializes the client with the base URL and API credentials.
        """
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api"

        # Szurubooru authenticates via a base64 encoded "username:token" string
        auth_string = f"{username}:{token}".encode("utf-8")
        encoded_auth = base64.b64encode(auth_string).decode("utf-8")

        self.tag_cache_path = Path(tag_cache_path) if tag_cache_path is not None else None
        self._tag_cache_write_lock = asyncio.Lock()
        self.http_impersonate = http_impersonate
        if not http_impersonate:
            self.session = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=30),
            )
        else:
            self.session = curl_cffi.AsyncSession(
                timeout=30.0,
                max_clients=100,
                impersonate=http_impersonate,
            )
        self.session.headers.update({
            "Authorization": f"Token {encoded_auth}",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        })

    async def close(self):
        if isinstance(self.session, httpx.AsyncClient):
            await self.session.aclose()
        elif isinstance(self.session, curl_cffi.AsyncSession):
            await self.session.close()

    async def _request(self, method: HttpMethod, endpoint: str, **kwargs) -> dict:
        """Internal helper for API requests."""
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        # Check if kwargs has "files", if yes we should not set "Content-Type" header to "application/json"
        if "files" not in kwargs and "multipart" not in kwargs:
            kwargs.setdefault("headers", {})["Content-Type"] = "application/json"
        response = await self.session.request(method, url, **kwargs)
        if response.status_code >= 400:
            raise Exception(f"API request failed: {response.status_code} {response.text}")
        return response.json()

    async def _repeated_post_update(
        self, post_id: int, base_version: str, base_payload: dict, max_retries: int = 3
    ) -> SimplePost:
        """
        Helper method to handle optimistic concurrency control when updating a post.
        It retries the update if a version conflict occurs, up to a maximum number of retries.
        """
        version = base_version
        for attempt in range(max_retries):
            try:
                payload = {**base_payload, "version": version}
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
            except Exception as e:
                if (
                    "version conflict" in str(e).lower() or "resourcemodified" in str(e).lower()
                ) and attempt < max_retries - 1:
                    await asyncio.sleep(BACKOFF_BASE * (2 ** attempt))  # Exponential backoff
                    # Fetch the latest version and retry
                    latest_data = await self._request("GET", f"post/{post_id}")
                    version = latest_data["version"]
                else:
                    raise
        raise Exception(f"Failed to update post {post_id} after {max_retries} attempts due to version conflicts.")

    def _read_tag_cache(self) -> tuple[list[str], float] | None:
        if self.tag_cache_path is None or not self.tag_cache_path.is_file():
            return None

        try:
            payload = json.loads(self.tag_cache_path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("base_url") != self.base_url
                or not isinstance(payload.get("tags"), list)
            ):
                return None
            tags = [tag for tag in payload["tags"] if isinstance(tag, str)]
            return list(dict.fromkeys(tags)), float(payload["fetched_at"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    async def _write_tag_cache(self, tags: list[str], fetched_at: float | None = None) -> None:
        if self.tag_cache_path is None:
            return

        path = self.tag_cache_path
        payload = {
            "base_url": self.base_url,
            "fetched_at": time.time() if fetched_at is None else fetched_at,
            "tags": list(dict.fromkeys(tags)),
        }
        serialized = json.dumps(payload, separators=(",", ":"))

        def write_cache() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_name(f".{path.name}.tmp")
            temporary_path.write_text(serialized, encoding="utf-8")
            temporary_path.replace(path)

        async with self._tag_cache_write_lock:
            await asyncio.to_thread(write_cache)

    async def get_current_tags(self) -> list[str]:
        """Loads cached tags, fetches newer tags, then stores the merged list."""
        cached = self._read_tag_cache()
        tags = await self._fetch_current_tags(cached[0] if cached is not None else None)
        await self._write_tag_cache(tags)
        return tags

    async def _fetch_current_tags(self, cached_tags: list[str] | None = None) -> list[str]:
        """Fetches tags newest-first until a cached tag is found."""
        offset = 0
        limit = 100
        all_tags = list(cached_tags or [])
        cached_tag_set = set(cached_tags or [])
        while True:
            params = {"query": "sort:creation-time", "offset": offset, "limit": limit}
            data = await self._request("GET", "tags", params=params)
            tags = data.get("results", [])
            if not tags:
                break

            found_cached_tag = False
            for tag in tags:
                names = tag["names"]
                all_tags.extend(names)
                if cached_tag_set.intersection(names):
                    found_cached_tag = True

            if found_cached_tag:
                break
            offset += limit
        return list(dict.fromkeys(all_tags))

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
        response = await self.session.get(url)
        response.raise_for_status()
        return response.content

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

        return await self._repeated_post_update(post_id, version, payload)

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
        if self.http_impersonate:
            from curl_cffi import CurlMime

            multipart = CurlMime()
            multipart.addpart(name="content", data=thumbnail_data, content_type="image/jpeg")
            try:
                token_resp = await self._request("POST", "uploads", multipart=multipart)
            finally:
                multipart.close()
        else:
            buffer_io = BytesIO(thumbnail_data)
            token_resp = await self._request(
                "POST", "uploads", files={"content": (None, buffer_io, "image/jpeg")}
            )
        payload = {"version": version, "thumbnailToken": token_resp["token"]}
        await self._repeated_post_update(post_id, version, payload)

    async def batch_create_tags(self, tags: list[str], category: str):
        if not tags:
            return
        created_tags = []
        for tag in tags:
            try:
                await self.create_tag(tag, category)
                created_tags.append(tag)
                print(f"Created tag '{tag}' in category '{category}'.")
            except Exception as e:
                print(f"Error creating tag '{tag}': {e}")
        cached = self._read_tag_cache()
        if cached is not None and created_tags:
            cached_tags, fetched_at = cached
            await self._write_tag_cache(cached_tags + created_tags, fetched_at=fetched_at)
