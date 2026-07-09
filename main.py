import asyncio
import random
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import orjson
from blacksheep import Application, FromJSON, FromQuery, accepted, get, json, post, status_code
from pydantic import BaseModel

from pixtaggers.camiedetect import MODEL_PATH as CAMIE_MODEL_PATH
from pixtaggers.camiedetect import CamieSession
from pixtaggers.cldetect import MODEL_PATH as CL_MODEL_PATH
from pixtaggers.cldetect import ClTaggerSession
from pixtaggers.commondetect import BaseTaggerSession
from pixtaggers.discordhook import DiscordHook
from pixtaggers.im_sess import Image
from pixtaggers.img_helpers import ModelThreshold, resize_by_longest_side
from pixtaggers.schema import Config, ModelName, SimpleSnapshot
from pixtaggers.szurubooru import SimplePost, SzurubooruClient
from pixtaggers.video_frames import extract_frames_from_video

ROOT_DIR = Path(__file__).parent.resolve()
config = ROOT_DIR / "config.json"
config_data = Config.from_json(orjson.loads(config.read_text(encoding="utf-8")))
GLOBAL_TAGS = set()  # Cache of existing tags to minimize API calls
BG_CHECK = re.compile(r"_?background$", re.IGNORECASE)

app = Application()


def init_tagger_session(model: ModelName, threshold: ModelThreshold, top_k: int) -> BaseTaggerSession:
    match model:
        case "camie-tagger-v2":
            return CamieSession(CAMIE_MODEL_PATH, threshold, top_k)
        case "cl-tagger-v2":
            return ClTaggerSession(CL_MODEL_PATH, threshold, top_k)
        case _:
            raise ValueError(f"Unsupported tagger model: {model}")


@app.lifespan
async def lifespan():
    global GLOBAL_TAGS

    model_threshold = ModelThreshold(
        config_data.threshold.general,
        config_data.threshold.characters,
        config_data.threshold.media,
        config_data.threshold.rating,
    )

    szuru_session = SzurubooruClient(config_data.szuru.host, config_data.szuru.user, config_data.szuru.token)
    print("Fetching available tags from Szurubooru...")
    GLOBAL_TAGS = set(await szuru_session.get_current_tags())
    app.services.register(SzurubooruClient, instance=szuru_session)

    webhook_svc = DiscordHook(config_data.discord_url, host_urL=config_data.szuru.host)
    app.services.register(DiscordHook, instance=webhook_svc)
    print(f"Registering ONNX client ({config_data.model})...")
    async with init_tagger_session(config_data.model, model_threshold, config_data.threshold.top_k) as session:
        app.services.register(BaseTaggerSession, instance=session)
        yield  # noqa: ASYNC119

    await szuru_session.close()


def find_missing_tags(tags: list[str], current_tags: set[str]) -> list[str]:
    return [tag for tag in tags if tag not in current_tags]


def merge_tags(*tag_lists: list[str]) -> list[str]:
    merged = set()
    for tag_list in tag_lists:
        merged.update(tag_list)
    return list(merged)


def sanitize_tags(tags: list[str]) -> list[str]:
    # If has alpha_transparency, remove any "background" related tags since
    # they are likely inaccurate and not useful for search
    if "alpha_transparency" in tags:
        tags = [tag for tag in tags if not BG_CHECK.search(tag)]
    return tags


async def maybe_work_on_alpha_thumbs(post: SimplePost, client: SzurubooruClient, bytes_data: bytes) -> bool:
    print(f"Post ID {post['id']} has alpha transparency, retrying thumbnail generation...")
    img = Image.open(BytesIO(bytes_data))
    if img.mode not in ("RGBA", "LA", "PA"):
        print("Image does not have an alpha channel. Skipping.")
        return True

    # Make white background
    ret_img = Image.new("RGBA", img.size, "white")
    ret_img.paste(img, (0, 0), mask=img)
    ret_img = ret_img.convert("RGB")  # Back to RGB

    # Resize
    thumb_img = resize_by_longest_side(ret_img, config_data.thumbnails.target_size)
    # Save as RGB24 PNG
    output_buffer = BytesIO()
    await asyncio.to_thread(
        lambda: thumb_img.save(output_buffer, format="JPEG", quality=95, optimize=True, progressive=True)
    )
    thumb_bytes = output_buffer.getvalue()

    print(f"Updating thumbnail for post ID {post['id']}...")
    try:
        await client.update_thumbnail(post["id"], version=post["version"], thumbnail_data=thumb_bytes)
        print(f"Thumbnail for post ID {post['id']} updated successfully.")
        return True
    except Exception as e:
        print(f"Error updating thumbnail for post ID {post['id']}: {e}")
        return False


async def maybe_upload_video_frame_as_thumbnail(post: SimplePost, client: SzurubooruClient, video_bytes: bytes) -> bool:
    img = Image.open(BytesIO(video_bytes))

    thumb_img = resize_by_longest_side(img, config_data.thumbnails.target_size)
    output_buffer = BytesIO()
    await asyncio.to_thread(
        lambda: thumb_img.save(output_buffer, format="JPEG", quality=95, optimize=True, progressive=True)
    )
    thumb_bytes = output_buffer.getvalue()
    print(f"Updating video thumbnail for post ID {post['id']}...")
    try:
        await client.update_thumbnail(post["id"], version=post["version"], thumbnail_data=thumb_bytes)
        print(f"Video thumbnail for post ID {post['id']} updated successfully.")
        return True
    except Exception as e:
        print(f"Error updating video thumbnail for post ID {post['id']}: {e}")
        return False


async def work_auto_tag_process(
    post_id: str, client: SzurubooruClient, tagger_session: BaseTaggerSession, discord: DiscordHook
) -> bool:
    global GLOBAL_TAGS

    print(f"Starting auto-tag process for post ID: {post_id}")
    try:
        post_id_int = int(post_id)
    except ValueError:
        print(f"Invalid post ID: {post_id}")
        return False

    try:
        post_data = await client.get_post(post_id_int)
    except Exception as e:
        print(f"Error fetching post data for ID {post_id}: {e}")
        await discord.report_error(post_id_int, f"Error fetching post data: {e}")
        return False
    print(f"Processing post data for ID {post_id} (v{post_data['version']}), kind: {post_data['kind']}")
    match post_data["kind"]:
        case "image":
            try:
                downloaded_image = await client.download_image(post_data["image_url"])
                print(f"Downloaded image for post ID {post_id}, size: {len(downloaded_image)} bytes")
            except Exception as e:
                print(f"Error downloading image for post ID {post_id}: {e}")
                await discord.report_error(post_id_int, f"Error downloading image: {e}")
                return False

            print("Running detection model...")
            try:
                tags_to_add = await tagger_session.detect(downloaded_image)
                print(f"Model suggested {tags_to_add.count()} tags for post ID {post_id}, rating {tags_to_add.rating}")
            except Exception as e:
                print(f"Error running detection model for post ID {post_id}: {e}")
                await discord.report_error(post_id_int, f"Error running detection model: {e}")
                return False
            if not config_data.tagging_enable.general:
                tags_to_add.general = []
            if not config_data.tagging_enable.media:
                tags_to_add.media = []
            if not config_data.tagging_enable.characters:
                tags_to_add.characters = []
            if not config_data.tagging_enable.meta:
                tags_to_add.meta = []
            if not config_data.tagging_enable.rating:
                tags_to_add.rating = None

            try:
                await client.batch_create_tags(
                    find_missing_tags(tags_to_add.general, GLOBAL_TAGS), config_data.tagging_map.general
                )
                await client.batch_create_tags(
                    find_missing_tags(tags_to_add.media, GLOBAL_TAGS), config_data.tagging_map.media
                )
                await client.batch_create_tags(
                    find_missing_tags(tags_to_add.characters, GLOBAL_TAGS), config_data.tagging_map.characters
                )
                await client.batch_create_tags(
                    find_missing_tags(tags_to_add.meta, GLOBAL_TAGS), config_data.tagging_map.meta
                )
            except Exception as e:
                print(f"Error creating missing tags: {e}")
                await discord.report_error(post_id_int, f"Error creating missing tags: {e}")
                return False

            merged_tags = merge_tags(tags_to_add.general, tags_to_add.media, tags_to_add.characters, tags_to_add.meta)
            merged_tags = sanitize_tags(merged_tags)
            GLOBAL_TAGS.update(tags_to_add.general)
            GLOBAL_TAGS.update(tags_to_add.media)
            GLOBAL_TAGS.update(tags_to_add.characters)
            GLOBAL_TAGS.update(tags_to_add.meta)
            if not merged_tags:
                print(f"No new tags to add for post ID {post_id}. Skipping update.")
                return True

            merged_tags = merge_tags(merged_tags, post_data["tags"])
            new_post = None
            try:
                print(f"Updating post ID {post_id} with automated tags {len(merged_tags)} tags...")
                new_post = await client.update_post(
                    post_id_int, version=post_data["version"], tags=merged_tags, safety=tags_to_add.rating
                )
                print(f"Post ID {post_id} updated successfully with new tags.")
            except Exception as e:
                print(f"Error updating post with new tags: {e}")
                await discord.report_error(post_id_int, f"Error updating post with new tags: {e}")
                return False

            if new_post is not None and "generated" in new_post["thumbnail_url"] and config_data.thumbnails.alpha_fix:
                return await maybe_work_on_alpha_thumbs(new_post, client, downloaded_image)
            return True
        case "video" | "animation":
            meta_new_tags = ["animated"]
            if post_data["kind"] == "video":
                meta_new_tags.append("video")
            general_new_tags = []
            media_new_tags = []
            characters_new_tags = []
            detected_rating = None

            presel_thumb_frame = None
            try:
                ugoira_data = await client.download_image(post_data["image_url"])
                print(f"Downloaded media for post ID {post_id}, size: {len(ugoira_data)} bytes")

                frames = extract_frames_from_video(
                    video_data=ugoira_data, num_frames=config_data.thumbnails.video.extract
                )
                img_counter = 0
                for frame_bytes in frames:
                    if presel_thumb_frame is None:
                        presel_thumb_frame = frame_bytes
                    try:
                        print(f"Running detection model for a video frame in post ID {post_id} (frame {img_counter})")
                        frame_tags = await tagger_session.detect(frame_bytes)
                        meta_new_tags.extend(frame_tags.meta)
                        general_new_tags.extend(frame_tags.general)
                        media_new_tags.extend(frame_tags.media)
                        characters_new_tags.extend(frame_tags.characters)
                        detected_rating = frame_tags.rating
                    except Exception as e:
                        print(f"Error running detection model for a video frame in post ID {post_id}: {e}")
                        return False
                    img_counter += 1
                    if img_counter >= config_data.thumbnails.video.detect:
                        break
                if presel_thumb_frame is None and frames:
                    presel_thumb_frame = frames[0]
            except Exception as e:
                print(f"Error processing video for post ID {post_id}: {e}")
                await discord.report_error(post_id_int, f"Error processing video: {e}")
                return False

            if not config_data.tagging_enable.general:
                general_new_tags = []
            if not config_data.tagging_enable.media:
                media_new_tags = []
            if not config_data.tagging_enable.characters:
                characters_new_tags = []
            if not config_data.tagging_enable.meta:
                meta_new_tags = []
            if not config_data.tagging_enable.rating:
                detected_rating = None

            try:
                await client.batch_create_tags(
                    find_missing_tags(general_new_tags, GLOBAL_TAGS), config_data.tagging_map.general
                )
                await client.batch_create_tags(
                    find_missing_tags(media_new_tags, GLOBAL_TAGS), config_data.tagging_map.media
                )
                await client.batch_create_tags(
                    find_missing_tags(characters_new_tags, GLOBAL_TAGS), config_data.tagging_map.characters
                )
                await client.batch_create_tags(
                    find_missing_tags(meta_new_tags, GLOBAL_TAGS), config_data.tagging_map.meta
                )
            except Exception as e:
                print(f"Error creating missing tags: {e}")
                await discord.report_error(post_id_int, f"Error creating missing tags: {e}")
                return False

            merged_tags = merge_tags(general_new_tags, media_new_tags, characters_new_tags, meta_new_tags)
            merged_tags = sanitize_tags(merged_tags)
            GLOBAL_TAGS.update(general_new_tags)
            GLOBAL_TAGS.update(media_new_tags)
            GLOBAL_TAGS.update(characters_new_tags)
            GLOBAL_TAGS.update(meta_new_tags)
            if not merged_tags:
                print(f"No new tags to add for post ID {post_id}. Skipping update.")
                return True

            merged_tags = merge_tags(merged_tags, post_data["tags"])

            new_post = None
            try:
                print(f"Updating post ID {post_id} with automated tags {len(merged_tags)} tags...")
                new_post = await client.update_post(
                    post_id_int, version=post_data["version"], tags=merged_tags, safety=detected_rating
                )
                print(f"Post ID {post_id} updated successfully with new tags.")
            except Exception as e:
                print(f"Error updating post with new tags: {e}")
                await discord.report_error(post_id_int, f"Error updating post with new tags: {e}")
                return False

            if (
                new_post is not None and
                "generated" in new_post["thumbnail_url"] and
                presel_thumb_frame is not None and
                config_data.thumbnails.video.enabled
            ):
                return await maybe_upload_video_frame_as_thumbnail(new_post, client, presel_thumb_frame)
            return True
        case "ugoira":
            # Ugoira is a zip file of images, sample the same amount of frames from thumbnails.video.extract
            meta_new_tags = ["animated", "ugoira"]
            general_new_tags = []
            media_new_tags = []
            characters_new_tags = []
            detected_rating = None

            try:
                ugoira_data = await client.download_image(post_data["image_url"])
                print(f"Downloaded media for post ID {post_id}, size: {len(ugoira_data)} bytes")

                # Open with zip file and read the animation.json
                frames = []
                with ZipFile(BytesIO(ugoira_data)) as zip_file:
                    animation = orjson.loads(zip_file.read("animation.json").decode("utf-8"))
                    # select random amount
                    raw_frames = random.sample(animation["frames"], config_data.thumbnails.video.extract)
                    for frame in raw_frames:
                        frame_bytes = zip_file.read(frame["file"])
                        frames.append(frame_bytes)

                img_counter = 0
                for frame_bytes in frames:
                    try:
                        print(f"Running detection model for a ugoira frame in post ID {post_id} (frame {img_counter})")
                        frame_tags = await tagger_session.detect(frame_bytes)
                        meta_new_tags.extend(frame_tags.meta)
                        general_new_tags.extend(frame_tags.general)
                        media_new_tags.extend(frame_tags.media)
                        characters_new_tags.extend(frame_tags.characters)
                        detected_rating = frame_tags.rating
                    except Exception as e:
                        print(f"Error running detection model for a ugoira frame in post ID {post_id}: {e}")
                        return False
                    img_counter += 1
                    if img_counter >= config_data.thumbnails.video.detect:
                        break
            except Exception as e:
                print(f"Error processing ugoira for post ID {post_id}: {e}")
                await discord.report_error(post_id_int, f"Error processing ugoira: {e}")
                return False

            if not config_data.tagging_enable.general:
                general_new_tags = []
            if not config_data.tagging_enable.media:
                media_new_tags = []
            if not config_data.tagging_enable.characters:
                characters_new_tags = []
            if not config_data.tagging_enable.meta:
                meta_new_tags = []
            if not config_data.tagging_enable.rating:
                detected_rating = None

            try:
                await client.batch_create_tags(
                    find_missing_tags(general_new_tags, GLOBAL_TAGS), config_data.tagging_map.general
                )
                await client.batch_create_tags(
                    find_missing_tags(media_new_tags, GLOBAL_TAGS), config_data.tagging_map.media
                )
                await client.batch_create_tags(
                    find_missing_tags(characters_new_tags, GLOBAL_TAGS), config_data.tagging_map.characters
                )
                await client.batch_create_tags(
                    find_missing_tags(meta_new_tags, GLOBAL_TAGS), config_data.tagging_map.meta
                )
            except Exception as e:
                print(f"Error creating missing tags: {e}")
                await discord.report_error(post_id_int, f"Error creating missing tags: {e}")
                return False

            merged_tags = merge_tags(general_new_tags, media_new_tags, characters_new_tags, meta_new_tags)
            merged_tags = sanitize_tags(merged_tags)
            GLOBAL_TAGS.update(general_new_tags)
            GLOBAL_TAGS.update(media_new_tags)
            GLOBAL_TAGS.update(characters_new_tags)
            GLOBAL_TAGS.update(meta_new_tags)
            if not merged_tags:
                print(f"No new tags to add for post ID {post_id}. Skipping update.")
                return True

            merged_tags = merge_tags(merged_tags, post_data["tags"])

            new_post = None
            try:
                print(f"Updating post ID {post_id} with automated tags {len(merged_tags)} tags...")
                new_post = await client.update_post(
                    post_id_int, version=post_data["version"], tags=merged_tags, safety=detected_rating
                )
                print(f"Post ID {post_id} updated successfully with new tags.")
            except Exception as e:
                print(f"Error updating post with new tags: {e}")
                await discord.report_error(post_id_int, f"Error updating post with new tags: {e}")
                return False
        case _:
            print(f"Unsupported post kind '{post_data['kind']}' for post ID {post_id}. Skipping.")
            return True


async def work_auto_tag_process_multiple(
    post_ids: list[int], client: SzurubooruClient, tagger_session: BaseTaggerSession, discord: DiscordHook
):
    # Rather than all of them, do one by one
    failure = 0
    for post_id in post_ids:
        success = await work_auto_tag_process(str(post_id), client, tagger_session, discord)
        if not success:
            failure += 1
    print(f"Completed auto-tag process for post IDs: {', '.join(str(pid) for pid in post_ids)}")
    await discord.report_batch_complete(post_ids, failure, len(post_ids))


@get("/")
def hello():
    return json({"status": "ok"})


@post("/webhooks")
def handle_webhook(
    tagger_session: BaseTaggerSession,
    client: SzurubooruClient,
    discord: DiscordHook,
    data: FromJSON[dict],
    t: FromQuery[str],
):
    payload = data.value
    snapshot = SimpleSnapshot(
        id=payload.get("resource_id", ""),
        operation=payload.get("operation", "unknown"),
        type=payload.get("resource_type", "unknown"),
    )
    if config_data.key != t.value:
        return status_code(401, "Unauthorized")

    if snapshot.type != "post":
        return accepted("ignored resource_type other than 'post'")
    if snapshot.operation != "created":
        return accepted("ignored operation other than 'created'")

    # queue the tagging process in background
    asyncio.create_task(work_auto_tag_process(snapshot.id, client, tagger_session, discord))  # noqa: RUF006

    return accepted()


class RangedIdsModel(BaseModel):
    start: int
    end: int


class ManualTagUpdateRequestModel(BaseModel):
    id: int | list[int] | RangedIdsModel

    def into_ranged_ids(self) -> list[int]:
        if isinstance(self.id, int):
            return [self.id]
        elif isinstance(self.id, list):
            return self.id
        elif isinstance(self.id, RangedIdsModel):
            return list(range(self.id.start, self.id.end + 1))
        else:
            raise ValueError("Invalid 'id' format. Must be int, list of ints, or RangedIds.")


@post("/tag")
def manual_tag_update(
    tagger_session: BaseTaggerSession,
    client: SzurubooruClient,
    discord: DiscordHook,
    data: FromJSON[ManualTagUpdateRequestModel],
    t: FromQuery[str],
):
    payload = data.value
    all_post_ids = payload.into_ranged_ids()
    if not all_post_ids:
        return status_code(400, "Missing 'post_id' in request body")
    if config_data.key != t.value:
        return status_code(401, "Unauthorized")

    asyncio.create_task(work_auto_tag_process_multiple(all_post_ids, client, tagger_session, discord))  # noqa: RUF006

    return accepted("Tag update process started for post IDs: " + ", ".join(str(pid) for pid in all_post_ids))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=23810, lifespan="on")
