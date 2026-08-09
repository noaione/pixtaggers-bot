from io import BytesIO

import av
import numpy as np
from av.container import InputContainer

from .im_sess import Image


def _is_solid_color_frame(image: Image.Image, threshold: float = 10.0) -> bool:
    """Check if a frame is a solid block of color by measuring pixel variance."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    return float(np.std(arr)) < threshold


def _sample_indices(total: int, num_frames: int) -> list[int]:
    if total <= 0 or num_frames <= 0:
        return []
    if total <= num_frames:
        return list(range(total))
    if num_frames == 1:
        return [0]
    return [round(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]


def _encode_frame(image: Image.Image) -> bytes:
    frame = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    buf = BytesIO()
    frame.save(buf, format="PNG")
    return buf.getvalue()


def extract_frames_from_animation(animation_data: bytes, num_frames: int = 5) -> list[bytes]:
    """Extract evenly-spaced frames from Pillow-supported animated images."""
    with Image.open(BytesIO(animation_data)) as image:
        frame_count = getattr(image, "n_frames", 1)
        extracted: list[bytes] = []

        for index in _sample_indices(frame_count, num_frames):
            image.seek(index)
            frame = image.copy()
            if _is_solid_color_frame(frame):
                continue
            extracted.append(_encode_frame(frame))

        return extracted


def extract_frames_from_video(video_data: bytes, num_frames: int = 5) -> list[bytes]:
    """Extract evenly-spaced frames from video bytes, skipping solid-color frames.

    Returns a list of PNG-encoded frame bytes suitable for detection functions.
    """
    container = av.open(BytesIO(video_data))
    if not isinstance(container, InputContainer) or not container.streams.video:
        container.close()
        return []
    stream = container.streams.video[0]
    stream.codec_context.skip_frame = "NONKEY"

    # Collect all keyframes first so we know total count
    keyframes: list[av.VideoFrame] = []
    for frame in container.decode(stream):
        keyframes.append(frame)

    container.close()

    if not keyframes:
        return []

    indices = _sample_indices(len(keyframes), num_frames)

    extracted: list[bytes] = []
    for idx in indices:
        pil_image: Image.Image = keyframes[idx].to_image()

        if _is_solid_color_frame(pil_image):
            continue

        extracted.append(_encode_frame(pil_image))

    return extracted
