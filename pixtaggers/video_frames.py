from io import BytesIO

import av
import numpy as np
from av.container import InputContainer
from PIL import Image


def _is_solid_color_frame(image: Image.Image, threshold: float = 10.0) -> bool:
    """Check if a frame is a solid block of color by measuring pixel variance."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    return float(np.std(arr)) < threshold


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

    total = len(keyframes)
    # Pick evenly-spaced indices across the keyframe list
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = [round(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]

    extracted: list[bytes] = []
    for idx in indices:
        pil_image: Image.Image = keyframes[idx].to_image()

        if _is_solid_color_frame(pil_image):
            continue

        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        extracted.append(buf.getvalue())

    return extracted
