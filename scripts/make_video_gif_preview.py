"""Create a lightweight GIF preview from the annotated video result."""

from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    input_video = project_root / "video_infer_result.mp4"
    output_gif = project_root / "video_infer_result.gif"

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    frames: list[Image.Image] = []
    frame_index = 0
    sample_step = 6
    max_frames = 45
    target_width = 640

    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % sample_step == 0:
            height, width = frame.shape[:2]
            target_height = int(height * target_width / width)
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        frame_index += 1

    capture.release()
    if not frames:
        raise RuntimeError("No frames were extracted for GIF preview.")

    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=True,
    )
    print(f"saved {output_gif}")
    print(f"frames={len(frames)}")


if __name__ == "__main__":
    main()
