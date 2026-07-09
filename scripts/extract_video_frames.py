"""Extract representative frames from 1.mp4 for video inference debugging."""

from __future__ import annotations

from pathlib import Path

import cv2


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    input_video = project_root / "1.mp4"
    output_dir = project_root / "video_debug_frames"
    output_dir.mkdir(exist_ok=True)

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"total={total_frames}, fps={fps:.2f}, size={width}x{height}")

    frame_indexes = [
        0,
        max(total_frames // 4, 0),
        max(total_frames // 2, 0),
        max(total_frames * 3 // 4, 0),
        max(total_frames - 1, 0),
    ]

    for order, frame_index in enumerate(frame_indexes, start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        output_path = output_dir / f"frame_{order}_{frame_index}.jpg"
        cv2.imwrite(str(output_path), frame)
        print(f"saved {output_path}")

    capture.release()


if __name__ == "__main__":
    main()
