"""Extract 20:25 <= player time < 20:40 from the left 502m view on Lumen."""
import csv
import math
from pathlib import Path
from tempfile import TemporaryDirectory


VIDEO = Path("/mnt/hatter/MPI_data/MaxPlanckData/ruffs_2021/ruffs2021lek3_502m/cut_Lek3_left 2021_502_m.mp4")
PILOT = Path(__file__).resolve().parent / "pilots/502m_left_20m25s_20m40s"


def extract(cv, video, pilot, start=1225000, end=1240000):
    target = pilot / "frames"
    if target.exists():
        raise ValueError(f"Output already exists; nothing overwritten: {target}")
    pilot.mkdir(parents=True, exist_ok=True)
    cap = cv.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        with TemporaryDirectory(dir=pilot) as tmp:
            stage = Path(tmp) / "frames"
            stage.mkdir()
            origin = previous = None
            count = idx = 0
            with (stage / "source_frames.csv").open("w", newline="", encoding="utf-8") as f:
                rows = csv.writer(f)
                rows.writerow(["file", "source_video", "source_index_zero_based", "elapsed_ms"])
                while cap.grab():
                    stamp = cap.get(cv.CAP_PROP_POS_MSEC)
                    if not math.isfinite(stamp) or (previous is not None and stamp <= previous):
                        raise ValueError(f"Video timestamps not strictly increasing at frame {idx}; no output published")
                    if origin is None:
                        origin = stamp
                    elapsed = stamp - origin
                    previous = stamp
                    if idx % 1000 == 0:
                        print(f"Reading player time {elapsed/1000:.1f} s / {end/1000:g} s", flush=True)
                    if elapsed >= end:
                        break
                    if elapsed >= start:
                        ok, frame = cap.retrieve()
                        name = f"{count:06d}.jpg"
                        if not ok or not cv.imwrite(str(stage / name), frame, [cv.IMWRITE_JPEG_QUALITY, 98]):
                            raise ValueError(f"Cannot extract source frame {idx}")
                        rows.writerow([name, video.name, idx, f"{elapsed:.6f}"])
                        count += 1
                    idx += 1
                else:
                    raise ValueError("Video reading stopped before the pilot end; no output published")
            if not count:
                raise ValueError("No frames in requested window")
            stage.rename(target)
    finally:
        cap.release()
    return count, target


if __name__ == "__main__":
    import cv2
    count, folder = extract(cv2, VIDEO, PILOT)
    print(f"Pilot ready: {count} frames. Open {folder / '000000.jpg'}")
