from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


# =========================================
# CONFIGURACIÓN
# =========================================
INPUT_DIRS = [Path("videos_nc"), Path("videos_crash")]
OUTPUT_DIR = Path("output_videos")

DAY_DIR = OUTPUT_DIR / "day"
NIGHT_DIR = OUTPUT_DIR / "night"
SKIPPED_DIR = OUTPUT_DIR / "skipped"

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# Duración de clips
FIXED_CLIP_SECONDS = 12       # cada clip principal durará 12 segundos
MIN_LAST_CLIP_SECONDS = 10    # guardar el último clip solo si dura al menos 10 s
MAX_LAST_CLIP_SECONDS = 15    # guardar el último clip solo si dura como máximo 15 s

# Día / noche
BRIGHTNESS_THRESHOLD = 95     # ajusta si ves errores de clasificación
SAMPLE_FRAMES = 12            # frames a muestrear por video


# =========================================
# UTILIDADES
# =========================================
def check_dependencies() -> None:
    """Verifica que ffmpeg y ffprobe estén instalados."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise EnvironmentError(
            "No se encontró ffmpeg/ffprobe.\n"
            "Instálalo con:\n"
            "brew install ffmpeg"
        )


def ensure_output_folders() -> None:
    """Crea carpetas de salida."""
    DAY_DIR.mkdir(parents=True, exist_ok=True)
    NIGHT_DIR.mkdir(parents=True, exist_ok=True)
    SKIPPED_DIR.mkdir(parents=True, exist_ok=True)

    for input_dir in INPUT_DIRS:
        (DAY_DIR / input_dir.name).mkdir(parents=True, exist_ok=True)
        (NIGHT_DIR / input_dir.name).mkdir(parents=True, exist_ok=True)


def get_video_files(folder: Path) -> list[Path]:
    """Devuelve videos soportados dentro de una carpeta."""
    if not folder.exists():
        return []

    return sorted(
        [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )


def get_duration_seconds(video_path: Path) -> float:
    """Obtiene la duración del video con ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    return float(result.stdout.strip())


def classify_day_or_night(
    video_path: Path,
    sample_frames: int = SAMPLE_FRAMES,
    brightness_threshold: float = BRIGHTNESS_THRESHOLD,
) -> tuple[str, float]:
    """
    Clasifica el video como day o night según brillo promedio.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"No se pudieron leer frames: {video_path}")

    indices = np.linspace(
        0,
        max(total_frames - 1, 0),
        num=min(sample_frames, total_frames),
        dtype=int,
    )

    brightness_values = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(np.mean(gray)))

    cap.release()

    if not brightness_values:
        raise ValueError(f"No se pudieron muestrear frames: {video_path}")

    avg_brightness = float(np.mean(brightness_values))
    label = "day" if avg_brightness >= brightness_threshold else "night"
    return label, avg_brightness


def build_segments(duration: float) -> list[tuple[float, float]]:
    """
    Crea segmentos:
    - clips fijos de 12 segundos
    - el último se guarda solo si dura entre 10 y 15 segundos
    """
    segments: list[tuple[float, float]] = []

    if duration < MIN_LAST_CLIP_SECONDS:
        return segments

    start = 0.0

    while start + FIXED_CLIP_SECONDS <= duration:
        segments.append((start, FIXED_CLIP_SECONDS))
        start += FIXED_CLIP_SECONDS

    remainder = duration - start
    if MIN_LAST_CLIP_SECONDS <= remainder <= MAX_LAST_CLIP_SECONDS:
        segments.append((start, remainder))

    return segments


def extract_segment(video_path: Path, start: float, duration: float, output_path: Path) -> None:
    """Extrae un clip usando ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output_path),
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )


def process_video(video_path: Path, source_folder_name: str) -> dict:
    """
    Procesa un video:
    - clasifica day/night
    - corta en segmentos
    - guarda en carpeta de salida correcta
    """
    label, avg_brightness = classify_day_or_night(video_path)
    duration = get_duration_seconds(video_path)
    segments = build_segments(duration)

    if not segments:
        skipped_output = SKIPPED_DIR / video_path.name
        return {
            "file": video_path.name,
            "source_folder": source_folder_name,
            "label": label,
            "avg_brightness": round(avg_brightness, 2),
            "duration_seconds": round(duration, 2),
            "clips_created": 0,
            "status": f"skipped_too_short (< {MIN_LAST_CLIP_SECONDS}s)",
            "output_folder": str(skipped_output.parent),
        }

    target_root = DAY_DIR if label == "day" else NIGHT_DIR
    target_dir = target_root / source_folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for i, (start, seg_duration) in enumerate(segments, start=1):
        output_name = f"{video_path.stem}_clip_{i:03d}.mp4"
        output_path = target_dir / output_name
        extract_segment(video_path, start, seg_duration, output_path)

    return {
        "file": video_path.name,
        "source_folder": source_folder_name,
        "label": label,
        "avg_brightness": round(avg_brightness, 2),
        "duration_seconds": round(duration, 2),
        "clips_created": len(segments),
        "status": "ok",
        "output_folder": str(target_dir),
    }


def save_log(rows: list[dict], output_csv: Path) -> None:
    """Guarda log en CSV."""
    if not rows:
        return

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "source_folder",
                "label",
                "avg_brightness",
                "duration_seconds",
                "clips_created",
                "status",
                "output_folder",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    check_dependencies()
    ensure_output_folders()

    all_videos: list[tuple[Path, str]] = []

    for input_dir in INPUT_DIRS:
        if not input_dir.exists():
            print(f"Advertencia: no existe la carpeta {input_dir.resolve()}")
            continue

        videos = get_video_files(input_dir)
        for video in videos:
            all_videos.append((video, input_dir.name))

    if not all_videos:
        print("No se encontraron videos en videos_nc ni en videos_crash.")
        return

    print(f"Se encontraron {len(all_videos)} videos.\n")

    results: list[dict] = []

    for video_path, source_folder_name in all_videos:
        print(f"Procesando: {video_path.name}  |  carpeta: {source_folder_name}")
        try:
            result = process_video(video_path, source_folder_name)
            results.append(result)
            print(
                f"  -> {result['status']} | "
                f"{result['label']} | "
                f"brillo={result['avg_brightness']} | "
                f"clips={result['clips_created']}"
            )
        except Exception as e:
            error_row = {
                "file": video_path.name,
                "source_folder": source_folder_name,
                "label": "unknown",
                "avg_brightness": "",
                "duration_seconds": "",
                "clips_created": 0,
                "status": f"error: {str(e)}",
                "output_folder": "",
            }
            results.append(error_row)
            print(f"  -> error: {e}")

    save_log(results, OUTPUT_DIR / "processing_log.csv")

    print("\nListo.")
    print(f"Salida: {OUTPUT_DIR.resolve()}")
    print(f"Log: {(OUTPUT_DIR / 'processing_log.csv').resolve()}")


if __name__ == "__main__":
    main()