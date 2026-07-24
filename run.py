import mlx_whisper
from pathlib import Path
import sys
import time
import subprocess
import tempfile
import math

# Расширения аудио/видео, которые поддерживает Whisper
MEDIA_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".mpeg", ".mpg",
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".oga", ".opus", ".mpga", ".m4b",
}

CHUNK_SEC = 10 * 60  # 10 минут
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


def get_media_duration_sec(media_path):
    """Длительность медиа в секундах через ffprobe. Возвращает None при ошибке."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(media_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def extract_chunk_to_wav(media_path, start_sec, length_sec, out_wav_path):
    """Извлекает фрагмент в 16kHz mono WAV через ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(start_sec), "-i", str(media_path),
            "-t", str(length_sec), "-ar", "16000", "-ac", "1",
            "-vn", str(out_wav_path),
        ],
        capture_output=True,
        check=True,
    )


def get_media_files(folder_path):
    """Возвращает список медиа-файлов в папке (без подпапок)."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return []
    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS
    )


def transcribe_media(media_path):
    """Транскрибирует медиа по чанкам, дописывает текст в реальном времени, показывает прогресс и ETA."""
    path = Path(media_path)
    if not path.exists():
        print(f"⚠️ Файл не найден: {media_path}")
        return False

    duration_sec = get_media_duration_sec(path)
    if duration_sec is None:
        print(f"⚠️ Не удалось получить длительность (нужны ffprobe/ffmpeg): {path.name}")
        return False

    n_chunks = max(1, math.ceil(duration_sec / CHUNK_SEC))
    output_file = path.parent / f"{path.stem}_transcript.txt"

    print(f"🎬 Обработка: {path.name} ({duration_sec / 60:.1f} мин, {n_chunks} чанков по ~{CHUNK_SEC // 60} мин)")
    print("⏳ Загрузка модели...")
    start_time = time.time()

    try:
        # Файл создаём сразу, пишем по чанкам
        with open(output_file, "w", encoding="utf-8") as out_f:
            for i in range(n_chunks):
                start_sec = i * CHUNK_SEC
                length_sec = min(CHUNK_SEC, duration_sec - start_sec)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    extract_chunk_to_wav(path, start_sec, length_sec, tmp_path)
                    result = mlx_whisper.transcribe(
                        tmp_path,
                        path_or_hf_repo=WHISPER_MODEL,
                    )
                    text = (result.get("text") or "").strip()
                    if text:
                        out_f.write(text)
                        if i < n_chunks - 1:
                            out_f.write("\n")
                    out_f.flush()
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

                elapsed = time.time() - start_time
                done = i + 1
                pct = int(100 * done / n_chunks)
                eta_sec = (elapsed / done) * (n_chunks - done) if done < n_chunks else 0
                eta_str = f"ETA {eta_sec / 60:.1f} мин" if eta_sec > 0 else ""
                print(f"   Чанк {done}/{n_chunks} ({pct}%) — {eta_str}")

        total = time.time() - start_time
        print(f"✅ Готово: {output_file.name}")
        print(f"⏱️ Время: {total / 60:.1f} мин\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка ffmpeg: {e}\n")
        return False
    except Exception as e:
        print(f"❌ Ошибка при обработке {path.name}: {e}\n")
        return False


def main():
    if len(sys.argv) < 2:
        print("📝 Использование:")
        print(f"   python {Path(__file__).name} <путь_к_папке>")
        print("\n💡 Скрипт найдёт все аудио/видео в папке, транскрибирует по очереди")
        print("   и сохранит .txt с тем же именем в эту же папку.")
        print("\n   Пример:")
        print(f"   python {Path(__file__).name} /Users/name/Videos/lectures")
        sys.exit(1)

    folder_path = Path(sys.argv[1]).resolve()
    if not folder_path.is_dir():
        print(f"❌ Ошибка: '{folder_path}' не является папкой или не существует.")
        sys.exit(1)

    media_files = get_media_files(folder_path)
    if not media_files:
        print(f"📂 В папке нет медиа-файлов (поддерживаются: {', '.join(sorted(MEDIA_EXTENSIONS))})")
        sys.exit(0)

    print(f"📂 Папка: {folder_path}")
    print(f"📋 Найдено файлов: {len(media_files)}\n")
    ok = 0
    for f in media_files:
        if transcribe_media(f):
            ok += 1
    print(f"✅ Обработано: {ok}/{len(media_files)}")
    sys.exit(0 if ok == len(media_files) else 1)


if __name__ == "__main__":
    main()
