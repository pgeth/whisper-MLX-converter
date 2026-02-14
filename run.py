import mlx_whisper
from pathlib import Path
import sys
import time

# Расширения аудио/видео, которые поддерживает Whisper
MEDIA_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".mpeg", ".mpg",
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".oga", ".opus", ".mpga", ".m4b",
}

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
    """Транскрибирует один медиа-файл и сохраняет .txt в ту же папку."""
    path = Path(media_path)
    if not path.exists():
        print(f"⚠️ Файл не найден: {media_path}")
        return False

    print(f"🎬 Обработка: {path.name}")
    print("⏳ Загрузка модели и транскрибация...")
    start_time = time.time()

    try:
        result = mlx_whisper.transcribe(
            str(path),
            path_or_hf_repo='mlx-community/whisper-large-v3-turbo'
        )
        duration = time.time() - start_time

        # Сохраняем в ту же папку, имя как у файла + _transcript.txt
        output_file = path.parent / f"{path.stem}_transcript.txt"
        output_file.write_text(result["text"], encoding="utf-8")

        print(f"✅ Готово: {output_file.name}")
        print(f"⏱️ Время: {duration:.1f} сек\n")
        return True
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
