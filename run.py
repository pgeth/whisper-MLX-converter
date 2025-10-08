import mlx_whisper
from pathlib import Path
import sys
import time

def transcribe_video(video_path):
  if not Path(video_path).exists():
    print(f" Ошибка: файл '{video_path}' не найден!")
    return

  print(f"🎬 Обработка файла: {video_path}")
  print("⏳ Загрузка модели и транскрибация...")

  start_time = time.time()

  try:
    result = mlx_whisper.transcribe(
      video_path,
      path_or_hf_repo='mlx-community/whisper-large-v3-turbo'
    )

    end_time = time.time()
    duration = end_time - start_time

    video_name = Path(video_path).stem  # имя файла без расширения
    script_dir = Path(__file__).parent.resolve()  # resolve() даёт абсолютный путь
    output_file = script_dir / f"{video_name}_transcript.txt"

    with open(output_file, 'w', encoding='utf-8' ) as f:
      f.write(result['text'])

    print(f"✅ Готово!")
    print(f"📄 Текст сохранён: {output_file}")
    print(f"⏱️ Время транскрипции: {duration:.1f} сек")

  except Exception as e:
    print(f"❌ Ошибка при обработке: {e}")

def main():
    if len(sys.argv) < 2:
        print("📝 Использование:")
        print(f"   python {Path(__file__).name} <путь_к_видео>")
        print("\n💡 Примеры:")
        print(f"   python {Path(__file__).name} video.mp4")
        print(f"   python {Path(__file__).name} /Users/name/Videos/lecture.mov")
        sys.exit(1)
    
    video_path = sys.argv[1]
    transcribe_video(video_path)

if __name__ == "__main__":
    main()
