# -*- coding: utf-8 -*-
"""
transcribe_file.py — видео/аудио → текстовый файл. Локально, faster-whisper.
Без лимита длины (сегменты стримятся), бесплатно, офлайн, на GPU если есть.

Форматы входа: mp4, mkv, webm, mp3, wav, m4a, ... (декод через PyAV, ffmpeg не нужен).

Запуск:
    .venv\\Scripts\\python.exe transcribe_file.py "видео.mp4" [ещё файлы...]
    опции: --model large-v3-turbo|medium|small  --lang ru|en|auto  --srt  --device auto|cuda|cpu
Результат: рядом с файлом создаётся <имя>.txt (и <имя>.srt при --srt).
"""
import sys, argparse, time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from faster_whisper import WhisperModel

try:
    import numpy as _np
except Exception:
    _np = None


def load_whisper(name, device="auto", compute_type="auto"):
    """Грузит Whisper с авто-откатом на CPU, если GPU/cuDNN недоступны.
    Возвращает (model, фактическое_устройство)."""
    order = ["cpu"] if device == "cpu" else ["cuda", "cpu"]
    last = None
    for dev in order:
        ct = compute_type if compute_type != "auto" else ("float16" if dev == "cuda" else "int8")
        try:
            m = WhisperModel(name, device=dev, compute_type=ct)
            if _np is not None:  # мини-прогон: провоцирует загрузку CUDA-библиотек
                segs, _ = m.transcribe(_np.zeros(8000, dtype="float32"), beam_size=1)
                list(segs)
            return m, dev
        except Exception as e:
            last = e
            if dev != order[-1]:
                print(f"[i] {dev} недоступен ({type(e).__name__}), перехожу на CPU...")
    raise last

# Пунктуированный образец + термины: заставляет Whisper ставить точки/запятые/
# заглавные и правильнее писать имена собственные (GitLab, Kubernetes и т.п.).
INITIAL_PROMPT = (
    "Это запись интервью. Расшифровка ведётся с пунктуацией и заглавными буквами. "
    "Обсуждаем Kubernetes, GitLab CI, Jenkins, Terraform, Ansible, Docker, Helm, "
    "Argo CD, Nginx, Ingress, etcd, kube-proxy, kubelet, Nexus, Harbor, "
    "HashiCorp Vault, network policy, микросервисы, кластер, облако Cloud.ru."
)
# Термины для смещения распознавания (faster-whisper hotwords).
HOTWORDS = (
    "Kubernetes GitLab CI Jenkins Terraform Ansible Docker Helm Argo CD "
    "Nginx Ingress etcd kube-proxy kubelet Nexus Harbor Vault network policy"
)


def fmt_ts(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60)
    s = int(sec % 60); ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="файлы видео/аудио")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--lang", default="auto", help="ru / en / auto (автоопределение)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--compute-type", default="auto")
    ap.add_argument("--srt", action="store_true", help="также сохранить субтитры .srt с таймкодами")
    args = ap.parse_args()

    lang = None if args.lang == "auto" else args.lang
    print(f"Загружаю faster-whisper {args.model}...")
    model, dev = load_whisper(args.model, args.device, args.compute_type)
    print(f"Устройство: {dev}")

    for inp in args.inputs:
        p = Path(inp)
        if not p.exists():
            print(f"[!] нет файла: {inp}"); continue
        print(f"\n=== {p.name} ===")
        t0 = time.perf_counter()
        segments, info = model.transcribe(
            str(p), language=lang, beam_size=5, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            initial_prompt=INITIAL_PROMPT, hotwords=HOTWORDS,
            condition_on_previous_text=True,
        )
        print(f"язык: {info.language} ({info.language_probability:.0%}), "
              f"длительность: {info.duration/60:.1f} мин — транскрибирую...")
        txt_path = p.with_suffix(".txt")
        srt_path = p.with_suffix(".srt")
        n = 0
        with open(txt_path, "w", encoding="utf-8") as ftxt, \
             (open(srt_path, "w", encoding="utf-8") if args.srt else _Null()) as fsrt:
            for seg in segments:
                text = seg.text.strip()
                ftxt.write(text + "\n")
                if args.srt:
                    n += 1
                    fsrt.write(f"{n}\n{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n{text}\n\n")
                # прогресс по ходу (файл может быть длинным)
                if int(seg.end) % 30 < 2:
                    print(f"  ...{seg.end/60:.1f} мин", end="\r", flush=True)
        dt = time.perf_counter() - t0
        speed = (info.duration / dt) if dt else 0
        print(f"\n[ok] {txt_path.name}" + (f" + {srt_path.name}" if args.srt else "")
              + f"  ({dt:.0f}с, x{speed:.1f} быстрее реального времени)")


class _Null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
