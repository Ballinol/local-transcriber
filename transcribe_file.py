# -*- coding: utf-8 -*-
"""
transcribe_file.py — видео/аудио → текстовый файл. Локально, faster-whisper.
Без лимита длины (сегменты стримятся), бесплатно, офлайн, на GPU если есть.

Форматы входа: mp4, mkv, webm, mp3, wav, m4a, ... (декод через PyAV, ffmpeg не нужен).

Запуск:
    .venv\\Scripts\\python.exe transcribe_file.py "видео.mp4" [ещё файлы...]
    опции: --model large-v3-turbo|large-v3|medium  --lang ru|en|auto  --srt  --device auto|cuda|cpu
    (по умолчанию large-v3-turbo — быстрый и полный; large-v3 точнее по терминам,
     но в ~5 раз медленнее и иногда галлюцинирует вступление)
Результат: рядом с файлом создаётся <имя>.txt (и <имя>.srt при --srt).
"""
import sys, argparse, time, re
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

try:
    import numpy as _np
except Exception:
    _np = None

SR = 16000  # частота, к которой приводится аудио


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


# Схлопывает 3+ подряд одинаковых слова/фразы внутри строки (петли Whisper),
# не трогая естественные двойные повторы.
_REPEAT = re.compile(r'(\b\w+(?:\s+\w+){0,5}?)(?:\s+\1\b){2,}', re.IGNORECASE)


def _collapse_repeats(text):
    return _REPEAT.sub(r'\1', text)


def _transcribe_array(model, wav, lang):
    """Транскрибирует один numpy-массив аудио. Возвращает (список сегментов, info)."""
    segs, info = model.transcribe(
        wav, language=lang, beam_size=5, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        initial_prompt=INITIAL_PROMPT, hotwords=HOTWORDS,
        condition_on_previous_text=True,
    )
    return list(segs), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="файлы видео/аудио")
    ap.add_argument("--model", default="large-v3-turbo", help="large-v3-turbo (быстрый, полный) | large-v3 (точнее термины, медленнее) | medium")
    ap.add_argument("--lang", default="auto", help="ru / en / auto (автоопределение)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--compute-type", default="auto")
    ap.add_argument("--srt", action="store_true", help="также сохранить субтитры .srt с таймкодами")
    ap.add_argument("--chunk-min", type=int, default=20,
                    help="длинные файлы бьются на куски по N мин (стабильность на 1ч+)")
    args = ap.parse_args()

    lang = None if args.lang == "auto" else args.lang
    print(f"Загружаю faster-whisper {args.model}...")
    model, dev = load_whisper(args.model, args.device, args.compute_type)
    print(f"Устройство: {dev}")

    chunk_len = max(1, args.chunk_min) * 60 * SR

    for inp in args.inputs:
        p = Path(inp)
        if not p.exists():
            print(f"[!] нет файла: {inp}"); continue
        print(f"\n=== {p.name} ===")
        t0 = time.perf_counter()
        print("декодирую аудио...", flush=True)
        wav = decode_audio(str(p), sampling_rate=SR)
        total = len(wav) / SR

        # длинные файлы — по чанкам (иначе VAD на многочасовом аудио может зависнуть)
        if len(wav) > chunk_len + 120 * SR:
            starts = list(range(0, len(wav), chunk_len))
            print(f"длительность: {total/60:.1f} мин → режим длинного файла: "
                  f"{len(starts)} чанков по {args.chunk_min} мин")
        else:
            starts = [0]
            print(f"длительность: {total/60:.1f} мин — транскрибирую...")

        txt_path = p.with_suffix(".txt")
        srt_path = p.with_suffix(".srt")
        n = 0
        prev_line = None  # для схлопывания подряд-идущих одинаковых строк (артефакт зацикливания)
        with open(txt_path, "w", encoding="utf-8") as ftxt, \
             (open(srt_path, "w", encoding="utf-8") if args.srt else _Null()) as fsrt:
            for ci, a in enumerate(starts):
                off = a / SR
                segs, info = _transcribe_array(model, wav[a:a + chunk_len], lang)
                if ci == 0:
                    print(f"язык: {info.language} ({info.language_probability:.0%})")
                for seg in segs:
                    st, en = seg.start + off, seg.end + off
                    text = _collapse_repeats(seg.text.strip())  # гасим петли внутри строки
                    if not text or text == prev_line:  # пропускаем пустое и подряд-повторы
                        continue
                    prev_line = text
                    ftxt.write(text + "\n")
                    if args.srt:
                        n += 1
                        fsrt.write(f"{n}\n{fmt_ts(st)} --> {fmt_ts(en)}\n{text}\n\n")
                ftxt.flush()  # чтобы прогресс был виден сразу
                if len(starts) > 1:
                    print(f"  чанк {ci+1}/{len(starts)} готов "
                          f"(~{min((a+chunk_len)/SR, total)/60:.0f} мин)", flush=True)
        dt = time.perf_counter() - t0
        speed = (total / dt) if dt else 0
        print(f"[ok] {txt_path.name}" + (f" + {srt_path.name}" if args.srt else "")
              + f"  ({dt:.0f}с, x{speed:.1f} быстрее реального времени)")


class _Null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
