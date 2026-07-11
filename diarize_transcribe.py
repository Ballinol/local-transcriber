# -*- coding: utf-8 -*-
"""
diarize_transcribe.py — транскрипция + разбивка по спикерам (кто говорит).
Локально, бесплатно, офлайн, без HuggingFace-токена.

Движок: faster-whisper (текст) + speechbrain ECAPA (голосовые эмбеддинги) +
агломеративная кластеризация (sklearn) для группировки реплик по спикерам.

Запуск:
    .venv\\Scripts\\python.exe diarize_transcribe.py "видео.mp4" [--speakers 4]
      --speakers N  : сколько человек говорит (если знаешь). По умолчанию — авто.
      --model / --lang : как в transcribe_file.py
Результат: рядом с файлом <имя>.speakers.txt  (реплики вида "Спикер 1: ...").
"""
import sys, argparse, time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from faster_whisper.audio import decode_audio

from transcribe_file import INITIAL_PROMPT, HOTWORDS, load_whisper  # переиспользуем

SR = 16000
ECAPA = "speechbrain/spkrec-ecapa-voxceleb"


def _load_ecapa(device):
    from speechbrain.inference.speaker import EncoderClassifier
    kw = {}
    try:  # Windows запрещает symlink без прав админа — просим копировать
        from speechbrain.utils.fetching import LocalStrategy
        kw["local_strategy"] = LocalStrategy.COPY
    except Exception:
        pass
    return EncoderClassifier.from_hparams(
        source=ECAPA, savedir="models/ecapa", run_opts={"device": device}, **kw)


def _embed(clf, wav, start, end):
    """Голосовой эмбеддинг для отрезка [start,end] (сек). Короткие расширяем до ~1.5с."""
    import torch
    a, b = int(start * SR), int(end * SR)
    if b - a < int(1.5 * SR):  # слишком коротко — берём окно вокруг центра
        c = (a + b) // 2
        a = max(0, c - int(0.75 * SR)); b = min(len(wav), c + int(0.75 * SR))
    seg = wav[a:b]
    if len(seg) < SR // 2:  # совсем крохи — не эмбеддим
        return None
    with torch.no_grad():
        t = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)
        emb = clf.encode_batch(t).squeeze().cpu().numpy()
    return emb


def _cluster(embs, n_speakers):
    from sklearn.cluster import AgglomerativeClustering
    X = np.vstack(embs)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)  # L2-норм → косинус
    if n_speakers and n_speakers > 0:
        cl = AgglomerativeClustering(n_clusters=n_speakers, metric="cosine",
                                     linkage="average")
    else:
        cl = AgglomerativeClustering(n_clusters=None, distance_threshold=0.7,
                                     metric="cosine", linkage="average")
    return cl.fit_predict(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--speakers", type=int, default=0, help="кол-во спикеров (0=авто)")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    lang = None if args.lang == "auto" else args.lang
    # ECAPA (torch): GPU только если torch реально видит CUDA, иначе CPU
    try:
        import torch
        ecapa_dev = "cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu"
    except Exception:
        ecapa_dev = "cpu"

    print(f"Загружаю Whisper {args.model} + ECAPA...")
    model, wdev = load_whisper(args.model, args.device, "auto")  # авто-откат на CPU
    clf = _load_ecapa(ecapa_dev)
    print(f"Устройство: Whisper={wdev}, ECAPA={ecapa_dev}")

    for inp in args.inputs:
        p = Path(inp)
        if not p.exists():
            print(f"[!] нет файла: {inp}"); continue
        print(f"\n=== {p.name} ===")
        t0 = time.perf_counter()
        wav = decode_audio(str(p), sampling_rate=SR)
        segments, info = model.transcribe(
            str(p), language=lang, beam_size=5, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            initial_prompt=INITIAL_PROMPT, hotwords=HOTWORDS,
            condition_on_previous_text=True,
        )
        print(f"язык: {info.language} ({info.language_probability:.0%}), "
              f"{info.duration/60:.1f} мин — транскрибирую + считаю голоса...")

        segs, embs = [], []
        for s in segments:
            txt = s.text.strip()
            if not txt:
                continue
            e = _embed(clf, wav, s.start, s.end)
            segs.append((s.start, s.end, txt))
            embs.append(e)
            if int(s.end) % 60 < 2:
                print(f"  ...{s.end/60:.1f} мин", end="\r", flush=True)

        # у сегментов без эмбеддинга берём эмбеддинг соседа
        for i, e in enumerate(embs):
            if e is None:
                embs[i] = next((embs[j] for j in range(i, -1, -1) if embs[j] is not None),
                               next((embs[j] for j in range(i, len(embs)) if embs[j] is not None), None))
        if any(e is None for e in embs):  # вообще нет голосов
            labels = [0] * len(segs)
        else:
            labels = _cluster(embs, args.speakers)

        n_found = len(set(labels))
        # порядок появления → «Спикер 1,2,3…»
        order, name = {}, {}
        for lb in labels:
            if lb not in order:
                order[lb] = len(order) + 1
        for lb in order:
            name[lb] = f"Спикер {order[lb]}"

        out = p.with_suffix(".speakers.txt")
        with open(out, "w", encoding="utf-8") as f:
            cur, buf = None, []
            for (st, en, txt), lb in zip(segs, labels):
                who = name[lb]
                if who != cur:
                    if buf:
                        f.write(f"{cur}: {' '.join(buf)}\n\n")
                    cur, buf = who, [txt]
                else:
                    buf.append(txt)
            if buf:
                f.write(f"{cur}: {' '.join(buf)}\n\n")

        dt = time.perf_counter() - t0
        print(f"\n[ok] {out.name}  (спикеров: {n_found}, {dt:.0f}с, "
              f"x{info.duration/dt:.1f} realtime)")


if __name__ == "__main__":
    main()
