import argparse
from pathlib import Path

from dubpipeline.utils.audio_process import MuxMode, mux_smart


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Слияние видео и русской озвучки (full WAV) с помощью ffmpeg"
    )
    p.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Путь к исходному видео (mp4/mkv/...)",
    )
    p.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Путь к русской аудиодорожке (ru_full.wav)",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Путь к выходному видеофайлу",
    )
    p.add_argument(
        "--mode",
        choices=["replace", "add", "rus_first"],
        default="replace",
        help=(
            "Режим: replace=заменить оригинальную аудиодорожку, "
            "add=добавить русскую как вторую, "
            "rus_first=добавить русскую и сделать первой"
        ),
    )
    p.add_argument(
        "--ffmpeg",
        type=str,
        default="ffmpeg",
        help="Имя/путь к бинарнику ffmpeg (по умолчанию 'ffmpeg' в PATH)",
    )
    p.add_argument(
        "--orig-lang",
        type=str,
        default="eng",
        help="Код языка оригинальной аудиодорожки (для метаданных, только для mode=add)",
    )
    return p.parse_args()


def resolve_mux_mode(mode: str) -> MuxMode:
    mode = (mode or "").strip().lower()
    if mode == "replace":
        return MuxMode.REPLACE
    if mode == "add":
        return MuxMode.ADD
    if mode == "rus_first":
        return MuxMode.RUS_FIRST
    raise ValueError(f"Unknown mux mode: {mode}")


def main() -> None:
    args = parse_args()

    video = args.video
    audio = args.audio
    out_path = args.out

    if not video.exists():
        raise SystemExit(f"Video file not found: {video}")
    if not audio.exists():
        raise SystemExit(f"Audio file not found: {audio}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Video: {video}")
    print(f"[INFO] Audio (RU full): {audio}")
    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] Mode: {args.mode}")

    mux_mode = resolve_mux_mode(args.mode)
    mux_smart(
        video,
        audio,
        out_path,
        mode=mux_mode,
        ffmpeg=args.ffmpeg,
        orig_lang=args.orig_lang,
    )


if __name__ == "__main__":
    main()
