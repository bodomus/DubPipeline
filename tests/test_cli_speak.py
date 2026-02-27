from __future__ import annotations

import wave
from pathlib import Path

import pytest

from dubpipeline import cli


def _write_wav(path: Path, frames: int = 1000, sr: int = 22050) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * frames)


def test_speak_parse_text_and_out_audio() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["speak", "--text", "Привет", "--out-audio", "out.wav"])
    assert args.command == "speak"
    assert args.text == "Привет"


def test_speak_parse_text_file_and_out_audio(tmp_path: Path) -> None:
    parser = cli.build_parser()
    text_file = tmp_path / "book.txt"
    text_file.write_text("Текст", encoding="utf-8")
    args = parser.parse_args(["speak", "--text-file", str(text_file), "--out-audio", "out.wav"])
    assert args.text_file == str(text_file)


def test_speak_requires_text_or_file() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["speak", "--out-audio", "out.wav"])


def test_speak_rejects_both_text_and_file(tmp_path: Path) -> None:
    parser = cli.build_parser()
    text_file = tmp_path / "book.txt"
    text_file.write_text("Текст", encoding="utf-8")
    with pytest.raises(SystemExit):
        parser.parse_args([
            "speak",
            "--text",
            "A",
            "--text-file",
            str(text_file),
            "--out-audio",
            "out.wav",
        ])


def test_speak_plan_does_not_call_real_tts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["speak", "--text", "Привет мир", "--out-audio", str(tmp_path / "out.wav"), "--plan"])

    called = {"tts": False}

    def fake_synthesize(*_args, **_kwargs):
        called["tts"] = True
        return [tmp_path / "out_segments" / "seg_0001.wav"]

    monkeypatch.setattr(cli, "synthesize_segments_to_wavs", fake_synthesize)
    cli._run_speak(args)
    assert called["tts"]
    assert not (tmp_path / "out.wav").exists()


def test_speak_smoke_concat_with_mock_tts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parser = cli.build_parser()
    out_audio = tmp_path / "final.wav"
    args = parser.parse_args(["speak", "--text", "Один. Два.", "--out-audio", str(out_audio)])

    def fake_synthesize(segments, _cfg, out_dir: Path, **_kwargs):
        result = []
        for i, _ in enumerate(segments, start=1):
            wav = out_dir / f"seg_{i:04d}.wav"
            _write_wav(wav)
            result.append(wav)
        return result

    monkeypatch.setattr(cli, "synthesize_segments_to_wavs", fake_synthesize)
    cli._run_speak(args)

    assert out_audio.exists()
    with out_audio.open("rb") as f:
        assert f.read(4) == b"RIFF"
