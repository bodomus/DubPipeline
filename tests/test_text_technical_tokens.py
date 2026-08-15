from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from dubpipeline.config import PipelineConfig
from dubpipeline.translation.service import TranslationModelError
from dubpipeline.steps import step_translate
from dubpipeline.steps.step_tts_core import synthesize_segments_to_wavs
from dubpipeline.text.technical_tokens import TechnicalTokenError, TechnicalTokenProtector
from dubpipeline.text.tts_normalizer import normalize_text_for_tts


RU_ZERO = "\u043d\u043e\u043b\u044c"
RU_ONE = "\u043e\u0434\u0438\u043d"
RU_TWO = "\u0434\u0432\u0430"
RU_THREE = "\u0442\u0440\u0438"
RU_FIVE = "\u043f\u044f\u0442\u044c"
RU_MINUS = "\u043c\u0438\u043d\u0443\u0441"
RU_DECIMAL = "\u0437\u0430\u043f\u044f\u0442\u0430\u044f"
RU_TIMES_POWER = (
    "\u0443\u043c\u043d\u043e\u0436\u0438\u0442\u044c \u043d\u0430 "
    "\u0434\u0435\u0441\u044f\u0442\u044c \u0432 \u0441\u0442\u0435\u043f\u0435\u043d\u0438"
)


def test_technical_token_protector_preserves_decimal_exactly() -> None:
    protector = TechnicalTokenProtector()

    protected = protector.protect("value 0.0000001")
    restored = protector.restore(f"\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 {protected.text.split()[-1]}", protected.tokens)

    assert restored == "\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 0.0000001"


def test_technical_token_protector_preserves_negative_decimal_and_percent() -> None:
    protector = TechnicalTokenProtector()

    protected = protector.protect("set -0.25 and then 50%")
    restored = protector.restore(
        f"\u0443\u0441\u0442\u0430\u043d\u043e\u0432\u0438 {protected.text}",
        protected.tokens,
    )

    assert "-0.25" in restored
    assert "50%" in restored


def test_technical_token_protector_preserves_resolution_and_scientific_notation() -> None:
    protector = TechnicalTokenProtector()

    protected = protector.protect("resolution 1920x1080 value 1e-05")
    restored = protector.restore(protected.text, protected.tokens)

    assert "1920x1080" in restored
    assert "1e-05" in restored


def test_technical_token_protector_detects_corrupted_placeholder() -> None:
    protector = TechnicalTokenProtector()

    protected = protector.protect("value 0.0000001")
    corrupted = protected.text.replace("DUBTECHTOKEN0000", "DUBTECHTOKEN9999")

    with pytest.raises(TechnicalTokenError):
        protector.restore(corrupted, protected.tokens)


def test_technical_token_protector_restores_safe_placeholder_variants() -> None:
    protector = TechnicalTokenProtector()

    protected = protector.protect("value 0.0000001")
    damaged_but_recognizable = protected.text.replace("DUBTECHTOKEN0000", "dub tech token 0000")

    restored = protector.restore(damaged_but_recognizable, protected.tokens)

    assert restored == "value 0.0000001"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "0,00001",
            f"{RU_ZERO} {RU_DECIMAL} {RU_ZERO} {RU_ZERO} {RU_ZERO} {RU_ZERO} {RU_ONE}",
        ),
        (
            "0.0000001",
            f"{RU_ZERO} {RU_DECIMAL} {RU_ZERO} {RU_ZERO} {RU_ZERO} {RU_ZERO} {RU_ZERO} {RU_ZERO} {RU_ONE}",
        ),
        ("-0.25", f"{RU_MINUS} {RU_ZERO} {RU_DECIMAL} {RU_TWO} {RU_FIVE}"),
        ("1e-05", f"{RU_ONE} {RU_TIMES_POWER} {RU_MINUS} {RU_FIVE}"),
        ("2.5e+03", f"{RU_TWO} {RU_DECIMAL} {RU_FIVE} {RU_TIMES_POWER} {RU_THREE}"),
    ],
)
def test_russian_tts_normalizes_numeric_forms(source: str, expected: str) -> None:
    assert normalize_text_for_tts(source, "ru") == expected


def test_tts_normalizer_leaves_other_languages_unchanged() -> None:
    assert normalize_text_for_tts("value 0.0000001", "en") == "value 0.0000001"


def test_tts_normalizer_regression_for_observed_decimal_shape() -> None:
    original = (
        "\u0421\u0438\u043b\u0430 \u043d\u0430\u043a\u043b\u043e\u043d\u0430 "
        "\u044d\u0442\u043e \u0431\u044b\u043b \u043f\u0440\u043e\u0441\u0442\u043e "
        "\u043f\u043e \u0443\u043c\u043e\u043b\u0447\u0430\u043d\u0438\u044e "
        "\u044f \u043f\u043e\u0439\u0434\u0443 0,00001 "
        "\u0438 \u044f \u043d\u0430 \u0441\u0430\u043c\u043e\u043c "
        "\u0434\u0435\u043b\u0435 \u0438\u0437\u043c\u0435\u043d\u044e "
        "\u044d\u0442\u043e \u043d\u0430 a"
    )

    normalized = normalize_text_for_tts(original, "ru")

    assert "0,00001" not in normalized
    assert "1e-05" not in normalized
    assert f"{RU_ZERO} {RU_DECIMAL} {RU_ZERO} {RU_ZERO} {RU_ZERO} {RU_ZERO} {RU_ONE}" in normalized
    assert original.endswith("a")


def test_translate_step_preserves_protected_value_even_if_translator_would_change_numbers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "segments.json"
    output_path = tmp_path / "segments.ru.json"
    input_path.write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "value 0.0000001"}], ensure_ascii=False),
        encoding="utf-8",
    )

    cfg = PipelineConfig(project_name="sample", project_dir=tmp_path)
    cfg.paths.segments_file = input_path
    cfg.paths.segments_tgt_file = output_path
    cfg.languages.tgt = "ru"

    class FakeTranslator:
        cache_scope = "fake|en->ru"

        def translate_texts(self, texts: list[str], *, sent_fallback: bool = True) -> list[str]:
            assert texts == ["value DUBTECHTOKEN0000"]
            return ["\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 DUBTECHTOKEN0000"]

    def in_memory_cache(_db_path: Path) -> sqlite3.Connection:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE IF NOT EXISTS translations (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        return con

    monkeypatch.setattr(step_translate, "_open_cache", in_memory_cache)

    step_translate.translate_segments(cfg, input_path, output_path, FakeTranslator())

    translated = json.loads(output_path.read_text(encoding="utf-8"))
    assert translated[0]["text_tgt"] == "\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 0.0000001"
    assert "0,00001" not in translated[0]["text_tgt"]


def test_translate_step_retries_when_translator_drops_placeholder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "segments.json"
    output_path = tmp_path / "segments.ru.json"
    input_path.write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "value 0.0000001"}], ensure_ascii=False),
        encoding="utf-8",
    )

    cfg = PipelineConfig(project_name="sample", project_dir=tmp_path)
    cfg.paths.segments_file = input_path
    cfg.paths.segments_tgt_file = output_path
    cfg.languages.src = "en"
    cfg.languages.tgt = "ru"

    class FakeTranslator:
        cache_scope = "fake|en->ru"
        calls: list[tuple[list[str], bool]] = []

        def translate_texts(self, texts: list[str], *, sent_fallback: bool = True) -> list[str]:
            self.calls.append((texts, sent_fallback))
            assert texts == ["value DUBTECHTOKEN0000"]
            if len(self.calls) == 1:
                return ["\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435"]
            return ["\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 DUBTECHTOKEN0000"]

    def in_memory_cache(_db_path: Path) -> sqlite3.Connection:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE IF NOT EXISTS translations (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        return con

    monkeypatch.setattr(step_translate, "_open_cache", in_memory_cache)
    translator = FakeTranslator()

    step_translate.translate_segments(cfg, input_path, output_path, translator)

    translated = json.loads(output_path.read_text(encoding="utf-8"))
    assert translator.calls == [
        (["value DUBTECHTOKEN0000"], True),
        (["value DUBTECHTOKEN0000"], False),
    ]
    assert translated[0]["text_tgt"] == "\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 0.0000001"
    assert translated[0]["text_ru"] == "\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 0.0000001"
    assert "0,00001" not in translated[0]["text_tgt"]


def test_translate_step_fails_when_placeholder_retry_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "segments.json"
    output_path = tmp_path / "segments.ru.json"
    input_path.write_text(
        json.dumps([{"id": 12, "start": 0.0, "end": 1.0, "text": "value 0.0000001"}], ensure_ascii=False),
        encoding="utf-8",
    )

    cfg = PipelineConfig(project_name="sample", project_dir=tmp_path)
    cfg.paths.segments_file = input_path
    cfg.paths.segments_tgt_file = output_path
    cfg.languages.src = "en"
    cfg.languages.tgt = "ru"

    class FakeTranslator:
        cache_scope = "fake|en->ru"

        def translate_texts(self, texts: list[str], *, sent_fallback: bool = True) -> list[str]:
            assert texts == ["value DUBTECHTOKEN0000"]
            return ["\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435"]

    def in_memory_cache(_db_path: Path) -> sqlite3.Connection:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE IF NOT EXISTS translations (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        return con

    monkeypatch.setattr(step_translate, "_open_cache", in_memory_cache)

    with pytest.raises(TranslationModelError, match="technical_token_restore_failed"):
        step_translate.translate_segments(cfg, input_path, output_path, FakeTranslator())

    assert not output_path.exists()


def test_translate_step_rejects_meaningful_source_text_as_ru_translation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = "In this tutorial, we are going to create a river in Unreal Engine."
    input_path = tmp_path / "segments.json"
    output_path = tmp_path / "segments.ru.json"
    input_path.write_text(
        json.dumps([{"id": 0, "start": 0.0, "end": 1.0, "text": source}], ensure_ascii=False),
        encoding="utf-8",
    )

    cfg = PipelineConfig(project_name="sample", project_dir=tmp_path)
    cfg.paths.segments_file = input_path
    cfg.paths.segments_tgt_file = output_path
    cfg.languages.src = "en"
    cfg.languages.tgt = "ru"

    class FakeTranslator:
        cache_scope = "fake|en->ru"

        def translate_texts(self, texts: list[str], *, sent_fallback: bool = True) -> list[str]:
            return [source]

    def in_memory_cache(_db_path: Path) -> sqlite3.Connection:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE IF NOT EXISTS translations (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        return con

    monkeypatch.setattr(step_translate, "_open_cache", in_memory_cache)

    with pytest.raises(TranslationModelError, match="target_matches_source"):
        step_translate.translate_segments(cfg, input_path, output_path, FakeTranslator())


def test_translate_step_allows_ru_translation_with_legitimate_technical_terms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = "Unreal Engine 5.7 uses Nanite, Lumen, and PCG."
    target = "\u0412 Unreal Engine DUBTECHTOKEN0000 Nanite, Lumen \u0438 PCG \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u044e\u0442\u0441\u044f \u0432\u043c\u0435\u0441\u0442\u0435."
    input_path = tmp_path / "segments.json"
    output_path = tmp_path / "segments.ru.json"
    input_path.write_text(
        json.dumps([{"id": 1, "start": 0.0, "end": 1.0, "text": source}], ensure_ascii=False),
        encoding="utf-8",
    )

    cfg = PipelineConfig(project_name="sample", project_dir=tmp_path)
    cfg.paths.segments_file = input_path
    cfg.paths.segments_tgt_file = output_path
    cfg.languages.src = "en"
    cfg.languages.tgt = "ru"

    class FakeTranslator:
        cache_scope = "fake|en->ru"

        def translate_texts(self, texts: list[str], *, sent_fallback: bool = True) -> list[str]:
            assert texts == ["Unreal Engine DUBTECHTOKEN0000 uses Nanite, Lumen, and PCG."]
            return [target]

    def in_memory_cache(_db_path: Path) -> sqlite3.Connection:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE IF NOT EXISTS translations (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        return con

    monkeypatch.setattr(step_translate, "_open_cache", in_memory_cache)

    step_translate.translate_segments(cfg, input_path, output_path, FakeTranslator())

    translated = json.loads(output_path.read_text(encoding="utf-8"))
    restored_target = target.replace("DUBTECHTOKEN0000", "5.7")
    assert translated[0]["text_tgt"] == restored_target
    assert translated[0]["text_ru"] == restored_target


@pytest.mark.parametrize(("source", "target"), [("and", "\u0438"), ("OK", "OK"), ("UE", "UE"), ("PCG", "PCG")])
def test_translate_step_allows_short_segments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: str,
    target: str,
) -> None:
    input_path = tmp_path / "segments.json"
    output_path = tmp_path / "segments.ru.json"
    input_path.write_text(
        json.dumps([{"id": 2, "start": 0.0, "end": 1.0, "text": source}], ensure_ascii=False),
        encoding="utf-8",
    )

    cfg = PipelineConfig(project_name="sample", project_dir=tmp_path)
    cfg.paths.segments_file = input_path
    cfg.paths.segments_tgt_file = output_path
    cfg.languages.src = "en"
    cfg.languages.tgt = "ru"

    class FakeTranslator:
        cache_scope = f"fake|{source}->ru"

        def translate_texts(self, texts: list[str], *, sent_fallback: bool = True) -> list[str]:
            return [target]

    def in_memory_cache(_db_path: Path) -> sqlite3.Connection:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE IF NOT EXISTS translations (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        return con

    monkeypatch.setattr(step_translate, "_open_cache", in_memory_cache)

    step_translate.translate_segments(cfg, input_path, output_path, FakeTranslator())

    translated = json.loads(output_path.read_text(encoding="utf-8"))
    assert translated[0]["text_tgt"] == target


def test_tts_rejects_source_text_fallback_for_ru_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = PipelineConfig(project_name="sample", project_dir=tmp_path)
    cfg.languages.src = "en"
    cfg.languages.tgt = "ru"
    cfg.paths.tts_segments_dir = tmp_path / "tts"

    def fail_load_tts(*_args, **_kwargs):
        raise AssertionError("TTS must not load for invalid translated text")

    monkeypatch.setattr("dubpipeline.steps.step_tts_core._load_tts", fail_load_tts)

    source = "In this tutorial, we are going to create a river in Unreal Engine."
    segment = {"id": "83", "start": 0.0, "end": 1.0, "text": source, "text_tgt": source}
    with pytest.raises(RuntimeError, match="Invalid translated text"):
        synthesize_segments_to_wavs([segment], cfg, cfg.paths.tts_segments_dir, show_progress=False)


def test_tts_synthesis_normalizes_text_without_mutating_segment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = PipelineConfig(project_name="sample", project_dir=tmp_path)
    cfg.languages.tgt = "ru"
    cfg.paths.tts_segments_dir = tmp_path / "tts"
    captured: dict[str, object] = {}

    class FakeTts:
        speakers = ["speaker"]

        def tts_to_file(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("dubpipeline.steps.step_tts_core._load_tts", lambda *_args, **_kwargs: FakeTts())

    segment = {"id": "320", "start": 0.0, "end": 1.0, "text_tgt": "\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 0,00001"}
    synthesize_segments_to_wavs([segment], cfg, cfg.paths.tts_segments_dir, show_progress=False)

    assert captured["text"] == (
        f"\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 "
        f"{RU_ZERO} {RU_DECIMAL} {RU_ZERO} {RU_ZERO} {RU_ZERO} {RU_ZERO} {RU_ONE}"
    )
    assert segment["text_tgt"] == "\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 0,00001"


def test_tts_synthesis_rejects_unresolved_technical_placeholder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = PipelineConfig(project_name="sample", project_dir=tmp_path)
    cfg.languages.tgt = "ru"
    cfg.paths.tts_segments_dir = tmp_path / "tts"

    def fail_load_tts(*_args, **_kwargs):
        raise AssertionError("TTS must not load when placeholders are unresolved")

    monkeypatch.setattr("dubpipeline.steps.step_tts_core._load_tts", fail_load_tts)

    segment = {"id": "320", "start": 0.0, "end": 1.0, "text_tgt": "value DUBTECHTOKEN0000"}
    with pytest.raises(RuntimeError, match="Unresolved technical placeholder"):
        synthesize_segments_to_wavs([segment], cfg, cfg.paths.tts_segments_dir, show_progress=False)
