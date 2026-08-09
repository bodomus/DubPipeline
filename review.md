# Review: DUB-77 technical tokens and TTS numeric normalization

Scope reviewed:
- Technical token protection/restoration in `dubpipeline/text/technical_tokens.py`
- Russian-only TTS text normalization in `dubpipeline/text/tts_normalizer.py`
- Translation integration in `dubpipeline/steps/step_translate.py`
- TTS integration in `dubpipeline/steps/step_tts_core.py`
- Regression/unit coverage in `tests/test_text_technical_tokens.py`

Findings:
- No blockers found in the DUB-77 diff.
- Protected numeric/technical tokens are translated as deterministic placeholders and restored before `text_tgt` is persisted.
- Placeholder restoration tolerates safe translator damage such as case changes and inserted spaces/underscores/hyphens.
- If a translator fully drops a placeholder, the affected segment falls back to the original source text with exact technical values and emits a warning instead of stopping the whole pipeline.
- TTS normalization is language-aware: Russian numeric forms are expanded before synthesis, non-Russian text is left unchanged.
- Stored translated text remains separate from the spoken TTS text; segment dictionaries are not mutated by synthesis.

Validation:
- Passed: `.\\.venv\\Scripts\\python.exe -m pytest tests/test_text_technical_tokens.py tests/test_translation_models.py::TranslationStepIntegrationTests::test_translate_step_uses_translation_model_from_config tests/test_tts_provider_import.py::TtsProviderImportTests::test_segment_synthesis_prefers_translated_text -q`
  - Latest result after missing-placeholder fallback: 18 passed, 1 warning.
- Full project command attempted: `.\\.venv\\Scripts\\python.exe -m pytest -q`
  - Failed during collection in `tools/test_whisperx_basic.py` because the local `whisperx` object is a `types.SimpleNamespace` without `load_model`.
- Test package command attempted: `.\\.venv\\Scripts\\python.exe -m pytest tests -q`
  - 116 passed.
  - 2 existing failures remain outside the DUB-77 diff:
    - `tests/test_audio_mix_step.py::AudioMixStepTests::test_external_voice_skips_extract_audio_step`
    - `tests/test_tts_provider_import.py::TtsProviderImportTests::test_coqui_provider_without_package_has_clear_error`

Known limitations:
- The Russian number normalizer intentionally handles only focused technical numeric forms required by DUB-77, not a full linguistic number system.
- Source-text fallback preserves technical values but leaves the affected segment untranslated when the model completely removes a placeholder.
- Version-like decimals are protected during translation. In TTS normalization, obvious version contexts such as `Version`, `Engine`, `Unreal Engine`, and `Версия` are left unchanged to avoid over-pronouncing product/version text.
