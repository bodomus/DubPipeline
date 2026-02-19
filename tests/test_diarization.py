from __future__ import annotations

import copy

from dubpipeline.steps import step_whisperx


def _aligned_result() -> dict:
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"start": 0.0, "end": 0.5, "word": "hello"},
                    {"start": 0.5, "end": 1.0, "word": "world"},
                ],
            }
        ]
    }


def _all_speaker_00(result: dict) -> bool:
    for seg in result.get("segments", []):
        if seg.get("speaker") != "SPEAKER_00":
            return False
        for word in seg.get("words", []):
            if word.get("speaker") != "SPEAKER_00":
                return False
    return True


def test_diarization_disabled_returns_fallback(monkeypatch):
    monkeypatch.setenv("DUBPIPELINE_DIARIZATION", "0")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    result = step_whisperx.run_diarization_safe("audio.wav", copy.deepcopy(_aligned_result()))

    assert _all_speaker_00(result)


def test_diarization_enabled_without_token_returns_fallback(monkeypatch):
    monkeypatch.setenv("DUBPIPELINE_DIARIZATION", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    result = step_whisperx.run_diarization_safe("audio.wav", copy.deepcopy(_aligned_result()))

    assert _all_speaker_00(result)


def test_diarization_pipeline_init_failure_returns_fallback(monkeypatch):
    monkeypatch.setenv("DUBPIPELINE_DIARIZATION", "1")
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")

    class FailingPipeline:
        def __init__(self, *args, **kwargs):
            raise Exception("401 unauthorized")

    monkeypatch.setattr(step_whisperx.whisperx, "DiarizationPipeline", FailingPipeline, raising=False)

    result = step_whisperx.run_diarization_safe("audio.wav", copy.deepcopy(_aligned_result()))

    assert _all_speaker_00(result)


def test_diarization_success(monkeypatch):
    monkeypatch.setenv("DUBPIPELINE_DIARIZATION", "1")
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")

    class OkPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, _audio):
            return [{"speaker": "SPEAKER_01", "start": 0.0, "end": 1.0}]

    def fake_assign_word_speakers(_diar_segments, aligned):
        enriched = copy.deepcopy(aligned)
        for seg in enriched["segments"]:
            seg["speaker"] = "SPEAKER_01"
            for word in seg["words"]:
                word["speaker"] = "SPEAKER_01"
        return enriched

    monkeypatch.setattr(step_whisperx.whisperx, "DiarizationPipeline", OkPipeline, raising=False)
    monkeypatch.setattr(step_whisperx.whisperx, "assign_word_speakers", fake_assign_word_speakers)

    result = step_whisperx.run_diarization_safe("audio.wav", copy.deepcopy(_aligned_result()))

    assert result["segments"][0]["speaker"] == "SPEAKER_01"


def test_diarization_waveform_typeerror_retries_with_path(monkeypatch):
    monkeypatch.setenv("DUBPIPELINE_DIARIZATION", "1")
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")

    class RetryPipeline:
        def __init__(self, *args, **kwargs):
            self.calls = []

        def __call__(self, audio_input):
            self.calls.append(audio_input)
            if len(self.calls) == 1:
                raise TypeError("waveform not supported")
            return [{"speaker": "SPEAKER_01", "start": 0.0, "end": 1.0}]

    class AudioRef:
        path = "audio.wav"

    def fake_assign_word_speakers(_diar_segments, aligned):
        enriched = copy.deepcopy(aligned)
        enriched["segments"][0]["speaker"] = "SPEAKER_01"
        for word in enriched["segments"][0]["words"]:
            word["speaker"] = "SPEAKER_01"
        return enriched

    monkeypatch.setattr(step_whisperx.whisperx, "DiarizationPipeline", RetryPipeline, raising=False)
    monkeypatch.setattr(step_whisperx.whisperx, "assign_word_speakers", fake_assign_word_speakers)

    result = step_whisperx.run_diarization_safe(AudioRef(), copy.deepcopy(_aligned_result()))

    assert result["segments"][0]["speaker"] == "SPEAKER_01"
