from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from dubpipeline.config import PathsConfig, PipelineConfig
from dubpipeline.steps import step_merge_py


def _make_cfg(root: Path) -> PipelineConfig:
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(project_name="sample", project_dir=root)
    cfg.paths = PathsConfig(
        workdir=root,
        out_dir=out_dir,
        input_video=root / "input.mp4",
        audio_wav=out_dir / "sample.wav",
        segments_file=out_dir / "sample.segments.json",
        segments_ru_file=out_dir / "sample.segments.ru.json",
        srt_file_en=out_dir / "sample.srt",
        tts_segments_dir=out_dir / "segments" / "tts_ru_segments",
        tts_segments_aligned_dir=out_dir / "segments" / "tts_ru_segments_aligned",
        final_video=out_dir / "sample.ru.muxed.mp4",
    )
    cfg.output.update_existing_file = True
    cfg.output.audio_update_mode = "add"
    cfg.audio_merge.mode = "hq_ducking"
    return cfg


class StepMergePyTests(unittest.TestCase):
    @staticmethod
    def _case_dir(prefix: str) -> Path:
        root = Path("tests/.tmp_runtime")
        root.mkdir(parents=True, exist_ok=True)
        case = root / f"{prefix}_{uuid4().hex}"
        case.mkdir(parents=True, exist_ok=True)
        return case

    def test_hq_ducking_update_existing_uses_temp_and_atomic_replace(self):
        root = self._case_dir("merge_py_hq")
        cfg = _make_cfg(root)
        cfg.mux.orig_lang = "fra"
        cfg.mux.target_lang = "deu"
        cfg.mux.target_track_title = "German (DubPipeline)"
        cfg.paths.input_video.write_text("video", encoding="utf-8")
        cfg.paths.audio_wav.write_text("audio", encoding="utf-8")

        mux_temp_out = root / "input.tmp.muxed.mp4"
        expected_mix_path = Path(cfg.paths.out_dir) / f"{cfg.paths.input_video.stem}.hq_mix.m4a"

        with (
            patch(
                "dubpipeline.steps.step_merge_py.merge_hq_config_from_pipeline",
                return_value=(object(), "auto"),
            ),
            patch("dubpipeline.steps.step_merge_py.render_hq_mix_audio") as render_hq_mix_audio,
            patch("dubpipeline.steps.step_merge_py.mux_smart") as mux_smart,
            patch(
                "dubpipeline.steps.step_merge_py.AtomicFileReplacer.make_temp_path",
                return_value=mux_temp_out,
            ) as make_temp_path,
            patch("dubpipeline.steps.step_merge_py.AtomicFileReplacer.replace_with_temp") as replace_with_temp,
            patch("dubpipeline.steps.step_merge_py.AtomicFileReplacer.cleanup_temp") as cleanup_temp,
        ):
            step_merge_py.run(cfg)

        make_temp_path.assert_called_once_with(cfg.paths.input_video)
        render_hq_mix_audio.assert_called_once()
        self.assertEqual(render_hq_mix_audio.call_args.kwargs["out_audio"], expected_mix_path)

        mux_call = mux_smart.call_args
        self.assertIsNotNone(mux_call)
        self.assertEqual(mux_call.args[0], cfg.paths.input_video)
        self.assertEqual(mux_call.args[1], expected_mix_path)
        self.assertEqual(mux_call.args[2], mux_temp_out)
        self.assertEqual(mux_call.kwargs["orig_lang"], "fra")
        self.assertEqual(mux_call.kwargs["ru_lang"], "deu")
        self.assertEqual(mux_call.kwargs["ru_title"], "German (DubPipeline)")

        replace_with_temp.assert_called_once_with(cfg.paths.input_video, mux_temp_out, keep_backup=False)
        cleanup_temp.assert_not_called()


if __name__ == "__main__":
    unittest.main()
