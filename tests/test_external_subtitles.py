from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dubpipeline.external_subtitles import (
    find_external_subtitle_for_video,
    prepare_external_subtitles,
    segments_from_subtitle_file,
)


class ExternalSubtitlesTests(unittest.TestCase):
    def test_find_external_subtitle_prefers_priority_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "movie.mp4"
            video.write_text("x", encoding="utf-8")
            (root / "movie.vtt").write_text("WEBVTT\n\n00:00.000 --> 00:01.000\nA", encoding="utf-8")
            srt = root / "movie.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nA\n", encoding="utf-8")

            found = find_external_subtitle_for_video(video)
            self.assertEqual(found, srt)

    def test_segments_from_srt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            srt = root / "movie.srt"
            srt.write_text(
                "1\n00:00:01,000 --> 00:00:02,500\nHello world\n\n"
                "2\n00:00:03,000 --> 00:00:04,000\nSecond line\n",
                encoding="utf-8",
            )

            segments = segments_from_subtitle_file(srt)
            self.assertEqual(len(segments), 2)
            self.assertEqual(segments[0]["id"], 0)
            self.assertAlmostEqual(segments[0]["start"], 1.0)
            self.assertAlmostEqual(segments[0]["end"], 2.5)
            self.assertEqual(segments[0]["text"], "Hello world")

    def test_prepare_external_subtitles_writes_segments_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "movie.mp4"
            video.write_text("x", encoding="utf-8")
            (root / "movie.srt").write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nLine one\n",
                encoding="utf-8",
            )
            segments_path = root / "out" / "movie.segments.json"
            cfg = SimpleNamespace(
                paths=SimpleNamespace(
                    input_video=video,
                    segments_file=segments_path,
                )
            )

            selected = prepare_external_subtitles(cfg)
            self.assertEqual(selected, root / "movie.srt")
            self.assertTrue(segments_path.exists())

            with segments_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["text"], "Line one")


if __name__ == "__main__":
    unittest.main()
