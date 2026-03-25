from __future__ import annotations


def test_progress_basic(capsys) -> None:
    from dubpipeline.utils.progress import SegmentProgress

    progress = SegmentProgress(total=10)
    for i in range(1, 11):
        progress.update(i)
    progress.finish()

    out = capsys.readouterr().out
    assert "\r" in out
    assert out.endswith("\n")
    assert "СДЕЛАНО: 10/10 (100%)" in out.split("\r")[-1]
