# точка входа: python scripts/run_subtitles.py input.mp4 output.vtt

import sys
from pathlib import Path
from dubpipeline.pipeline import SubtitlesPipeline

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/run_subtitles.py <input_video> <output_srt>")
        raise SystemExit(1)

    video = Path(sys.argv[1])
    out_srt = Path(sys.argv[2])

    pipeline = SubtitlesPipeline()
    result = pipeline.run(video, out_srt)
    print(f"Subtitles written to: {result}")

if __name__ == "__main__":
    main()
