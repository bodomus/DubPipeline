# tools/segmenter.py
import json

def merge_words_to_segments(words, max_gap=0.8):
    segments = []
    if not words:
        return segments

    current = {
        "speaker": words[0]["speaker"],
        "start": words[0]["start"],
        "end": words[0]["end"],
        "text": words[0]["text"]
    }

    for w in words[1:]:
        same_speaker = (w["speaker"] == current["speaker"])
        small_gap = (w["start"] - current["end"] <= max_gap)

        if same_speaker and small_gap:
            current["text"] += " " + w["text"]
            current["end"] = w["end"]
        else:
            segments.append(current)
            current = {
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "text": w["text"]
            }

    segments.append(current)
    return segments

if __name__ == "__main__":
    with open("whisperx_words.json", "r", encoding="utf-8") as f:
        words = json.load(f)

    segments = merge_words_to_segments(words)
    with open("segments.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print("Segments saved to segments.json")
