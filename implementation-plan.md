# Implementation Plan

## DUB-85 corrective pass

1. Replace floating `audio-separator[gpu]` with pinned CPU-extra audio-separator and a
   pinned CPU-only ONNX Runtime, preserving all Torch/CUDA pins.
2. Replace ambiguous stem substrings with explicit boundary-aware vocals/background
   markers and add `InstVoc` regressions for both output orders.
3. Update retained `audio-separator` model instances with the per-input output directory
   and test two-request directory isolation with one model load.
4. Remove `tests/TXT/1.txt` from the branch diff.
5. Run focused source separation, merge, and CLI tests, then compile/help/plan checks.
6. Normalize the local validation environment without changing Torch, record exact
   Torch/audio-separator/ORT versions, providers, and `pip check` output.
7. Run real single-file separation and DUB-84 cache validation.
8. Create a second valid short WAV and run both different inputs through one provider in
   one Python process; record one model load, two calls, clean close, and exit.
9. Inspect raw/canonical stem hashes plus FFprobe/RMS/peak metadata and document why the
   speech sample has a nearly silent background stem.
10. Update CRG, inspect the post-change blast radius, refresh Graphify only if structural
    relationships changed, and finalize `reviews/review-DUB-85.md`.
