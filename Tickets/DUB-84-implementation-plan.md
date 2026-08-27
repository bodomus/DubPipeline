# DUB-84 Implementation Plan

1. Add config dataclasses for source separation and derived separation paths.
2. Add a provider/cache module with:
   - provider abstraction;
   - original/background fallback provider;
   - command-template BS Roformer provider;
   - metadata-based cache validation.
3. Add a pipeline step after `extract_audio` that runs only for `source_separation.mode: separated_background`.
4. Teach HQ mix rendering to use separated `background.wav` as the original/background input only in separated-background mode.
5. Add focused tests for config parsing, cache hits/misses, fallback behavior, command provider behavior, path derivation, and merge source selection.
6. Run narrow tests first, update CRG, then run CLI plan/help validation.
7. Save completion review as `reviews/review-DUB-84.md`.
