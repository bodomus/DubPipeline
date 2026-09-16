# DUB-86 Implementation Plan

1. Extend configuration:
   - add residual-suppression typed config and conservative defaults;
   - add cleaned-background and residual metadata paths;
   - preserve YAML/ENV/CLI precedence and batch path rebuilding;
   - expose an independently controllable pipeline step.
2. Add `dubpipeline/residual_suppression.py`:
   - typed request/result and explicit error;
   - validation and effective-enable helper;
   - named FFmpeg filter graph/command builders;
   - atomic temp-output promotion;
   - separate cache metadata with both stem identities and all tuning values;
   - stale-output invalidation and explicit fallback behavior;
   - merge resolver selecting cleaned, raw separated, or legacy input.
3. Add `dubpipeline/steps/step_residual_suppression.py` and insert it as `01c` after source separation.
4. Update merge routing to use the residual-suppression resolver while leaving legacy HQ ducking logic unchanged.
5. Add focused tests for:
   - config, paths, defaults, and batch derivation;
   - effective mode/disabled behavior;
   - filter graph/command composition and max-reduction floor;
   - enabled processing using generated WAVs;
   - cache hit and config-driven miss;
   - stale artifact invalidation;
   - fallback and strict failures;
   - merge routing.
6. Update README and sample config documentation without adding GUI controls.
7. Run narrow tests, full tests, CLI help, plan mode, and FFmpeg/FFprobe validation.
8. Run DUB-85 separation and real DUB-86 comparison on `en_710_760.wav` if the native dependency/model is available; record objective measurements and manual-listening status.
9. Update CRG, inspect post-change impact, refresh Graphify because a new module/step changes pipeline structure, and produce `reviews/review-DUB-86.md`.
