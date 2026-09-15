# Investigation

## Ticket

DUB-85 corrective pass before PR.

## Workflow

- Level: 2 (dependency/CUDA, model lifecycle, batch reuse, and media validation).
- Branch: `codex/DUB-85-native-headless-source-separation-provider`.
- Baseline HEAD: `b7a37eb9bab4cbefc4fb63fb288effbcd94ac576`.
- Working tree before changes: clean, apart from permission warnings for generated
  temporary directories.
- Graphify graph was last refreshed on 2026-09-07. Focused queries covered provider
  lifecycle, CLI batching, cache, stem mapping, merge consumers, and tests.
- CRG was updated with `code-review-graph update --brief`; expected impact is confined
  to source separation, its tests, CLI batch reachability, requirements, and docs.

## Current behavior and root causes

- `requirements.txt` uses floating `audio-separator[gpu]`, which installed
  `audio-separator 0.47.0` and `onnxruntime-gpu 1.29.0` in the validation environment.
- ONNX Runtime 1.27+ PyPI GPU wheels target CUDA 13, incompatible with the project's
  pinned PyTorch CUDA 12.4 stack. The managed RoFormer `.ckpt` runs through PyTorch,
  not ONNX Runtime, so the GPU ORT extra is unnecessary for DUB-85.
- `map_audio_separator_outputs()` accepts the substring `inst` as background. A vocals
  filename containing the model label `InstVoc` can therefore be selected as both roles.
- The run-scoped provider updates the outer separator's output directory between files,
  but the already-loaded `model_instance` retains the directory captured at model load.
  A real second separation can therefore write into the first file's directory.
- Existing real stems are not reversed: normalized outputs are byte-identical to their
  explicit raw `(Vocals)` and `(Instrumental)` counterparts. The vocals stem is about
  -20.9 dB RMS while the background is about -52.6 dB RMS. The nearly silent stem is
  the instrumental/background stem, expected for this speech-dominant sample.

## Smallest correct change

- Pin `audio-separator[cpu]==0.47.0` and `onnxruntime==1.23.2`; do not install
  `onnxruntime-gpu`. CPU ORT satisfies audio-separator's unconditional import while the
  managed RoFormer remains CUDA-accelerated by the pinned PyTorch build.
- Replace substring matching with boundary-aware explicit stem semantics, prioritizing
  `No Vocals`/instrumental semantics over the generic vocals marker.
- Propagate each request's output directory to the retained native model instance.
- Add regressions for `InstVoc`, reversed output ordering, and reused-model output dirs.
- Remove unrelated `tests/TXT/1.txt`.

## Risk and compatibility assessment

- CLI: existing run-scoped provider lifecycle remains; only the retained model output
  path is corrected.
- GUI/config: no behavior or precedence changes.
- Models/runtime: lazy load, one model load per key, CUDA via PyTorch, and close behavior
  remain intact.
- Media/artifacts: canonical `vocals.wav`, `background.wav`, metadata, cache identity,
  durations, sample rate, and channels remain unchanged.
- Merge: continues consuming only canonical `background.wav`.
- Validation requires focused tests, compile/help/plan checks, dependency/import checks,
  one real file, a cache hit, and two distinct inputs in one process.
