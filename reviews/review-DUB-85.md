# DUB-85 Review

Issue: https://bodomus.youtrack.cloud/issue/DUB-85/Native-Headless-Source-Separation-Provider

Branch: `codex/DUB-85-native-headless-source-separation-provider`

Corrective-pass baseline: `b7a37eb9bab4cbefc4fb63fb288effbcd94ac576`

## Final verdict

**READY FOR REVIEW**

The dependency/CUDA mismatch, real batch reuse blocker, ambiguous `InstVoc` stem
classification, unrelated text fixture, and stem-quality investigation are resolved.

## Final dependency and CUDA state

`requirements.txt` now pins:

```text
torch==2.6.0+cu124
torchvision==0.21.0+cu124
torchaudio==2.6.0+cu124
audio-separator[cpu]==0.47.0
onnxruntime==1.23.2
numpy==2.0.2
```

Validated in the repository `.venv`:

```text
torch=2.6.0+cu124
torch_cuda=12.4
cuda_available=True
audio_separator=0.47.0
onnxruntime=1.23.2
onnxruntime providers=AzureExecutionProvider,CPUExecutionProvider
onnxruntime-gpu installed=False
pip check: No broken requirements found.
```

The managed BS-RoFormer checkpoint is an MDXC/RoFormer `.ckpt` model and performs
inference through PyTorch. It does not require the ONNX Runtime CUDA execution provider.
The CPU extra is intentional: it satisfies audio-separator's ONNX Runtime import without
installing a CUDA-13 `onnxruntime-gpu` wheel into the CUDA-12.4 PyTorch environment.

`audio-separator` logs that its ONNX execution provider is CPU-only. During the same run
it reports CUDA available in Torch, and both `Separator.torch_device` and the loaded
MDXC model's `torch_device` were `cuda`. There was no CUDA-13 DLL load failure.

## Code corrections

- Removed broad background substring `inst` from stem classification.
- Added boundary-aware explicit roles for `Vocals`, `Instrumental`, `No Vocals`,
  `Background`, `Karaoke`, and `Accompaniment`.
- Background semantics are evaluated before the generic vocals marker, so `No Vocals`
  cannot be classified as vocals.
- Added regression coverage for exact examples:
  - `en_710_760_(Vocals)_BS_Roformer_InstVoc_2.wav`;
  - `en_710_760_(Instrumental)_BS_Roformer_InstVoc_2.wav`;
  - both normal and reversed returned output ordering.
- Propagated each request's output directory into the retained
  `Separator.model_instance`. Previously the outer separator changed directories while
  the loaded model kept the first file's directory.
- Preserved `OriginalAudioBackgroundProvider`, `BsRoformerProvider`,
  `AudioSeparatorProvider`, lazy loading, run-scoped reuse, DUB-84 cache/fallback
  behavior, and canonical artifacts.
- Removed unrelated `tests/TXT/1.txt` from the final working-tree diff against `master`.

## Exact model relationship

Successful validation used:

```text
model_bs_roformer_ep_368_sdr_12.9628.ckpt
audio-separator friendly name: BS-Roformer-Viperx-1296
size: 639,317,465 bytes
MD5: 40780DD7FB1EF17D368A2499CE07A55E
```

The managed checkpoint is byte-identical to the locally installed UVR file
`BS Roformer - InstVoc 1.ckpt` (same size and MD5).

It is **not** identical to the previously tested `BS Roformer - InstVoc 2.ckpt`:

```text
size: 639,331,213 bytes
MD5: 261E8988646EC49498E46FDAF97F0703
```

The two are related BS-RoFormer/InstVoc-style models, but they are distinct checkpoints.

## Real single-file and stem inspection

Input: `tests/Test_for_SPLIT/en_710_760.wav`.

The real CUDA run generated explicit raw and canonical stems. Hash comparison proves
the mapping rather than relying on output order:

```text
canonical vocals.wav SHA256:
1B5392726787BE331750DFE1266F84843799079CBC8EE12B7B79E2864C6E859D
raw (Vocals) SHA256:
1B5392726787BE331750DFE1266F84843799079CBC8EE12B7B79E2864C6E859D

canonical background.wav SHA256:
4DB6146D984FC53F82E40A752B17CD7747DD5421C5C3B8EF9831FFA32F6B4815
raw (Instrumental) SHA256:
4DB6146D984FC53F82E40A752B17CD7747DD5421C5C3B8EF9831FFA32F6B4815
```

FFprobe/FFmpeg inspection:

| File | Duration | Rate/channels | Mean | Peak |
| --- | ---: | --- | ---: | ---: |
| source | 50.0 s | 44.1 kHz stereo | -20.0 dB | 0.0 dB |
| vocals | 50.0 s | 44.1 kHz stereo | -20.9 dB | -0.9 dB |
| background | 50.0 s | 44.1 kHz stereo | -52.6 dB | -26.3 dB |

The review's “nearly silent vocals” observation was a role interpretation problem, not
the current output. The nearly silent file is the explicit `(Instrumental)`/background
stem. This source is speech-dominant, so the model places almost all energy into the
explicit `(Vocals)` stem; vocals are only 0.9 dB below source mean level, while the
background is 31.7 dB below vocals. The stems are not reversed.

## Real same-process batch validation

Inputs:

1. original 50-second `en_710_760.wav`;
2. a different valid 10-second excerpt beginning at 30 seconds.

Observed lifecycle:

```text
successful Separator constructions: 1
load_model calls: 1
separate calls: 2
separator device: cuda
model device: cuda
provider_closed: True
process exit: clean
```

The first separation completed in about 9 seconds after a 2-second model load; the
second completed in under one second. Both files wrote raw and canonical stems into
their own directories. No hang occurred.

A new provider was then passed the first configuration with cache enabled. It returned
`cache_hit=True`; construction/load/separation counts remained `1/1/2`, proving the
DUB-84 cache path does not initialize the model.

## Validation

- `python -m unittest discover -s tests -p "test_source_separation.py"`: 23 passed.
- `python -m unittest discover -s tests -p "test_step_merge_hq.py"`: 6 passed.
- `python -m unittest discover -s tests -p "test_cli.py"`: 35 passed.
- Production Python files compiled with `python -m py_compile`: passed.
- `python -m dubpipeline.cli --help`: passed.
- Native-provider `run ... --plan`: passed; no audio-separator import/model load occurred.
- Exact dependency/import/CUDA checks: passed.
- `pip check`: passed with no broken requirements.
- Real single-file CUDA separation: passed.
- Real two-file, one-process CUDA batch: passed with one model load.
- Cache-hit rerun: passed without model construction/load.
- FFprobe, volume, and hash-based stem inspection: passed.
- Post-change `code-review-graph update --brief --base master`: passed; 15 branch
  files analyzed, no affected flows reported. Focused reverse traversal confirmed the
  expected provider -> source-separation step -> CLI path and merge/test consumers.
- Graphify was queried before and after implementation. A full refresh was not needed
  because this corrective pass changes no module, provider, step, or orchestration
  relationship.

`pytest` is not installed in this repository environment, so the repository's existing
`unittest` suites were used directly.

## Remaining limitations

- CPU-only ONNX Runtime intentionally cannot accelerate ONNX-format separation models.
  DUB-85's managed RoFormer `.ckpt` remains CUDA-accelerated through PyTorch.
- `audio-separator` emits a generic warning that ONNX Runtime lacks CUDA support even
  for the PyTorch RoFormer path; device inspection confirms this run used CUDA.
- Objective metadata, hashes, and signal levels were inspected. Interactive human
  listening is still a useful optional subjective check, but no qualitative listening
  claim is needed to establish role correctness.
