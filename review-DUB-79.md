# Implementation Report

## Ticket

DUB-79 - Local Qwen Translation Provider (final quality pass).

Roadmap document number: DUB-76.

## Scope Completed

- Kept the existing local-only `Qwen/Qwen3-8B-FP8` provider, RTX 4080-safe quantization, cache identity, model lifecycle, output validation, and offline installer behavior.
- Added a fixed 25-line technical EN-to-RU quality dataset and a reproducible local GPU benchmark runner.
- Reworked the default system prompt around general technical interpretation, data/control direction, Russian dubbing style, protected tokens, and translation-only output. No phrase-specific replacement table or output post-processing was added.
- Compared Qwen-recommended non-thinking sampling, conservative sampling, and deterministic decoding on the same dataset.
- Selected and versioned `qwen3-nothink-conservative-translation-v3`.
- Preserved the prompt hash and generation profile in Qwen `cache_scope`, so both changes invalidate stale Qwen cache entries.

## Translation Quality Pass

### Dataset

The source dataset is [qwen_translation_quality_dataset.json](benchmarks/qwen_translation_quality_dataset.json). It contains 25 isolated segments covering Unreal Engine/material nodes, programming, callbacks/threads, numbers, units, versions, timestamps, paths, API/GPU/VRAM/TTS abbreviations, and more difficult shader/widget instructions.

### Compared Profiles

Research basis: the official [Qwen Transformers guide](https://qwen.readthedocs.io/en/stable/inference/transformers.html) and [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) document `enable_thinking=False` and recommend `temperature=0.7`, `top_p=0.8`, `top_k=20` for non-thinking inference. Conservative and deterministic candidates were evaluated empirically for this translation workload rather than treated as undocumented defaults.

| Candidate | Settings | 25-line time | Repeatability result |
| --- | --- | ---: | --- |
| A: Qwen recommended | `do_sample=true`, `temperature=0.7`, `top_p=0.8`, `top_k=20` | 35.79 s | Meaning improved, but wording varied on `Feed` across 5 runs. |
| B: conservative | `do_sample=true`, `temperature=0.3`, `top_p=0.8`, `top_k=20` | 34.68 s | All three difficult phrases were identical in 5/5 runs. |
| C: deterministic | `do_sample=false`, greedy | 35.77 s | Stable, but no quality gain over B and not the documented non-thinking default. |

The full A/B/C source and output matrix is saved in [qwen_translation_quality_results.json](benchmarks/qwen_translation_quality_results.json).

### Selected Profile

`qwen3-nothink-conservative-translation-v3` uses conservative sampling:

```text
do_sample=true
temperature=0.3
top_p=0.8
top_k=20
num_beams=1
enable_thinking=false
```

It won because it corrected the technical meaning, matched or slightly beat the other profiles on dataset time, and was stable in the required uncached repeatability test. Deterministic decoding did not improve the wider translations enough to replace Qwen's documented non-thinking sampling approach.

### Difficult Examples

| Source | Final Russian |
| --- | --- |
| `Feed it into the node.` | `Подайте его в узел.` |
| `Drive this value with the parameter.` | `Управляйте этим значением параметром.` |
| `Pass the result into the Lerp node.` | `Передай результат в узел Lerp.` |
| `Connect this output to the material input.` | `Подключите этот выход к входу материала.` |

The food-related mistranslation is gone. Data/control direction is preserved. A few isolated commands still vary between polite plural and natural singular imperative, but they remain concise, understandable, and semantically correct; no context-aware rewriting from DUB-80 was introduced.

### Final Production Outputs

| EN source | RU result |
| --- | --- |
| Feed it into the node. | Подайте его в узел. |
| Connect this output to the material input. | Подключите этот выход к входу материала. |
| Plug the texture into the normal input. | Подключите текстуру к входу нормали. |
| Drive this value with the parameter. | Управляйте этим значением параметром. |
| Pass the result into the Lerp node. | Передай результат в узел Lerp. |
| Route the output through this function. | Направьте выход через эту функцию. |
| Sample the texture using these UV coordinates. | Сэмплируйте текстуру с использованием этих координат UV. |
| Multiply the normal by this value. | Умножьте нормаль на это значение. |
| Clamp the result between zero and one. | Ограничь результат между нулём и единицей. |
| Blend the two materials together. | Смешайте два материала вместе. |
| Pass the value to the function. | Передайте значение функции. |
| The function returns a boolean value. | Функция возвращает значение типа булево. |
| Store the result in the variable. | Сохраните результат в переменную. |
| Call the method from the main thread. | Вызовите метод из основного потока. |
| The API returns a JSON object. | API возвращает объект в формате JSON. |
| The callback is invoked when the request completes. | Вы получаете вызов обратного вызова при завершении запроса. |
| Set the gain to -12 dB. | Установите коэффициент усиления на -12 дБ. |
| Wait 00:01:23 before starting TTS. | Подождите 00:01:23 перед началом TTS. |
| Use version 1.2.3 of the API. | Используйте версию 1.2.3 API. |
| The file is saved as C:/Temp/output.wav. | Файл сохранён как C:/Temp/output.wav. |
| The GPU has 16 GB of VRAM. | Графический процессор имеет 16 ГБ видеопамяти. |
| Expose the scalar parameter so the material instance can override it. | Доступ к скалярному параметру должен быть предоставлен, чтобы материал мог его переопределить. |
| Normalize the vector before passing it to the shader. | Нормализуйте вектор перед передачей его в шейдер. |
| Bind the event before starting the asynchronous task. | Привяжите событие до запуска асинхронной задачи. |
| Read the return value on the game thread and update the widget. | Прочтите возвращаемое значение на игровом потоке и обновите виджет. |

The final selected-profile artifact, including all outputs and repeatability runs, is [qwen_translation_quality_final_results.json](benchmarks/qwen_translation_quality_final_results.json).

### Repeatability

Without SQLite cache, 5/5 runs were identical for each required phrase:

- `Feed it into the node.` -> `Подайте его в узел.`
- `Drive this value with the parameter.` -> `Управляйте этим значением параметром.`
- `Pass the result into the Lerp node.` -> `Передай результат в узел Lerp.`

### Performance

Final production-profile run:

- GPU: NVIDIA GeForce RTX 4080, 16375.5 MiB total.
- Model: `Qwen/Qwen3-8B-FP8`, local-only, FP8, dtype auto.
- Python 3.12.10, torch 2.6.0+cu124, CUDA 12.4, Transformers 4.57.3.
- Model load: 6.57 s.
- 25-line dataset: 35.34 s.
- Peak allocated VRAM: 9223.7 MiB.
- Peak reserved VRAM: 9260.0 MiB.
- CUDA OOM: none.
- Model release: completed.

On the same final prompt and 25-line dataset, conservative sampling took 34.68 s versus 35.79 s for the previous DUB-79 sampling settings. The difference is small enough to treat as neutral; profile selection was based on meaning and repeatability. The final selected-only run used 9223.7 MiB peak allocated VRAM and stayed comfortably inside 16 GB.

## Validation

- Real Qwen3-8B-FP8 GPU A/B/C comparison: passed.
- Final production-profile GPU run: passed, no OOM.
- Focused translation/token/model suite: `61 passed`, 3 dependency warnings, 7 subtests passed.
- Main `tests/` suite: `146 passed`, 2 unrelated failures, 3 dependency warnings, 9 subtests passed.
- Unrelated failures: `test_external_voice_skips_extract_audio_step` (existing extract-audio/external-voice behavior) and `test_coqui_provider_without_package_has_clear_error` (environment has Coqui, so the test reaches missing segments instead of its expected missing-package error).
- Repository-wide `pytest`: collection is blocked by legacy `tools/test_whisperx_basic.py`, which executes WhisperX at import and sees a test stub without `load_model`.
- CLI `python -m dubpipeline.cli --help`: passed.
- Prompt/profile cache invalidation, thinking disablement, generation kwargs, and output validation: covered without loading Qwen in unit tests.
- TechnicalTokenProtector numbers/units/versions/paths/timestamps and Argos/NLLB/OPUS regressions: passed in focused/main suites.

## Risks and Boundaries

- Qwen3-8B remains a probabilistic 8B model. Conservative sampling was stable on the required repeated phrases, but arbitrary future segments can still vary before SQLite cache records the first result.
- The provider translates isolated segments only. Neighboring context, segment merging, and timing changes remain intentionally reserved for DUB-80.
- Transformers still emits the existing `torch_dtype` deprecation warning during FP8 load; it does not affect this ticket's output or measurements.
- Working-tree note: the tracked deletions under `tests/.tmp_runtime` and untracked `tests/TXT/1.txt` were already present before this pass and were not modified. Additional untracked `tests/.tmp_runtime` cases were generated by validation and are not production changes.
- No PR or merge to `master` was performed.

READY FOR REVIEW
