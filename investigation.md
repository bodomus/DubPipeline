# Investigation

## Ticket

DUB-79 - Local Qwen Translation Provider.

Roadmap document number: DUB-76.

## Workflow

- Level: 2, because this adds a translation runtime/provider and model lifecycle behavior.
- Branch: `codex/dub-79-local-qwen-translation-provider`.
- Baseline HEAD: `bd42a917dfc4b03892ddb7627243c782a5b6d5c8`.
- Working tree before changes: only pre-existing untracked `tests/TXT/1.txt`.

## Graphify / CRG

- `graphify update .` succeeded: 1025 nodes, 2445 edges, 56 communities.
- `code-review-graph build --repo .` succeeded: 75 files, 797 nodes, 7851 edges.
- Graphify query identified `TranslatorService`, `dubpipeline.translation.providers`, `load_pipeline_config_ex`, `build_parser`, `step_translate.run`, model catalog/installer, and TTS/WhisperX cache/device patterns as the relevant working set.
- CRG showed expected impact around CLI/config/translation/tests with no existing flow changes before implementation.

## Current Behavior

- `TranslatorService` is the public facade used by `step_translate`.
- Provider abstraction lives in `dubpipeline.translation.providers`.
- Public provider values are currently `auto` and `argos`.
- HF seq2seq provider handles NLLB/OPUS-MT through `AutoModelForSeq2SeqLM`, local-only loading, caching, and release.
- Argos provider remains a compatibility/fallback provider.
- Model catalog already contains planned Qwen2.5 entries using backend `llm_qwen`, but they are unsupported and not installable.
- Model installer already supports Hugging Face snapshot downloads into the local DubPipeline models root.
- TTS and HF translation cache loaded models by `(model_name, device)`/`(device, model_ref)` and avoid per-segment reloads.

## Expected Behavior

- `translation.provider: qwen` and `--translation-provider qwen` select a local Qwen provider.
- Default behavior remains unchanged when provider is omitted or `auto`.
- Initial supported Qwen model is `Qwen/Qwen3-8B-FP8` for CUDA/`quantization=auto`, with configurable model reference and `quantization=none` escape hatch for explicit full-precision models.
- Qwen load is local-only during translation.
- Model/tokenizer are loaded once per process/cache key and reused across translation calls.
- Missing local model produces a clear actionable error.
- CUDA requested but unavailable fails clearly.

## Decisions

- Use Hugging Face Transformers for Qwen because `torch`, `transformers`, and `huggingface_hub` are already available in the environment and the repository already has HF snapshot storage/install patterns.
- Do not use llama.cpp/GGUF in this ticket because no such runtime lifecycle exists in the repo today.
- Add `qwen3_8b` as a supported HF snapshot model in the catalog and route backend `llm_qwen` to provider `qwen`.
- Add provider-specific config under `translation`: `model`, `device`, `dtype`, `quantization`, `max_new_tokens`, optional `prompt`, and folder-run lifecycle flags.
- Use official Qwen FP8 on Windows with `accelerate` plus `triton-windows`; regular `triton` has no Windows wheel in this environment.
- Keep shared legacy `translate.max_new_tokens` as fallback when provider-specific max tokens are not set.

## Affected Surfaces

- CLI: provider choices and help.
- Config: `TranslationConfig`, YAML/env/CLI precedence, model-ref override.
- Translation runtime: provider registry, Qwen provider, runtime resolution.
- Model catalog/installer: supported Qwen3 HF snapshot.
- Tests/docs: provider registration, config parsing, prompt/output cleanup, lifecycle with fakes, CLI parsing.

## Final Translation Quality Pass

- Workflow level remains Level 2 because prompt and generation behavior affect cached production output and require a real GPU comparison.
- The current production profile is Qwen3 non-thinking sampling with `temperature=0.7`, `top_p=0.8`, and `top_k=20`.
- Official Qwen3 guidance recommends those sampling settings for non-thinking mode. Its explicit greedy-decoding warning is attached to thinking mode, so the ticket's deterministic non-thinking candidate still had to be measured rather than rejected from documentation alone.
- The observed failure (`Feed it into the node.` translated with a food-related meaning) is a semantic prompt/inference-quality issue, not a token-protection or output-validation failure.
- The smallest coherent change is to add general technical-context guidance, compare three explicit profiles on one fixed isolated-segment dataset, and version the selected prompt/profile through the existing Qwen cache scope.
- DUB-80 context work is explicitly out of scope: no neighboring segments, timing changes, or segment merging will be introduced.
- A real local `Qwen/Qwen3-8B-FP8` run on RTX 4080 is required for selection, repeatability, VRAM, and performance evidence.
- The fixed 25-line GPU comparison showed that English-only prompt guidance did not reliably preserve data/control direction. Russian target-language instructions corrected the semantic failures without phrase-specific replacements.
- Standard sampling (`0.7/0.8/20`) varied on the required difficult phrases. Conservative sampling (`0.3/0.8/20`) produced identical outputs in all 5 repeats for each required phrase and retained better natural phrasing than greedy decoding on the wider dataset.
- Selected profile: `qwen3-nothink-conservative-translation-v3`.
