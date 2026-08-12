from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

from dubpipeline.config import PipelineConfig
from dubpipeline.translation.providers import (
    QWEN_GENERATION_CANDIDATES,
    QWEN_GENERATION_PROFILE,
    QwenTranslationProvider,
)
from dubpipeline.translation.service import TranslatorService


REPEATABILITY_PHRASES = (
    "Feed it into the node.",
    "Drive this value with the parameter.",
    "Pass the result into the Lerp node.",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local Qwen translation generation profiles.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("benchmarks/qwen_translation_quality_dataset.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/qwen_translation_quality_results.json"),
    )
    parser.add_argument("--profiles", nargs="+", choices=tuple(QWEN_GENERATION_CANDIDATES), default=None)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def _runtime_config() -> PipelineConfig:
    cfg = PipelineConfig(project_name="qwen-quality", project_dir=Path.cwd())
    cfg.languages.src = "en"
    cfg.languages.tgt = "ru"
    cfg.usegpu = True
    cfg.translation.provider = "qwen"
    cfg.translation.device = "cuda"
    cfg.translation.dtype = "auto"
    cfg.translation.quantization = "fp8"
    cfg.translation.max_new_tokens = 128
    return cfg


def _translate_with_profile(
    provider: QwenTranslationProvider,
    tokenizer: object,
    model: object,
    device: str,
    texts: list[str],
    profile_name: str,
) -> tuple[list[str], float]:
    settings = QWEN_GENERATION_CANDIDATES[profile_name]
    started = time.perf_counter()
    outputs: list[str] = []
    for source in texts:
        prompt = provider.build_prompt(source)
        raw = provider._generate(prompt, tokenizer, model, device, settings=settings)
        cleaned = provider.clean_translation_output(raw)
        outputs.append(provider.validate_translation_output(source, raw, cleaned))
    return outputs, time.perf_counter() - started


def main() -> int:
    args = _parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or not dataset or not all(isinstance(item, str) for item in dataset):
        raise ValueError("Dataset must be a non-empty JSON array of strings.")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    import torch
    import transformers

    cfg = _runtime_config()
    service = TranslatorService(cfg)
    provider = service._provider
    if not isinstance(provider, QwenTranslationProvider):
        raise RuntimeError("Qwen provider was not selected.")

    profile_names = args.profiles or list(QWEN_GENERATION_CANDIDATES)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_before, total_vram = torch.cuda.mem_get_info()
    try:
        load_started = time.perf_counter()
        tokenizer, model, device = provider._load_qwen()
        load_seconds = time.perf_counter() - load_started

        profiles: dict[str, object] = {}
        for profile_name in profile_names:
            outputs, elapsed = _translate_with_profile(
                provider,
                tokenizer,
                model,
                device,
                dataset,
                profile_name,
            )
            repeated: dict[str, list[str]] = {}
            for phrase in REPEATABILITY_PHRASES:
                repeated[phrase] = [
                    _translate_with_profile(
                        provider,
                        tokenizer,
                        model,
                        device,
                        [phrase],
                        profile_name,
                    )[0][0]
                    for _ in range(args.repeats)
                ]
            profiles[profile_name] = {
                "settings": asdict(QWEN_GENERATION_CANDIDATES[profile_name]),
                "dataset_seconds": elapsed,
                "translations": [
                    {"source": source, "translation": translation}
                    for source, translation in zip(dataset, outputs, strict=True)
                ],
                "repeatability": repeated,
            }

        result = {
            "production_profile": QWEN_GENERATION_PROFILE,
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "model": service._active.model_ref,
                "quantization": cfg.translation.quantization,
                "dtype": cfg.translation.dtype,
                "load_seconds": load_seconds,
                "vram_total_mib": total_vram / 1024**2,
                "vram_free_before_mib": free_before / 1024**2,
                "vram_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "vram_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            },
            "profiles": profiles,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        service.release()
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
