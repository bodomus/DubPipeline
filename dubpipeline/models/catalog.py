from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

Tier = Literal["A", "B", "C"]

SUPPORTED_BACKENDS = {"nllb", "opus_mt", "argos"}
NOT_INSTALLED_REASON = "not installed"
NOT_SUPPORTED_REASON = "not supported yet"

_HF_REQUIRED_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "model.safetensors",
    "pytorch_model.bin",
}


@dataclass(frozen=True)
class ModelStatus:
    available: bool
    enabled: bool
    reason: str = ""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    tier: Tier
    label: str
    backend: str
    model_ref: str
    local_check: Callable[["ModelSpec"], tuple[bool, str]]


@dataclass(frozen=True)
class ModelChoice:
    display: str
    model_id: str | None
    tier_label: str
    enabled: bool
    available: bool
    reason: str
    color: str
    is_group_header: bool = False


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _local_model_dirs() -> list[Path]:
    dirs: list[Path] = []
    env_models = os.getenv("DUBPIPELINE_MODELS_DIR", "").strip()
    if env_models:
        dirs.append(Path(env_models).expanduser())
    dirs.append(_project_root() / "models")
    return dirs


def _llm_model_dirs() -> list[Path]:
    dirs: list[Path] = []
    env_llm = os.getenv("DUBPIPELINE_LLM_MODELS_DIR", "").strip()
    if env_llm:
        dirs.append(Path(env_llm).expanduser())
    for base in _local_model_dirs():
        dirs.append(base / "llm")
    return dirs


def _hf_cache_roots() -> list[Path]:
    roots: list[Path] = []

    hf_home = os.getenv("HF_HOME", "").strip()
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")

    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE", "").strip()
    if hub_cache:
        roots.append(Path(hub_cache).expanduser())

    transformers_cache = os.getenv("TRANSFORMERS_CACHE", "").strip()
    if transformers_cache:
        roots.append(Path(transformers_cache).expanduser())

    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    roots.append(Path.home() / ".huggingface" / "hub")
    return roots


def _dir_has_hf_artifacts(folder: Path) -> bool:
    if not folder.exists() or not folder.is_dir():
        return False
    try:
        names = {p.name for p in folder.iterdir() if p.is_file()}
    except OSError:
        return False
    return bool(_HF_REQUIRED_FILES.intersection(names))


def _local_hf_candidates(spec: ModelSpec, model_ref_safe: str) -> list[Path]:
    model_ref = spec.model_ref
    model_ref_name = Path(model_ref).name
    candidates: list[Path] = []
    for base in _local_model_dirs():
        candidates.extend(
            [
                base / model_ref,
                base / model_ref_safe,
                base / spec.id,
                base / model_ref_name,
            ]
        )
    return candidates


def _has_local_hf_model(spec: ModelSpec, model_ref_safe: str) -> bool:
    for candidate in _local_hf_candidates(spec, model_ref_safe):
        if candidate.is_file() or _dir_has_hf_artifacts(candidate):
            return True
    return False


def _has_hf_cache_model(model_ref_safe: str) -> bool:
    repo_dir_name = f"models--{model_ref_safe}"
    for root in _hf_cache_roots():
        repo_dir = root / repo_dir_name
        if not repo_dir.exists():
            continue

        snapshots_dir = repo_dir / "snapshots"
        if snapshots_dir.exists() and snapshots_dir.is_dir():
            for snapshot in snapshots_dir.iterdir():
                if _dir_has_hf_artifacts(snapshot):
                    return True

        if _dir_has_hf_artifacts(repo_dir):
            return True
    return False


def _check_hf_model_available(spec: ModelSpec) -> tuple[bool, str]:
    model_ref_safe = spec.model_ref.replace("/", "--")
    if _has_local_hf_model(spec, model_ref_safe):
        return True, ""
    if _has_hf_cache_model(model_ref_safe):
        return True, ""
    return False, NOT_INSTALLED_REASON


def _check_argos_available(spec: ModelSpec) -> tuple[bool, str]:
    _ = spec
    try:
        from argostranslate import package
    except Exception:
        return False, NOT_INSTALLED_REASON

    try:
        installed = package.get_installed_packages()
    except Exception:
        return False, NOT_INSTALLED_REASON

    for pkg in installed:
        if getattr(pkg, "from_code", None) == "en" and getattr(pkg, "to_code", None) == "ru":
            return True, ""

    return False, NOT_INSTALLED_REASON


def _check_gguf_available(spec: ModelSpec) -> tuple[bool, str]:
    tokens = {
        spec.id.lower().replace("_", "-"),
        spec.model_ref.lower().replace("/", "-"),
        Path(spec.model_ref).name.lower().replace("_", "-"),
    }

    for model_dir in _llm_model_dirs():
        if not model_dir.exists():
            continue
        try:
            for gguf_file in model_dir.rglob("*.gguf"):
                file_name = gguf_file.name.lower().replace("_", "-")
                if any(token in file_name for token in tokens):
                    return True, ""
        except OSError:
            continue

    return False, NOT_INSTALLED_REASON


_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="nllb_200_1_3b",
        tier="A",
        label="Meta NLLB-200 (1.3B)",
        backend="nllb",
        model_ref="facebook/nllb-200-1.3B",
        local_check=_check_hf_model_available,
    ),
    ModelSpec(
        id="nllb_200_3_3b",
        tier="A",
        label="Meta NLLB-200 (3.3B)",
        backend="nllb",
        model_ref="facebook/nllb-200-3.3B",
        local_check=_check_hf_model_available,
    ),
    ModelSpec(
        id="qwen2_5_7b",
        tier="B",
        label="Qwen2.5 (7B)",
        backend="llm_qwen",
        model_ref="Qwen/Qwen2.5-7B-Instruct",
        local_check=_check_gguf_available,
    ),
    ModelSpec(
        id="qwen2_5_14b",
        tier="B",
        label="Qwen2.5 (14B)",
        backend="llm_qwen",
        model_ref="Qwen/Qwen2.5-14B-Instruct",
        local_check=_check_gguf_available,
    ),
    ModelSpec(
        id="mistral_7b",
        tier="B",
        label="Mistral 7B",
        backend="llm_mistral",
        model_ref="mistralai/Mistral-7B-Instruct-v0.3",
        local_check=_check_gguf_available,
    ),
    ModelSpec(
        id="mixtral_8x7b",
        tier="B",
        label="Mixtral (8x7B)",
        backend="llm_mistral",
        model_ref="mistralai/Mixtral-8x7B-Instruct-v0.1",
        local_check=_check_gguf_available,
    ),
    ModelSpec(
        id="opus_mt",
        tier="C",
        label="OPUS-MT (Helsinki-NLP)",
        backend="opus_mt",
        model_ref="Helsinki-NLP/opus-mt-en-ru",
        local_check=_check_hf_model_available,
    ),
    ModelSpec(
        id="argos",
        tier="C",
        label="Argos Translate",
        backend="argos",
        model_ref="argos-en-ru",
        local_check=_check_argos_available,
    ),
)

_MODELS_BY_ID = {spec.id: spec for spec in _MODEL_SPECS}
if len(_MODELS_BY_ID) != len(_MODEL_SPECS):
    raise ValueError("Duplicate model ids in translation model catalog")


def list_model_specs() -> list[ModelSpec]:
    return list(_MODEL_SPECS)


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return _MODELS_BY_ID[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown translation model_id: '{model_id}'") from exc


def get_model_status(model_id: str) -> ModelStatus:
    spec = get_model_spec(model_id)
    available, reason = spec.local_check(spec)

    if spec.backend not in SUPPORTED_BACKENDS:
        return ModelStatus(available=available, enabled=False, reason=NOT_SUPPORTED_REASON)
    if not available:
        return ModelStatus(available=False, enabled=False, reason=reason or NOT_INSTALLED_REASON)
    return ModelStatus(available=True, enabled=True, reason="")


def is_model_available(model: ModelSpec) -> bool:
    return get_model_status(model.id).available


def resolve_default_model_id() -> str:
    primary = "nllb_200_1_3b"
    if get_model_status(primary).enabled:
        return primary
    for fallback in ("opus_mt", "argos"):
        if get_model_status(fallback).enabled:
            return fallback
    return primary


def infer_model_id_from_legacy_translate(backend: str, hf_model: str) -> str | None:
    backend_norm = (backend or "").strip().lower()
    hf_model_norm = (hf_model or "").strip().lower()

    if backend_norm == "argos":
        return "argos"

    if "nllb-200-3.3b" in hf_model_norm:
        return "nllb_200_3_3b"
    if "nllb-200-1.3b" in hf_model_norm:
        return "nllb_200_1_3b"
    if "opus-mt" in hf_model_norm:
        return "opus_mt"

    if backend_norm in {"nllb", "opus_mt"} and backend_norm in _MODELS_BY_ID:
        return backend_norm
    if backend_norm == "hf" and hf_model_norm:
        return "opus_mt"

    return None


def legacy_translate_backend_for_model(spec: ModelSpec) -> str:
    if spec.backend in {"nllb", "opus_mt"}:
        return "hf"
    return spec.backend


def _tier_label(tier: Tier) -> str:
    if tier == "C":
        return "Lightweight / fallback (Tier C)"
    return f"Tier {tier}"


def build_model_choices() -> list[ModelChoice]:
    choices: list[ModelChoice] = []
    for tier in ("A", "B", "C"):
        tier_label = _tier_label(tier)
        choices.append(
            ModelChoice(
                display=f"{tier_label}",
                model_id=None,
                tier_label=tier_label,
                enabled=False,
                available=False,
                reason="",
                color="gray40",
                is_group_header=True,
            )
        )

        for spec in [m for m in _MODEL_SPECS if m.tier == tier]:
            status = get_model_status(spec.id)
            suffix = ""
            if status.reason == NOT_SUPPORTED_REASON:
                suffix = " (not supported yet)"
            elif not status.available:
                suffix = " (not installed)"

            color = "black"
            if status.reason == NOT_SUPPORTED_REASON:
                color = "gray45"
            elif not status.available:
                color = "firebrick"

            choices.append(
                ModelChoice(
                    display=f"  {spec.label}{suffix}",
                    model_id=spec.id,
                    tier_label=tier_label,
                    enabled=status.enabled,
                    available=status.available,
                    reason=status.reason,
                    color=color,
                )
            )
    return choices
