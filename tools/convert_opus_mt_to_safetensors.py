from pathlib import Path
import shutil

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


src_dir = Path(
    r"C:\Users\bodom\AppData\Local\DubPipeline\models\hf\Helsinki-NLP\opus-mt-en-de"
)

out_dir = Path(
    r"C:\Users\bodom\AppData\Local\DubPipeline\models\hf\Helsinki-NLP\opus-mt-en-de-safetensors"
)

if not src_dir.exists():
    raise SystemExit(f"Source model folder not found: {src_dir}")

out_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading model from: {src_dir}")
model = AutoModelForSeq2SeqLM.from_pretrained(
    src_dir,
    local_files_only=True,
    use_safetensors=False,
)

tokenizer = AutoTokenizer.from_pretrained(
    src_dir,
    local_files_only=True,
)

print(f"Saving safetensors model to: {out_dir}")
model.save_pretrained(
    out_dir,
    safe_serialization=True,
)

tokenizer.save_pretrained(out_dir)

# На всякий случай копируем дополнительные файлы, если tokenizer.save_pretrained
# не перенёс специфичные файлы Marian/OPUS-MT.
for name in [
    "source.spm",
    "target.spm",
    "vocab.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "generation_config.json",
]:
    src_file = src_dir / name
    dst_file = out_dir / name
    if src_file.exists() and not dst_file.exists():
        shutil.copy2(src_file, dst_file)

print("Done.")
print("Check file:", out_dir / "model.safetensors")