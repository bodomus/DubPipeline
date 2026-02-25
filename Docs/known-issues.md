WhisperX Alignment Fails on Russian Audio (torch < 2.6)
Symptoms

Ошибка при обработке русскоязычного файла.

Падение на этапе whisperx.load_align_model.

В логе встречается:

torch.load(..., weights_only=True)

упоминание CVE-2025-32434

ошибка загрузки модели jonatasgrosman/wav2vec2-large-xlsr-53-russian

Trigger

Alignment для языка ru требует загрузки модели
jonatasgrosman/wav2vec2-large-xlsr-53-russian.

При использовании PyTorch < 2.6 загрузка через torch.load блокируется (ограничение безопасности после CVE-2025-32434).

Root Cause

transformers вызывает torch.load(..., weights_only=True).
В версиях torch ниже 2.6 такая загрузка запрещена.

Проблема проявляется только когда:

включён alignment

язык распознан как ru

Resolution

Upgrade PyTorch and torchaudio:

CUDA:
pip install -U "torch>=2.6.0" "torchaudio>=2.6.0" --index-url https://download.pytorch.org/whl/cu121
CPU:
pip install -U "torch>=2.6.0" "torchaudio>=2.6.0" --index-url https://download.pytorch.org/whl/cpu

Verify:

python -c "import torch; print(torch.__version__)"
Temporary Workaround

If upgrade is not possible:

Disable alignment step

Or disable word-level timestamps

Or skip align in CLI --steps

This reduces timestamp precision but prevents pipeline crash.

Recommendation

Pin minimal versions in project dependencies:

torch>=2.6.0
torchaudio>=2.6.0

Add preflight version check in pipeline startup to fail fast with clear message.