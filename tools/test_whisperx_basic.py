import torch
import whisperx

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

print("Device:", DEVICE, "compute_type:", COMPUTE_TYPE)

# Модель с невысоким потреблением VRAM:
model_name = "small.en"  # можно будет поднять до "medium.en"

model = whisperx.load_model(
    model_name,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
)
print("Model loaded:", model_name)
