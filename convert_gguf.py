from unsloth import FastLanguageModel
import os

max_seq_length = 16384

# 最新チェックポイントを自動検出
output_dir = "outputs_qwen35"
checkpoints = sorted(
    [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
    key=lambda x: int(x.split("-")[1]),
)
if checkpoints:
    model_path = os.path.join(output_dir, checkpoints[-1])
    print(f"Using checkpoint: {model_path}")
else:
    model_path = output_dir  # チェックポイントがなければ直接使う
    print(f"Using output dir: {model_path}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    load_in_16bit=True,
)

# ─── GGUF 変換 ────────────────────────────────────────────────────────────────
model.save_pretrained_gguf("gguf_q4_k_m", tokenizer, quantization_method="q4_k_m")
model.save_pretrained_gguf("gguf_q8_0",   tokenizer, quantization_method="q8_0")
model.save_pretrained_gguf("gguf_f16",    tokenizer, quantization_method="f16")

# ─── HuggingFace へアップロード ───────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")  # コンテナ起動時に export HF_TOKEN=xxx しておく
REPO_ID = "ChiTako/Qwen3.5-0.8B-frank-ja"

model.push_to_hub_gguf(
    REPO_ID,
    tokenizer,
    quantization_method=["q4_k_m", "q8_0", "f16"],
    token=HF_TOKEN,
)
print(f"Upload complete: https://huggingface.co/{REPO_ID}")
