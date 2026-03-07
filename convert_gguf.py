from unsloth import FastLanguageModel

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs_qwen35/checkpoint-100",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    load_in_16bit = True,
)

model.save_pretrained_gguf("gguf_q4_k_m", tokenizer, quantization_method = "q4_k_m")
model.save_pretrained_gguf("gguf_q8_0",   tokenizer, quantization_method = "q8_0")
model.save_pretrained_gguf("gguf_f16",    tokenizer, quantization_method = "f16")
