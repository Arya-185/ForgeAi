import os
import numpy as np
import onnxruntime as ort
from transformers import T5Tokenizer

# === Model and Path Setup ===
MODEL_NAME = "t5-small"
MODEL_DIR = "C:/Users/Arya/PycharmProjects/ForgeAi/t5-small/onnx/"
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_model.onnx")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder_model.onnx")

# === Load Tokenizer ===
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
decoder_start_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id

# === Load ONNX Runtime Sessions with Hybrid Execution (CPU + NPU) ===
session_options = ort.SessionOptions()
session_options.log_severity_level = 0  # Enable debug logging

encoder_session = ort.InferenceSession(ENCODER_PATH, session_options, providers=["CPUExecutionProvider", "DmlExecutionProvider"])
decoder_session = ort.InferenceSession(DECODER_PATH, session_options, providers=["CPUExecutionProvider", "DmlExecutionProvider"])

def log_softmax(x, axis=-1):
    """Compute log softmax for numerical stability."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return x - x_max - np.log(sum_exp_x)

def generate_text(prompt, num_beams=4, max_length=200):
    """Generate text using ONNX T5 model and beam search."""
    input_text = f"write a detailed explanation: {prompt}"

    # Tokenize input (Ensuring proper truncation to avoid errors)
    tokens = tokenizer(
        input_text,
        return_tensors="np",
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    # Encode input using ONNX Runtime
    encoder_outputs = encoder_session.run(
        None,
        {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }
    )
    encoder_hidden_states = encoder_outputs[0]

    # Initialize beams with start token
    beams = [(np.array([[decoder_start_token_id]], dtype=np.int64), 0.0)]
    completed_beams = []

    for _ in range(max_length):
        new_beams = []

        for seq, score in beams:
            if seq[0, -1] == eos_token_id:
                completed_beams.append((seq, score))
                continue

            decoder_outputs = decoder_session.run(
                None,
                {
                    "input_ids": seq,
                    "encoder_attention_mask": tokens["attention_mask"],
                    "encoder_hidden_states": encoder_hidden_states
                }
            )

            logits = decoder_outputs[0][:, -1, :]
            log_probs = log_softmax(logits, axis=-1)[0]
            top_k_tokens = np.argsort(log_probs)[-num_beams:]

            for token_id in top_k_tokens:
                new_seq = np.concatenate([seq, np.array([[token_id]], dtype=np.int64)], axis=1)
                new_score = score + log_probs[token_id]
                new_beams.append((new_seq, new_score))

        if not new_beams:
            break

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:num_beams]

    final_beams = completed_beams if completed_beams else beams
    best_seq = sorted(final_beams, key=lambda x: x[1], reverse=True)[0][0]

    generated_ids = best_seq[0, 1:]
    if eos_token_id in generated_ids:
        eos_idx = np.where(generated_ids == eos_token_id)[0][0]
        generated_ids = generated_ids[:eos_idx]

    if len(generated_ids) == 0:
        return "[ERROR] No valid response generated. Try rephrasing your input."

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

# === Debugging: Print execution provider info ===
print("\n[INFO] Execution providers for encoder:", encoder_session.get_providers())
print("\n[INFO] Execution providers for decoder:", decoder_session.get_providers())

# === Test Execution ===
if __name__ == "__main__":
    input_text = "Tell me something interesting about AI."
    print("Input:", input_text)
    print("Output:", generate_text(input_text))