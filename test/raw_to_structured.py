import os
import requests
import onnxruntime as ort
from transformers import T5Tokenizer
import numpy as np

# Define paths to your models
onnx_model_dir = "../onnx/t5-small/"
encoder_path = os.path.join(onnx_model_dir, "encoder_model.onnx")
decoder_path = os.path.join(onnx_model_dir, "decoder_model.onnx")


encoder_session = ort.InferenceSession(encoder_path, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
decoder_session = ort.InferenceSession(decoder_path, providers=["DmlExecutionProvider", "CPUExecutionProvider"])

print("Encoder active execution provider:", encoder_session.get_providers())
print("Decoder active execution provider:", decoder_session.get_providers())
# Function to download ONNX model if missing
def download_onnx_model():
    model_url = "https://huggingface.co/ken11/t5-small-onnx/resolve/main/model.onnx"  # Update if needed
    os.makedirs(onnx_model_dir, exist_ok=True)

    if not os.path.exists(encoder_path):
        print("Downloading ONNX model...")
        response = requests.get(model_url, stream=True)

        if response.status_code == 200:
            with open(encoder_path, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully!")
        else:
            print("Failed to download model. Check the URL.")


# Check if model exists, else download it
if not os.path.exists(encoder_path):
    download_onnx_model()

# Load tokenizer properly
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

# Load ONNX models
if os.path.exists(encoder_path) and os.path.exists(decoder_path):
    encoder_session = ort.InferenceSession(encoder_path)
    decoder_session = ort.InferenceSession(decoder_path)

    import numpy as np


    def generate_text(raw_text):
        print("\n[STEP 1] Raw Input Text:", raw_text)

        tokens = tokenizer(raw_text, return_tensors="np")
        print("\n[STEP 2] Tokenized Input:")
        print("Input IDs:", tokens["input_ids"])
        print("Attention Mask:", tokens["attention_mask"])

        # Encode text using ONNX model
        encoder_outputs = encoder_session.run(
            None,
            {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"]
            }
        )
        print("\n[STEP 3] Encoder Outputs Shape:", encoder_outputs[0].shape)

        # Initialize decoder with the right start token (usually decoder_start_token_id or pad_token_id)
        # decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
        # decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)  # Current
        # Change to:
        decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
        print("\n[STEP 4] Decoder Input IDs (Start Token):", decoder_input_ids)

        # Greedy decoding loop for a fixed max length
        max_length = 1024
        output_ids = decoder_input_ids
        for step in range(max_length):
            decoder_outputs = decoder_session.run(
                None,
                {
                    "input_ids": output_ids,
                    "encoder_attention_mask": tokens["attention_mask"],
                    "encoder_hidden_states": encoder_outputs[0],
                }
            )
            # decoder_outputs[0] shape: (batch, seq_len, vocab_size)
            logits = decoder_outputs[0]
            next_token_logits = logits[:, -1, :]
            next_token_id = np.argmax(next_token_logits, axis=-1).reshape(1, 1)
            output_ids = np.concatenate([output_ids, next_token_id], axis=1)
            if next_token_id[0, 0] == tokenizer.eos_token_id:
                break

        # Remove the initial start token
        generated_ids = output_ids[0, 1:]
        print("\n[STEP 6] Output IDs before decoding:", generated_ids)

        if len(generated_ids) == 0 or all(generated_ids == tokenizer.pad_token_id):
            print("\n[ERROR] Output contains only PAD tokens or is empty.")
            return "<ERROR: Invalid model output>"

        try:
            decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if not decoded_text.strip():
                print("\n[ERROR] Decoded text is empty after removing special tokens.")
                return "<ERROR: Could not decode output>"
        except Exception as e:
            print("\n[ERROR] Issue during decoding:", e)
            decoded_text = "<ERROR: Could not decode output>"

        return decoded_text


    # Example usage
    raw_input = "Explain the key features of an AI-powered tutor app that connects students and allows payments."
    print("\n[FINAL OUTPUT] Refined Description:", generate_text(raw_input))
else:
    print("ONNX model is missing. Check if both 'encoder_model.onnx' and 'decoder_model.onnx' exist.")
