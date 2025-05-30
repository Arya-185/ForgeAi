import json
import threading
import time
import os
from model_handler import generate_text
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define paths
HISTORY_FILE = "modules/nlp_module/chat_history.json"
MODEL_PATH = "onnx/t5-small/"

# Load Tokenizer and Model for fine-tuning with error handling
try:
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    tokenizer = None
    model = None

def retrain_model(conversation_history=None):
    """
    Periodically retrains the model using past conversations stored in history.
    Runs in the background without blocking chat interaction.
    
    Args:
        conversation_history (list, optional): List of conversation entries to use for training.
    """
    if model is None or tokenizer is None:
        print("[ERROR] Model or tokenizer not loaded. Skipping retraining.")
        return

    try:
        # Use provided history or load from file
        history = conversation_history if conversation_history is not None else load_history()
        if not history:
            print("[INFO] No sufficient data for incremental learning.")
            return

        print("[INFO] Retraining model with new interactions...")

        # Prepare training examples
        new_data = []
        for entry in history:
            if entry.get("rating") is not None and entry["rating"] >= 3.5:  # Use only well-rated responses
                new_data.append((entry["user_input"], entry["ai_output"]))

        if not new_data:
            print("[INFO] Skipping retraining—no high-rated responses yet.")
            return

        # Apply incremental fine-tuning
        model.train()
        for question, answer in new_data:
            try:
                inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
                labels = tokenizer(answer, return_tensors="pt", padding=True, truncation=True).input_ids
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
            except Exception as e:
                print(f"[ERROR] Failed to process training example: {e}")
                continue

        print("[INFO] Model fine-tuned successfully with new data.")

        # Create directory if it doesn't exist
        os.makedirs(MODEL_PATH, exist_ok=True)
        
        # Save updated model
        model.save_pretrained(MODEL_PATH)
        tokenizer.save_pretrained(MODEL_PATH)

    except Exception as e:
        print(f"[ERROR] Retraining process failed: {e}")

def load_history():
    """Loads past conversation history from JSON."""
    try:
        with open(HISTORY_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[ERROR] Failed to load history: {e}")
        return []

# Start background learning thread only if model is loaded
if model is not None and tokenizer is not None:
    threading.Thread(target=lambda: retrain_model(), daemon=True).start()
