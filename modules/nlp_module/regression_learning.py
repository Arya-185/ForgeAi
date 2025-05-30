import json
import threading
import time
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define paths
HISTORY_FILE = "modules/nlp_module/chat_history.json"
MODEL_PATH = "onnx/t5-small/"

# Load Tokenizer and Model with error handling
try:
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    tokenizer = None
    model = None

def refine_responses():
    """
    Identifies low-rated responses and adjusts token probabilities to improve future outputs.
    Runs continuously in the background.
    """
    if model is None or tokenizer is None:
        print("[ERROR] Model or tokenizer not loaded. Skipping refinement.")
        return

    while True:
        try:
            time.sleep(600)  # Apply refinements every 10 minutes

            # Load conversation history
            history = load_history()
            if not history:
                print("[INFO] No data available for response refinement.")
                continue

            print("[INFO] Refining model based on low-rated responses...")

            # Gather low-rated interactions (rating < 3.0)
            bad_responses = []
            for entry in history:
                if entry.get("rating") is not None and entry["rating"] < 3.0:
                    bad_responses.append((entry["user_input"], entry["ai_output"]))

            if not bad_responses:
                print("[INFO] Skipping refinement—no low-rated responses found.")
                continue

            # Apply regression-based correction
            model.train()
            for question, bad_answer in bad_responses:
                try:
                    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
                    refined_answer = improve_answer(bad_answer)
                    labels = tokenizer(refined_answer, return_tensors="pt", padding=True, truncation=True).input_ids

                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                except Exception as e:
                    print(f"[ERROR] Failed to process example: {e}")
                    continue

            print("[INFO] Response refinement applied successfully.")

            # Create directory if it doesn't exist
            os.makedirs(MODEL_PATH, exist_ok=True)
            
            # Save updated model
            model.save_pretrained(MODEL_PATH)
            tokenizer.save_pretrained(MODEL_PATH)

        except Exception as e:
            print(f"[ERROR] Refinement process failed: {e}")
            time.sleep(60)  # Wait before retrying

def improve_answer(bad_answer):
    """
    Enhances a poorly rated AI response using structured refinement techniques.
    """
    refined_output = f"Provide a more detailed, structured, and accurate response: {bad_answer}"
    return refined_output

def load_history():
    """Loads conversation history from JSON."""
    try:
        with open(HISTORY_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[ERROR] Failed to load history: {e}")
        return []

# Start background refinement thread only if model is loaded
if model is not None and tokenizer is not None:
    threading.Thread(target=refine_responses, daemon=True).start()
