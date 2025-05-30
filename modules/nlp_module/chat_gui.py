import sys
import json
import threading
import time
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QPushButton, QLineEdit, QLabel, QSlider, \
    QWidget, QMessageBox
from model_handler import generate_text  # Import NLP model inference
from incremental_learning import retrain_model  # Background learning process

# Path to store conversation history
HISTORY_FILE = "modules/nlp_module/chat_history.json"


class ChatGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window Configuration
        self.setWindowTitle("Agentic AI Chat")
        self.setGeometry(300, 100, 600, 500)

        # Layout Setup
        layout = QVBoxLayout()

        self.chat_display = QTextEdit(self)
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        self.input_box = QLineEdit(self)
        layout.addWidget(self.input_box)

        self.send_button = QPushButton("Send", self)
        self.send_button.clicked.connect(self.process_chat)
        layout.addWidget(self.send_button)

        # Rating System (Stars 0-5 in 0.5 increments)
        self.rating_label = QLabel("Rate AI Response (0-5):", self)
        layout.addWidget(self.rating_label)

        self.rating_slider = QSlider()
        self.rating_slider.setOrientation(1)  # Horizontal slider
        self.rating_slider.setMinimum(0)
        self.rating_slider.setMaximum(10)  # 10 steps for 0.5 increments
        self.rating_slider.setTickInterval(1)
        self.rating_slider.setTickPosition(QSlider.TicksBelow)
        layout.addWidget(self.rating_slider)

        self.submit_rating_button = QPushButton("Submit Rating", self)
        self.submit_rating_button.clicked.connect(self.store_feedback)
        layout.addWidget(self.submit_rating_button)

        # Set Main Widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Load conversation history
        self.conversation_history = self.load_history()

        # Start background learning thread
        self.learning_thread = threading.Thread(target=self.run_background_learning, daemon=True)
        self.learning_thread.start()

    def process_chat(self):
        """Handles user input, generates AI response, and updates chat history."""
        try:
            user_input = self.input_box.text().strip()
            if not user_input:
                return

            self.chat_display.append(f"You: {user_input}")

            # Get AI-generated response
            ai_response = generate_text(user_input)
            if ai_response.startswith("<ERROR:"):
                QMessageBox.warning(self, "Error", f"Failed to generate response: {ai_response}")
                return

            # Display AI response
            self.chat_display.append(f"AI: {ai_response}")

            # Store interaction in history
            self.store_interaction(user_input, ai_response)

            # Clear input field
            self.input_box.clear()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def store_interaction(self, user_input, ai_response):
        """Saves conversation history persistently."""
        try:
            self.conversation_history.append({
                "user_input": user_input,
                "ai_output": ai_response,
                "rating": None  # Default before user rates
            })
            self.save_history()
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to store interaction: {str(e)}")

    def store_feedback(self):
        """Stores the user rating for the last AI response."""
        try:
            if not self.conversation_history:
                QMessageBox.warning(self, "Warning", "No conversation to rate.")
                return

            rating = self.rating_slider.value() / 2  # Convert from scale 0–10 to 0–5
            last_entry = self.conversation_history[-1]
            last_entry["rating"] = rating
            self.save_history()
            self.chat_display.append(f"Rating submitted: {rating} stars")
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to store rating: {str(e)}")

    def save_history(self):
        """Writes conversation history to a JSON file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            
            with open(HISTORY_FILE, "w") as file:
                json.dump(self.conversation_history, file, indent=4)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to save history: {str(e)}")

    def load_history(self):
        """Loads past conversation history from JSON."""
        try:
            if not os.path.exists(HISTORY_FILE):
                return []
                
            with open(HISTORY_FILE, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            QMessageBox.warning(self, "Warning", f"Failed to load history: {str(e)}")
            return []

    def run_background_learning(self):
        """Runs model retraining in background continuously without blocking chat."""
        while True:
            try:
                time.sleep(60)  # Retrain every 60 seconds
                retrain_model(self.conversation_history)  # Pass conversation history for learning
            except Exception as e:
                print(f"[ERROR] Background learning failed: {e}")
                time.sleep(60)  # Wait before retrying


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatGUI()
    window.show()
    sys.exit(app.exec_())
