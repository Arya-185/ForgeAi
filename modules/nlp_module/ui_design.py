from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel, QSlider
)

class ChatUI(QWidget):
    def __init__(self):
        super().__init__()

        # Layout Setup
        self.layout = QVBoxLayout()

        # Chat Display (Read-Only)
        self.chat_display = QTextEdit(self)
        self.chat_display.setReadOnly(True)
        self.layout.addWidget(self.chat_display)

        # Input Box
        self.input_box = QLineEdit(self)
        self.layout.addWidget(self.input_box)

        # Send Button
        self.send_button = QPushButton("Send", self)
        self.layout.addWidget(self.send_button)

        # Rating System (Stars 0-5 in 0.5 increments)
        self.rating_label = QLabel("Rate AI Response (0-5):", self)
        self.layout.addWidget(self.rating_label)

        self.rating_slider = QSlider()
        self.rating_slider.setOrientation(1)  # Horizontal slider
        self.rating_slider.setMinimum(0)
        self.rating_slider.setMaximum(10)  # 10 steps for 0.5 increments
        self.rating_slider.setTickInterval(1)
        self.rating_slider.setTickPosition(QSlider.TicksBelow)
        self.layout.addWidget(self.rating_slider)

        self.submit_rating_button = QPushButton("Submit Rating", self)
        self.layout.addWidget(self.submit_rating_button)

        # Set Layout
        self.setLayout(self.layout)