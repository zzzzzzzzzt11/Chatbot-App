from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QTextEdit
from chatbot_main import ChatBot

class ChatBotGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.chatbot = ChatBot()
        self.setFixedSize(1000, 1000)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ChatBot')

        layout = QVBoxLayout()

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        layout.addWidget(self.chat_history)

        self.user_input = QLineEdit()
        layout.addWidget(self.user_input)

        hbox = QHBoxLayout()
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        hbox.addWidget(self.send_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        hbox.addWidget(self.exit_button)

        layout.addLayout(hbox)

        self.setLayout(layout)
        self.show()

        self.append_bot_response("Hi, I'm a chatbot trained on random dialogs. Would you like to chat with me?\n\n>")

    def append_bot_response(self, message):
        self.chat_history.append(f"bot: {message}")

    def send_message(self):
        user_message = self.user_input.text()
        if user_message:
            self.chat_history.append(f'You: {user_message}')
            self.user_input.clear()
            if self.chatbot.make_exit(user_message):
                self.append_bot_response("Ok, have a great day!")
            else:
                bot_reply = self.chatbot.generate_response(user_message)
                self.append_bot_response(bot_reply)