from PyQt5.QtWidgets import QApplication
import sys
from UI import ChatBotGUI

def main():
    app = QApplication(sys.argv)
    chatbot_gui = ChatBotGUI() 
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()