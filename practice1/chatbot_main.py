import re
import numpy as np
from main1 import (num_encoder_tokens,
                  input_features_dict,
                  max_encoder_seq_length)
from get_ans import decode_response


class ChatBot:
    # Define negative responses and exit commands
    negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")


    # methods of how to start a conversation
    def start_chat(self):
        user_response = input("Hi, I'm a chatbot trained on random dialogs. Would you like to chat with me?\n\n>")

        # Check if the user response is a negative response
        if user_response in self.negative_responses:
            print("Ok, have a great day!")
            return
        # If it's not a negative response, start a conversation
        self.chat(user_response)

    # Methods for handling conversations
    def chat(self, reply):
        # Loop until the user input contains the exit command
        while not self.make_exit(reply):
            reply = input("bot:" + self.generate_response(reply) + "\n\n>")

    # Method to convert user input into a matrix
    def string_to_matrix(self, user_input):
        # Use regular expressions to split user input into words
        tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
        user_input_matrix = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')
        
        for timestep, token in enumerate(tokens):
            if token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1.
        return user_input_matrix

    # Method to generate responses using seq2seq model
    def generate_response(self, user_input):
        input_matrix = self.string_to_matrix(user_input)
        chatbot_response = decode_response(input_matrix)
        # Remove the <START> and <END> tags
        chatbot_response = chatbot_response.replace("<START>", '')
        chatbot_response = chatbot_response.replace("<END>", '')
        return chatbot_response

    def make_exit(self, reply):
        for exit_command in self.exit_commands:
            if exit_command in reply:
                print("Ok, have a great day!")
                return True
        return False

chatbot = ChatBot()