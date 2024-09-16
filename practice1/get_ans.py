import numpy as np 
from keras.models import Model
from keras.layers import Input
from keras.models import load_model
from train import (decoder_lstm,
                   decoder_inputs,
                   decoder_dense)
from main1 import (num_decoder_tokens,
                  target_features_dict,
                  reverse_target_features_dict,
                  max_decoder_seq_length)

# Load the saved model
training_model = load_model('training_model.keras')

# Get encoder input and output
encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)
latent_dim = 256

#Decoder status input
decoder_state_input_hidden = Input(shape=(latent_dim,))
decoder_state_input_cell = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

# Decoder output
decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_hidden, state_cell]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def decode_response(test_input):
    # Get the output state of the encoder to pass to the decoder
    states_value = encoder_model.predict(test_input)
    target_seq = np.zeros((1, 1, num_decoder_tokens))

    # Set the first position of the target sequence to the start marker (<START>)
    target_seq[0, 0, target_features_dict['<START>']] = 1.

    # Store the decoded sentences
    decoded_sentence = ''

    stop_condition = False
    while not stop_condition:
        # Predict the probability and state of output tokens
        output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]
        decoded_sentence += " " + sampled_token

        # Stop decoding if the end tag (<END>) is encountered or the sentence exceeds the maximum length
        if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update target sequence
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update Status
        states_value = [hidden_state, cell_state]

    return decoded_sentence