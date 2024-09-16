from keras.layers import Input, LSTM, Dense
from keras.models import Model
from main1 import (num_encoder_tokens, 
                  num_decoder_tokens, 
                  input_docs,
                  max_encoder_seq_length,
                  max_decoder_seq_length,
                  target_docs,
                  input_features_dict,
                  target_features_dict
                  )
from keras.callbacks import Callback
from keras.models import load_model
import re
import numpy as np

# Initialize a zero matrix to store one-hot encoded data
encoder_input_data = np.zeros(
    (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

# Fill one-hot encoded data
for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
    for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
        # Set one-hot encoding for the current sentence, time step, and word
        encoder_input_data[line, timestep, input_features_dict[token]] = 1.

    for timestep, token in enumerate(target_doc.split()):
        decoder_input_data[line, timestep, target_features_dict[token]] = 1.
        # The time step lag of decoder target data is 1
        if timestep > 0:
            decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.


dimensionality = 256
batch_size = 10
epochs = 1000
valid = 0.2

# Encoder
encoder_inputs = Input(shape=(None, num_encoder_tokens)) 
encoder_lstm = LSTM(dimensionality, return_state=True)  
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)  
encoder_states = [state_hidden, state_cell] 

# Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens)) 
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)  
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states) 
decoder_dense = Dense(num_decoder_tokens, activation='softmax') 
decoder_outputs = decoder_dense(decoder_outputs) 

training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)  # Define the training model

# Compile the model
training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')

# Training the model
history = training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=valid)

# save the model
training_model.save('training_model.keras')


# Automatic saving
class ModelCheckpointEveryNEpochs(Callback):
    def __init__(self, save_path, every_n_epochs, start_epoch):
        super(ModelCheckpointEveryNEpochs, self).__init__()
        self.save_path = save_path
        self.every_n_epochs = every_n_epochs
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        epoch_number = self.start_epoch + epoch + 1
        if epoch_number % self.every_n_epochs == 0:
            model_path = f"{self.save_path}_epoch_{epoch_number}.keras"
            self.model.save(model_path)
            print(f"Model saved to {model_path}")

# Get the epoch number of the read model
def extract_epoch_from_filename(filename):
    match = re.search(r'_epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return 0
    
training_model.summary()

#scores = training_model.evaluate([encoder_input_data, decoder_input_data], decoder_target_data, verbose=2)
#print('\033[91m', 'the model had', round(scores[1] * 100, 4), "% correctly")

# Load the saved model
training_model = load_model('training_model.keras')