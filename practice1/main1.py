import tensorflow as tf
print(tf.__version__)
import re
import random

import time
print("start")
time.sleep(1)
print("end")



#     file path
data_path = r'C:\Users\zzzzt\Desktop\11\human_text.txt'
data_path2 = r'C:\Users\zzzzt\Desktop\11\robot_text.txt'

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
with open(data_path2, 'r', encoding='utf-8') as f:
    lines2 = f.read().split('\n')

# Processing lines
lines = [re.sub(r"\[\w+\]", 'hi', line) for line in lines]
lines = [" ".join(re.findall(r"\w+", line)) for line in lines]

# Processing lines2
lines2 = [re.sub(r"\[\w+\]", '', line) for line in lines2]
lines2 = [" ".join(re.findall(r"\w+", line)) for line in lines2]

# Paired Rows
pairs = list(zip(lines, lines2))

# Shuffle (optional）
random.shuffle(pairs)

input_docs = []
target_docs = []
input_tokens = set()
target_tokens = set()

dataset_amount = 10 


for line in pairs[:dataset_amount]:
    input_doc, target_doc = line[0], line[1]
    input_docs.append(input_doc)
    # Process the target sentence: split words and add start and end markers
    target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
    target_doc = '<START> ' + target_doc + ' <END>'
    target_docs.append(target_doc)

    # Add each sentence's words to the vocabulary
    for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
        if token not in input_tokens:
            input_tokens.add(token)
    for token in target_doc.split():
        if token not in target_tokens:
            target_tokens.add(token)

# Convert the vocabulary into a sorted list
input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))

num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

# Create a dictionary mapping words to indices
input_features_dict = dict([(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict([(token, i) for i, token in enumerate(target_tokens)])

# Create a dictionary mapping index to words
reverse_input_features_dict = dict((i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict((i, token) for token, i in target_features_dict.items())

# Get the maximum length of input and target sentences
max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])