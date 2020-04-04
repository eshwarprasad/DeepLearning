#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  13 15:35:46 2020

@author: eshwarprasadb
"""

import numpy as np
import os
import pickle
import json

    
np.random.seed(6)

input_folder_path = "./MLDS_hw2_1_data/training_data/feat/"
label_set = "./MLDS_hw2_1_data/training_label.json"

input_files_name = os.listdir(input_folder_path)

input_path = []
ids = []

for file in input_files_name:
    file = input_folder_path + file
    input_path.append(file)
    file= file[:-4].replace(input_folder_path, "")
    ids.append(file)
    
training_set = {}
for path in input_path:
    single_set = np.load(path)
    key = path[: -4].replace(input_folder_path, "") #storing ID
    training_set[key] = single_set
    
label_json = json.load(open(label_set, 'r'))
labels = {}
letters_set = []

special_chars = '\n\t~{|}|_^[\\]@?;./:-*,<>=+%$#()!"`'
#    cleaned_captions = []
for i in label_json:
    captions = i["caption"]
    cleaned_caption = []
    for line in captions:
        for s in special_chars:
            line = line.replace(s, '')
        cleaned_caption.append(line)
    labels[i["id"]] = cleaned_caption
    letters_set += cleaned_caption
    
words = {}
count_letters = 0
for letters in letters_set:
    for word in letters.lower().split(' '):
        words[word] = words.get(word, 0) + 1
    count_letters += 1

word_list = [word for word in words if words[word] >= 3]

chartonum = {}
chartonum['<pad>'] = 3
numtochar = {}


numtochar = {}
numtochar[0] = '<pad>'
numtochar[1] = '<bos>'
numtochar[2] = '<eos>'
numtochar[3] = '<unk>'

chartonum = {}
chartonum['<pad>'] = 0
chartonum['<bos>'] = 1
chartonum['<eos>'] = 2
chartonum['<unk>'] = 3

for index, word in enumerate(word_list):
    chartonum['<pad>'] = index + 4
    numtochar[index + 4] = word

words['<pad>'] = count_letters
words['<bos>'] = count_letters
words['<eos>'] = count_letters
words['<unk>'] = count_letters

pickle.dump(chartonum, open('./chartonum.obj', 'wb'))
pickle.dump(numtochar, open('./numtochar.obj', 'wb'))

Captions = []
Captions_ID = []

words_set = []

#print(ids)
print(labels['UgUFP5baQ9Y_0_10.avi'])
for i in ids:
    for cap in labels[i]:
        Captions_ID.append((training_set[i], cap))
        words = cap.split()
        Captions.append(words)
        for word in words:
            words_set.append(word)

unique_words_set = np.unique(words_set, return_counts=True)[0]
max_length_cap = max([len(i) for i in Captions])
avg_length_cap = np.mean([len(i) for i in Captions])
length_unique_words_set = len(unique_words_set)

print("np.shape(ID_caption): ", np.shape(Captions_ID))
print("Max. length of captions: ", max_length_cap)
print("Avg. length of captions: ", avg_length_cap)
print("Number of unique tokens of captions: ", length_unique_words_set)

print("Shape of features of first video: ", Captions_ID[0][0].shape)
print("ID of first video: ", ids[0])
print("Caption of first video: ", Captions_ID[0][1])

# pickle.dump(ID_caption, open('ID_caption.obj', 'wb'))
pickle.dump(ids, open('ids.obj', 'wb'))
pickle.dump(labels, open('labels.obj', 'wb'))
pickle.dump(training_set, open('training_set.obj', 'wb'))

    
