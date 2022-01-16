#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (Parth Ravindra Rao[partrao], Pratap Roy Choudhury[prroyc], Tanu Kansal[takansal])
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image, ImageDraw, ImageFont
import sys
import re
import numpy as np
from collections import defaultdict

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    #print(im.size)
    #print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }


def preprocessing(file):

    # Since this file is taken from Part 1, it has a lot of words/characters in between that is not required for Part 3. We clean the file here.
    preprocessed_string = []
    data = open(file,'r')

    for text in data:
        new_text = re.sub(' ADJ | ADV | ADP | CONJ | DET | NOUN | NUM | PRON | PRT | VERB | X |\n', '', text)
        new_text = re.sub(' \'\' . ', '\" ', new_text)
        new_text = re.sub(' . . ', '. ', new_text)
        new_text = re.sub('    ', '', new_text)
        
        # new_text = text
        preprocessed_string.append(new_text)

    return preprocessed_string

# Finding emission probability
def emission_prob(test_char, train_char, m):

    temp_list = []

    for col in range(len(test_char[0])):
        for row in range(len(test_char)):
            if test_char[row][col] == train_char[row][col]:
                temp_list.append(1)
            else:
                temp_list.append(0)

    count = np.sum(temp_list)
    emission_probability = len(test_char)*len(test_char[0]) * np.log(m/100) + count * np.log(100-m/100)
    return emission_probability

# Finding initial probability for a particular letter
def FindInitProb(train_doc, temp_letter):

    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    init_prob = defaultdict(lambda: 0)

    # In init_prob_dist we calculated the number of time a sentence begins with a particular letter
    for line in train_doc:
        if line[0] in TRAIN_LETTERS and line[0] in init_prob.keys():
            init_prob[line[0]] += 1

    for letter in TRAIN_LETTERS:
        if letter not in init_prob.keys():
            init_prob[letter] = 1

    # Normalize values in init_prob_dist to get the intial probability distribution
    total_sum = sum(init_prob.values())
    
    for key in init_prob.keys():
        init_prob[key] = init_prob[key]/total_sum

    return init_prob[temp_letter]

# Storing transition probabilities as a dictionary of dictionaries
def FindTransProb(train_doc):

    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "    
    trans_prob = {}
    
    for line in train_doc:
        for index in range(len(line)-1):
            if line[index] in trans_prob.keys():
                if line[index+1] in trans_prob[line[index]].keys():
                        trans_prob[line[index]][line[index+1]]+=1
                else:
                    trans_prob[line[index]][line[index+1]]=1
            else:
                trans_prob[line[index]] = {line[index+1]: 1}

    for value in trans_prob.values():
        for letter in TRAIN_LETTERS:
            if letter not in value.keys():
                value[letter] = 1

    # Normalize values in init_prob_dist to get the transition probability distribution
    for main_letter in trans_prob.values():
        total_sum = sum(main_letter.values())
        for sub_letter in main_letter.keys():
            main_letter[sub_letter] = main_letter[sub_letter]/total_sum
    
    # print(trans_prob)

    return trans_prob


def hmm_viterbi(test_char, train_char, processed_char, m):
    
    best_sequence = ""
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    prob_list = np.zeros((len(TRAIN_LETTERS), len(test_char)))

    for letter_index in range(len(TRAIN_LETTERS)):
        initial_probability = FindInitProb(processed_char, TRAIN_LETTERS[letter_index])
        emission_probability = emission_prob(test_char[0], letters[TRAIN_LETTERS[letter_index]], m)
        prob_list[letter_index][0] = emission_probability + np.log(initial_probability) + 0.1
    
    best_index = np.argmax(prob_list[:,0])
    best_sequence = best_sequence + TRAIN_LETTERS[best_index]

    transition_probability = FindTransProb(processed_char)

    for noisy_index in range(1,len(test_char)):
        for letter_index in range(len(TRAIN_LETTERS)):
            emission_probability = emission_prob(test_char[noisy_index], train_char[TRAIN_LETTERS[letter_index]], m)
            
            prev_prob_list = []
            for temp_letter in range(len(TRAIN_LETTERS)):
                prev_prob_list.append( [prob_list[temp_letter][noisy_index - 1]] )

            temp_calc_val = np.zeros(len(TRAIN_LETTERS))
            for prev_ind in range(len(prev_prob_list)):
                temp_calc_val[prev_ind] = prev_prob_list[prev_ind] + np.log(transition_probability[TRAIN_LETTERS[prev_ind]][TRAIN_LETTERS[letter_index]]) + 0.1
            prob_list[letter_index][noisy_index] = np.max(temp_calc_val) + emission_probability + 0.1

        best_index = np.argmax(prob_list[:, noisy_index])
        best_sequence = best_sequence + TRAIN_LETTERS[best_index]
    return ''.join(best_sequence)

def naive_bayes(test_char, train_char, m):
    
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    best_sequence = ""

    for i in range(len(test_char)):
        
        best_letter = ''
        best_score = -100000
        
        for l in TRAIN_LETTERS:
            emmission_prob = emission_prob(test_char[i], train_char[l], m)
            if emmission_prob > best_score:
                best_letter = l
                best_score = emmission_prob
        best_sequence = best_sequence + best_letter

    return best_sequence

#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

train_img_file, train_txt_file, test_img_file = sys.argv[1:]
letters = load_training_letters(train_img_file)
test_letters = load_letters(test_img_file)
preprocessed_txt = preprocessing(train_txt_file)

# The final two lines of your output should look something like this:
print("Simple: " + naive_bayes(test_letters, letters, 35))
print("   HMM: " + hmm_viterbi(test_letters, letters, preprocessed_txt, 35))