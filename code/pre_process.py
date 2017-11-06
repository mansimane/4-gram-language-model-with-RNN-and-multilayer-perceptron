#!/usr/bin/env python
import math
'''
Author: Mansi
    -Reads raw text file and dumps processed text file with 'START' and 'END' tags
    -Dumps dictionary of 8000 words if data is training data
    
    Input: text file name
'''
import re
import pickle
def tokenizeDoc(cur_doc):
    return re.findall('\\w+',cur_doc)

import sys
import os
import select

def save_obj(obj, name, epoch =-1):
    if epoch is -1:
        epoch = ''
    with open('obj/'+ name + epoch +'.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#Without main code would be executed even if script is imported as module
def main():
    text_file_name = sys.argv[1].replace('.txt', '')
    vocab = {'START': 0, 'END': 0, 'UNK': 0}
    vocab_len = vocab.__len__()
    max_vocab_len = 8000
    Y = {}
    Y_wstar = {}
    Y_star = 0
    cur_path = os.path.abspath(os.path.dirname(__file__))

    print cur_path
    fd = open(cur_path + '/../data/' + text_file_name + '_with_tags.txt', 'w+')
    #If text input is given fron pipe
    if select.select([sys.stdin, ], [], [], 0.0)[0]:
        print "Have data!"
    # Read from text file
    else:
        print "No data"
    for line in sys.stdin:
        line = line.lower()
        line = line.replace('\n', '')
        line = 'START ' + line + ' END\n'
        fd.writelines(line)
        #train file check
        if text_file_name == 'train':
            words = line.split()
            for w in words:
                #in vocab, increase count
                if w in vocab:
                    vocab[w] = vocab[w] + 1
                # Not in vocab but max limit not reached yet
                elif w not in vocab and vocab_len < max_vocab_len:
                    vocab[w] = 1
                    vocab_len = vocab_len + 1   #since calling len(vocab may be computationally expensive, keeping
                                            #local variable
                #not in vocab, max vocab size reached, increase unknown count
                else:
                    vocab[w] = vocab[w] + 1

    fd.close()
    assert os.path.exists(cur_path + '/obj/'), ('obj directory does not exist')
    save_obj(vocab, 'vocab')


if __name__ == '__main__':
    main()