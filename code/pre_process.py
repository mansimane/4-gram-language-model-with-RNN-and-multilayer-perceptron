
#!/usr/bin/env python
import math
'''
Author: Mansi
    -Reads raw text file and dumps processed text file with 'START' and 'END' tags
    -Dumps dictionary of 8000 words if data is training data
    
    Input: text file name
'''
import pickle
import sys
import os
import select
from generic_functions import *
from config_3_2 import *
#Global Variables
max_vocab_len = hyper_para['vocab_size']


def process_line(line, vocab, vocab_len, fd_out, text_file_name):
    line = line.lower()
    line = line.replace('\n', '')
    line = 'START ' + line + ' END'
    # train file check
    if text_file_name == 'train':
        words = line.split()
#        for w in words:
        for i in range(len(words)):
            # in vocab, increase count
            if words[i] in vocab:
                vocab[words[i]] = vocab[words[i]] + 1
            # Not in vocab but max limit not reached yet
            elif words[i] not in vocab and vocab_len < max_vocab_len:
                vocab[words[i]] = 1
                vocab_len = vocab_len + 1  # since calling len(vocab may be computationally expensive, keeping
                # local variable
            # not in vocab, max vocab size reached, increase unknown count
            #possible optimization, write UNK in output instead of that word, so while training everything is known
            #and training will become fast as we don't have to do dict search
            else:
                vocab['UNK'] = vocab['UNK'] + 1
                words[i] = ' UNK '
    else:
        words = line.split()
        #        for w in words:
        vocab = load_obj('vocab')
        for i in range(len(words)):
            # in vocab, increase count
            if words[i] not in vocab:
                words[i] = ' UNK '

    line_updated = ''
    for w in words:
        line_updated = line_updated + ' ' + w
    fd_out.writelines(line_updated + '\n')

    return vocab, vocab_len

#Without main code would be executed even if script is imported as module
def main():
    text_file_name = sys.argv[1].replace('.txt', '')
    vocab = {'START': 0, 'END': 0, 'UNK': 0}
    vocab_len = vocab.__len__()
    Y = {}
    Y_wstar = {}
    Y_star = 0
    cur_path = os.path.abspath(os.path.dirname(__file__))

    print cur_path
    fd_out = open(cur_path + '/../data/' + text_file_name + '_with_tags.txt', 'w+')
    #If text input is given fron pipe
    if select.select([sys.stdin, ], [], [], 0.0)[0]:
        print "Have data, taiking text from command line"
        for line in sys.stdin:
        # Read from text file
            vocab, vocab_len = process_line(line, vocab, vocab_len, fd_out, text_file_name)
    else:
        print "No data, reading text from text file"
        fd_in_name = cur_path + '/../data/' + sys.argv[1]
        i=0
        with open(fd_in_name) as fd_in:         #No need of closing it, as with takes care of it
            for line in fd_in.readlines():
                i = i+1
                vocab, vocab_len = process_line(line, vocab, vocab_len, fd_out, text_file_name)

    fd_out.close()
    assert os.path.exists(cur_path + '/obj/'), ('obj directory does not exist')
    if text_file_name == 'train':
        save_obj(vocab, 'vocab')
    #print vocab


if __name__ == '__main__':
    main()
