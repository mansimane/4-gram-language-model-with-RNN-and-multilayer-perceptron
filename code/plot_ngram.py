'''
Author: Mansi
    -Reads raw text file and dumps processed text file with 'START' and 'END' tags
    -Dumps dictionary of 8000 words if data is training data
    Input: text file name either val.txt or train.txt
'''
import re
import pickle
import sys
import os
import select
import operator
from generic_functions import *
import matplotlib.pylab as plt
import time


def main ():
    #text_file_name = '../data/val_with_tags.txt'
    text_file_name = '../data/train_with_tags.txt'
    vocab = load_obj('vocab')
    ngram_dict = {}
    with open(text_file_name) as fd_in:
        for line in fd_in.readlines():
            words=line.split()
            #Check if lastword is read
            for i in range(0,len(words)-3):
                ngram_list = words[i:i+4]
                ngram_str = '  '.join(ngram_list)
                if ngram_str in ngram_dict:
                    ngram_dict[ngram_str] += 1
                else:
                    ngram_dict[ngram_str] = 1
    print len(ngram_dict)

    total_ngram_cnt = 0
    for key in ngram_dict.keys():
        total_ngram_cnt += ngram_dict[key]
    print 'Total ngrams in the train file are', total_ngram_cnt

    ngram_list = sorted(ngram_dict.items(), key=operator.itemgetter(1), reverse=True)

    x, y = zip(*ngram_list)
    #Will store integer for every unique ngram
    x_int = []
    for i in range(x.__len__()):
        x_int.append(i)
    plt.plot(x_int, y)
    date = time.strftime("%Y-%m-%d_%H_%M")
    plt.savefig('../results/Q3_1_ngram' + date + '.png')
    plt.show()

    text_file_name = '../results/Q3_1_top_50.txt'
    with open(text_file_name,'a') as fd_out:
        for i in range(50):
            fd_out.write(x[i] + '\t\t' + str(y[i]) + '\n')

if __name__ == '__main__':
    main()

