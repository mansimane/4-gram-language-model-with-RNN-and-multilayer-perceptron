'''
Author: Mansi
    -Reads raw text file and dumps processed text file with 'START' and 'END' tags
    -Dumps dictionary of 8000 words if data is training data
    Input: text file name
'''
import re
import pickle
import sys
import os
import select
from generic_functions import *
import matplotlib.pylab as plt
import time


def main ():
    text_file_name = '../data/train_with_tags.txt'
    vocab = load_obj('vocab')
    ngram_dict = {}
    with open(text_file_name) as fd_in:
        for line in fd_in.readlines():
            words=line.split()
            #Check if lastword is read
            for i in range(0,len(words)-3):
                ngram_list = words[i:i+4]
                ngram_str = '_'.join(ngram_list)
                if ngram_str in ngram_dict:
                    ngram_dict[ngram_str] += 1
                else:
                    ngram_dict[ngram_str] = 1
    print len(ngram_dict)


    ngram_dict_num = {}
    i=0
    for key in ngram_dict.keys():
        ngram_dict_num[i] = ngram_dict[key]
        i = i+1

    lists = sorted(ngram_dict_num.items())
    x, y = zip(*lists)
    plt.plot(x, y)
    date = time.strftime("%Y-%m-%d_%H_%M")

    plt.savefig('../results/Q3_1_ngram' + date + '.png')
    plt.show()
    # #print ngram_dict
    # lists = sorted(ngram_dict.items())  # sorted by key, return a list of tuples
    #
    # x, y = zip(*lists)  # unpack a list of pairs into two tuples
    #
    # plt.plot(x, y)
    # # plt.show()
    # #plt.bar(range(len(ngram_dict)), ngram_dict.values(), align='center')
    # #plt.xticks(range(len(ngram_dict)), ngram_dict.keys())
    # date = time.strftime("%Y-%m-%d_%H_%M")
    # plt.savefig('./results/Q3_1_ngram' + date + '.png')
    # #plt.show()
if __name__ == '__main__':
    main()