#!/usr/bin/env python
import math
'''
Author: Mansi
    -Reads raw text file and dumps processed text file with 'START' and 'END' tags
    -Dumps dictionary of 8000 words without any
'''
import re
def tokenizeDoc(cur_doc):
    return re.findall('\\w+',cur_doc)

import sys

dictlist = {}
Y = {}
Y_wstar = {}
Y_star = 0

vocab = []
for line in sys.stdin:
    line = line.lower()
    line = 'START ' + line + ' END\n'
    file_object = open(“../data/filename”, “w”)
    features = tokenizeDoc(doc[2])
    w_star = len(features)
    # for label in labels:
    #     for feature in features:
    #         sys.stdout.write("Y=")
    #         sys.stdout.write(label)
    #         sys.stdout.write(",W=")
    #         sys.stdout.write(feature)
    #         sys.stdout.write("\t")
    #         sys.stdout.write("1")
    #         sys.stdout.write("\n")
    #
    #     sys.stdout.write("Y=")
    #     sys.stdout.write(label)
    #     sys.stdout.write(",W=*")
    #     sys.stdout.write("\t")
    #     sys.stdout.write(str(w_star))
    #     sys.stdout.write("\n")
