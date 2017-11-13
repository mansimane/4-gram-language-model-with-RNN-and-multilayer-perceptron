import numpy as np
from generic_functions import *
from config_3_2 import *
from ngram_functions import *

def main():
    th_words = [['government', 'of', 'united'], ['city', 'of', 'new'], ['life', 'in', 'the'],
                ['he', 'is', 'the'], ['the','new', 'york']]
    x_map = np.zeros((5,3))

    setting = 'bs_256_lr_0.04_hl_128_ev_2_ep_2'
    # w1 = np.load('../results/' + setting + '/w1.npy')
    #
    # w2 = np.load('../results/' + setting + '/w2.npy')
    #
    # we_lookup = np.load('../results/' + setting + '/we.npy')
    #
    # b1 = np.load('../results/' + setting + '/b1.npy')
    #
    # b2 = np.load('../results/' + setting + '/b2.npy')
    #vocab_dict = np.load('../results/' + setting + '/we_map.npy')
    w1 = np.load('obj/w1bs_256_lr_0.01_hl_ 128_ev_16_ep_100.npy')

    w2 = np.load('obj/w2bs_256_lr_0.01_hl_ 128_ev_16_ep_100.npy')

    b1 = np.load('obj/b1bs_256_lr_0.01_hl_ 128_ev_16_ep_100.npy')

    b2 = np.load('obj/b2bs_256_lr_0.01_hl_ 128_ev_16_ep_100.npy')

    we_lookup = np.load('obj/webs_256_lr_0.01_hl_ 128_ev_16_ep_100.npy')

    vocab_dict = np.load('obj/we_mapbs_256_lr_0.01_hl_ 128_ev_16_ep_100.npy')
    vocab_dict = vocab_dict.item()
    vocab_dict_inv = {y:x for x, y in vocab_dict.iteritems()}

    #vocab = load_obj('vocab')
    param = {}
    param['we_lookup'] = we_lookup
    param['vocab_dict'] = vocab_dict
    param['vocab_dict_inv'] = vocab_dict_inv
    param['w1'] = w1
    param['w2'] = w2
    param['b1'] = b1
    param['b2'] = b2

    for line in th_words:
        for word in line:
            if word not in vocab_dict.keys():
                print word, '\tNot in vocab'


    for i in range(len(th_words)):
        for j in range(len(th_words[i])):
            x_map[i,j] = vocab_dict[th_words[i][j]]
        cnt = 0
        next_word = ''
        while (next_word != 'END') and cnt < 10:
            next_word = predict_word(x_map[i,:], param, hyper_para)
            cnt = cnt + 1
            print next_word
            x_map[i,0] = x_map[i,1]
            x_map[i,1] = x_map[i,2]
            x_map[i,2] = vocab_dict[next_word]
        print '\n\n next'

if __name__ == '__main__':
    main()