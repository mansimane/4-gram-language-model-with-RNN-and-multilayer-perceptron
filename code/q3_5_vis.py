import numpy as np
from generic_functions import *
from config_3_3 import *
from ngram_functions import *
from plotting_funs import *
import sys
import time
def main():
    th_words = [['government', 'of', 'united'], ['city', 'of', 'new'], ['life', 'in', 'the'],
                ['he', 'is', 'the'], ['the','new', 'york']]
    x_map = np.zeros((5,3))

    dir = '../results/bs_512_lr_0.02_hl_128_ev_2_ep_100_tanh_/'
    w1 = np.load(dir + 'w1.npy')
    w2 = np.load(dir + 'w2.npy')
    b1 = np.load(dir + 'b1.npy')
    b2 = np.load(dir + 'b2.npy')
    we_lookup = np.load(dir+'we.npy')
    vocab_dict = np.load(dir+'we_map.npy')

    vocab_dict = vocab_dict.item()
    vocab_dict_inv = {y:x for x, y in vocab_dict.iteritems()}

    param = {}
    param['we_lookup'] = we_lookup
    param['vocab_dict'] = vocab_dict
    param['vocab_dict_inv'] = vocab_dict_inv
    param['w1'] = w1
    param['w2'] = w2
    param['b1'] = b1
    param['b2'] = b2

    vocab_size = hyper_para['vocab_size']
    no_of_words = 500
    ind_arr = np.random.randint(vocab_size, size=no_of_words)
    x = we_lookup[ind_arr, 0]
    y = we_lookup[ind_arr, 1]
    ann = []
    for ind in ind_arr:
        ann.append(vocab_dict_inv[ind])

    # y = [2.56422, 3.77284, 3.52623, 3.51468, 3.02199]
    # z = [0.15, 0.3, 0.45, 0.6, 0.75]
    # n = [58, 651, 393, 203, 123]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, txt in enumerate(ann):
        ax.annotate(txt, (x[i], y[i]))
    date = time.strftime("%Y-%m-%d_%H_%M")
    plt.title('Visualization of word vectors'+ str(no_of_words))

    fig = plt.gcf()
    fig.savefig('../results/'+'/Q3_5_vis'+ str(no_of_words)+ '_' + date + '.png')
    plt.show()


if __name__ == '__main__':
    main()