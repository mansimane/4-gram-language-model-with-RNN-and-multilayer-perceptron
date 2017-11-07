import os
import select
from generic_functions import *
from config_3_2 import *
from ngram_functions import *


def main():
    param = initialize_weights(hyper_para)
    epochs = hyper_para['epochs']
    context_size = hyper_para['context_size']
    vocab_size = hyper_para['vocab_size']
    train_p_list = []
    val_p_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        with open(proc_train_file_name) as fd_in:         #No need of closing it, as with takes care of it
            for line in fd_in.readlines():
                words = line.split()
                #loop over 4 grams
                for i in range(0, len(words) - 3):
                    #send to function to get vec representation
                    ngram_list = words[i:i + 4]
                    x,y = get_word_vec(ngram_list[0:context_size], param, context_size, vocab_size)

                    # #calculate gradients
                    param_grad = grad_calc(param, x, y, hyper_para)
                    # #update parameters
                    # param = update_param(param, param_grad, hyper_para)
                    # #update embedding vecs
                    # param = update_word_vec(param, ngram_list[0:context_size])
                    # #calculate perplexity

            [train_p, val_p, train_loss, val_loss] = loss_calc(param, hyper_para)
            train_p_list.append(train_p)
            train_loss_list.append(train_loss)
            val_p_list.append(val_p)
            val_loss_list.append(val_loss)


if __name__ == '__main__':
    main()
