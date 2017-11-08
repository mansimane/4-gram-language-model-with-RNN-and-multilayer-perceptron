import os
import select
from generic_functions import *
from config_3_2 import *
from ngram_functions import *
from plotting_funs import *

def main():
    param = initialize_weights(hyper_para)
    epochs = hyper_para['epochs']
    context_size = hyper_para['context_size']
    vocab_size = hyper_para['vocab_size']
    train_p_list = []
    val_p_list = []
    train_loss_list = []
    val_loss_list = []

    with open(proc_train_file_name) as fd_in:  # No need of closing it, as with takes care of it
        train_data = fd_in.readlines()

        for epoch in range(epochs):
            #Until and unless obj is modified by function, it is pass by reference
            #We don't modify train_data in get_word_vec function hence it is pass by reference
            print 'epoch', epoch
            no_of_ngram_read = 0
            while (no_of_ngram_read <= total_ngrams_in_tr_data):
                ngram_list, x, y = get_word_vec(train_data, hyper_para, param)

                ##calculate gradients
                param_grad = grad_calc(param, x, y, hyper_para)
                ##update parameters
                param = update_param(param, param_grad, ngram_list, hyper_para)
                ##calculate perplexity

                no_of_ngram_read += x.shape[0]
                print no_of_ngram_read
            [train_p, val_p, train_loss, val_loss] = loss_calc(param, hyper_para, train_data)
            train_p_list.append(train_p)
            train_loss_list.append(train_loss)
            val_p_list.append(val_p)
            val_loss_list.append(val_loss)
            print 'epoch', epoch, 'Train loss', train_loss
    plot_ce_train_valid(train_p_list, val_p_list, hyper_para)


if __name__ == '__main__':
    main()
