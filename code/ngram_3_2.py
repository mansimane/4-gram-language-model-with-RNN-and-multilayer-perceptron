import os
import select
import time
from generic_functions import *
from config_3_2 import *
from ngram_functions import *
from plotting_funs import *
from prepare_text_data import *
import random

def main():
    param = initialize_weights(hyper_para)
    epochs = hyper_para['epochs']
    context_size = hyper_para['context_size']
    vocab_size = hyper_para['vocab_size']
    batch_size = hyper_para['batch_size']
    no_of_train_samples = hyper_para['total_ngrams_in_tr_data']
    max_iter =  no_of_train_samples / batch_size
    indices = range(no_of_train_samples)
    random.shuffle(indices)

    train_p_list = []
    val_p_list = []
    train_loss_list = []
    val_loss_list = []

    start = time.time()
    print 'start time', start
    #x_train, y_train, x_val, y_val = prepare_text_data(param)
    x_train = load_obj('x_train')
    y_train = np.load('obj/y_train.pkl.npy')
    x_val = load_obj('x_val')
    y_val = load_obj('y_val')

    for epoch in range(epochs):
         for step in range(max_iter):
             # get mini-batch and
            start_idx = step * batch_size % no_of_train_samples
            end_idx = (step + 1) * batch_size % no_of_train_samples

            if start_idx > end_idx:
                random.shuffle(indices)
                continue
            idx = indices[start_idx: end_idx]

            param_grad = grad_calc(param, x_train[idx, :], y_train[idx], hyper_para)
            param = update_param(param, param_grad, x_train[idx, :], hyper_para)



         train_p, train_loss = loss_calc(param, hyper_para, x_train, y_train)
         [val_p, val_loss] = loss_calc(param, hyper_para, x_val, y_val)

         train_p_list.append(train_p)
         train_loss_list.append(train_loss)
         val_p_list.append(val_p)
         val_loss_list.append(val_loss)
         print 'epoch', epoch, '\ttime', time.clock(), '\tTrain loss', train_loss, '\t Val_loss', val_loss
    plot_loss_train_valid(train_p_list, val_p_list,train_loss_list,val_loss_list,  hyper_para)


if __name__ == '__main__':
    main()
