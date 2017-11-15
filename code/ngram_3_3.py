import os
import select
import time
from generic_functions import *
from config_3_3 import *
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

    train_p_list = np.zeros((1, epochs))
    train_loss_list = np.zeros((1, epochs))
    #train_acc_list = np.zeros((1,epochs))

    val_loss_list = np.zeros((1, epochs))
    val_p_list = np.zeros((1, epochs))
    #val_acc_list = np.zeros((1, epochs))
    print  hyper_para
    print 'Reading data'
    start = time.time()
    print 'Starting time', start
    x_train, y_train, x_val, y_val = prepare_text_data_ngram(param)

    for epoch in range(epochs):
         train_p = 0
         train_loss = 0
         no_of_batches = 1
         for step in range(max_iter):
             # get mini-batch and
            start_idx = step * batch_size % no_of_train_samples
            end_idx = (step + 1) * batch_size % no_of_train_samples

            if start_idx > end_idx:
                random.shuffle(indices)
                continue
            idx = indices[start_idx: end_idx]

            param_grad, t_loss, t_per = grad_calc_with_tanh(param, x_train[idx, :], y_train[idx], hyper_para)
            param = update_param(param, param_grad, x_train[idx, :], hyper_para)
            train_p += t_per
            train_loss += t_loss
            no_of_batches += 1

         #train_p, train_loss = loss_calc(param, hyper_para, x_train, y_train)
         val_p, val_loss = loss_calc_tanh(param, hyper_para, x_val, y_val)

         train_p = train_p / float(no_of_batches)
         train_loss = train_loss / float(no_of_batches)
         train_p_list[0, epoch] = train_p
         train_loss_list[0, epoch] = train_loss

         val_p_list[0, epoch] = val_p
         val_loss_list[0, epoch] = val_loss

         print 'epoch', epoch, '\ttime', time.clock(), '\tTrain loss', train_loss, '\t Val_loss', val_loss,  '\tTrain Per',train_p, 'Val Per', val_p

    a =  'bs_' + str(hyper_para['batch_size'])
    b = '_lr_' + str(hyper_para['learning_rate'])
    c = '_hl_' + str(hyper_para['hidden_layer_size'])
    d = '_ev_' + str(hyper_para['embed_size'])
    e = '_ep_' + str(hyper_para['epochs'])
    f = hyper_para['non_lin']
    date = time.strftime("%Y-%m-%d_%H_%M")
    save_obj([train_p,train_loss,val_p,val_loss],a+b+c+d+e+f+'results_')
    if os.path.exists('../results/'+a+b+c+d+e+f):
        print '../results/'+a+b+c+d+e+f ,'\texists'
    else:
        os.mkdir('../results/'+a+b+c+d+e+f)
    np.save('../results/'+a+b+c+d+e+f+'/w2', param['w2'] )
    np.save('../results/'+a+b+c+d+e+f+'/b2' , param['b2'])
    np.save('../results/'+a+b+c+d+e+f+'/b1', param['b1'])
    np.save('../results/'+a+b+c+d+e+f+'/w1', param['w1'] )
    np.save('../results/'+a+b+c+d+e+f+'/we', param['we_lookup'])
    np.save('../results/'+a+b+c+d+e+f+'/we_map', param['vocab_dict'])
    np.save('../results/'+a+b+c+d+e+f+'/val_p_list', val_p_list)
    np.save('../results/'+a+b+c+d+e+f+'/val_loss_list', val_loss_list)
    np.save('../results/'+a+b+c+d+e+f+'/train_loss_list', train_loss_list)
    np.save('../results/'+a+b+c+d+e+f+'/train_p_list', train_p_list)

    plot_loss_train_valid(train_p_list, val_p_list, train_loss_list, val_loss_list,  hyper_para)

    #plot_acc_train_valid(train_acc_list, val_acc_list, hyper_para)
if __name__ == '__main__':
    main()
