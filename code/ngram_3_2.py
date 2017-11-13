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

    train_p_list = np.zeros((1,epochs))
    train_loss_list = np.zeros((1,epochs))
    #train_acc_list = np.zeros((1,epochs))

    val_loss_list = np.zeros((1, epochs))
    val_p_list = np.zeros((1, epochs))
    #val_acc_list = np.zeros((1, epochs))
    print  hyper_para
    print 'Reading data'
    start = time.time()
    print 'Starting time', start
    x_train, y_train, x_val, y_val = prepare_text_data(param)

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
         val_p, val_loss = loss_calc(param, hyper_para, x_val, y_val)

         train_p_list[0, epoch] =train_p
         train_loss_list[0, epoch] = train_loss
#         train_acc_list[0, epoch] = train_acc

         val_p_list[0, epoch] = val_p
         val_loss_list[0, epoch] = val_loss
#         val_acc_list[0, epoch] = val_acc

         print 'epoch', epoch, '\ttime', time.clock(), '\tTrain loss', train_loss, '\t Val_loss', val_loss,  '\tTrain Per',train_p, 'Val Per', val_p
         #if epoch == 100:

    a =  'bs_' + str(hyper_para['batch_size'])
    b = '_lr_' + str(hyper_para['learning_rate'])
    c = '_hl_' + str(hyper_para['hidden_layer_size'])
    d = '_ev_' + str(hyper_para['embed_size'])
    e = '_ep_' + str(hyper_para['epochs'])
    date = time.strftime("%Y-%m-%d_%H_%M")
    save_obj([train_p,train_loss,val_p,val_loss],a+b+c+d+e+'results_')
    if os.path.exists('../results/'+a+b+c+d+e):
        print '../results/'+a+b+c+d+e ,'\texists'
    else:
        os.mkdir('../results/'+a+b+c+d+e)
    np.save('../results/'+a+b+c+d+e+'/w2', param['w2'] )
    np.save('../results/'+ a+b+c+d+e+'/b2' , param['b2'])
    np.save('../results/'+a+b+c+d+e+'/b1', param['b1'])
    np.save('../results/'+a+b+c+d+e+'/w1', param['w1'] )
    np.save('../results/'+a+b+c+d+e+'/we', param['we_lookup'])
    np.save('../results/'+a+b+c+d+e+'/we_map', param['vocab_dict'])
    np.save('../results/'+a+b+c+d+e+'/val_p_list', val_p_list)
    np.save('../results/'+a+b+c+d+e+'/val_loss_list', val_loss_list)
    np.save('../results/'+a+b+c+d+e+'/train_loss_list', train_loss_list)
    np.save('../results/'+a+b+c+d+e+'/train_p_list', train_p_list)

    plot_loss_train_valid(train_p_list, val_p_list, train_loss_list, val_loss_list,  hyper_para)

    #plot_acc_train_valid(train_acc_list, val_acc_list, hyper_para)
if __name__ == '__main__':
    main()
