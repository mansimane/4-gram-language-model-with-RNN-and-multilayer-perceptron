import time
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

def plot_loss_train_valid (train_p, val_p,train_ce, valid_ce ,  hyper_para):
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    date = time.strftime("%Y-%m-%d_%H_%M")
    plt.title('Ngram Neural Network Loss vs Epochs')
    plt.legend(['Train Cross Loss', 'Valid Cross Loss'], loc='upper right')
    # a = 'Para:' + '\tBatch_size=' + str(hyper_para['batch_size'])
    # b =  '\tlr=' + str(hyper_para['learning_rate'])
    # c = '\thidden: ' + str(hyper_para['hidden_layer_size'])
    a =  'bs_' + str(hyper_para['batch_size'])
    b = '_lr_' + str(hyper_para['learning_rate'])
    c = '_hl_' + str(hyper_para['hidden_layer_size'])
    d = '_ev_' + str(hyper_para['embed_size'])
    e = '_ep_' + str(hyper_para['epochs'])

    plt.suptitle(a + b + c + d)
    ax = plt.subplot(111)
    ax.set_xlim(1,train_ce.shape[1])

    plt.plot(train_ce[0,:])
    plt.plot(valid_ce[0,:])
    plt.show()
    plt.savefig('../results/'+a+b+c+d+e+'/Q3_2_error' + date + '.png')
    plt.close()

    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    ax = plt.subplot(111)
    ax.set_xlim(1,val_p.shape[1])
    plt.plot(train_p[0,:])
    plt.plot(val_p[0,:])
    plt.title('Ngram Neural Network Perplexity vs Epochs')
    plt.suptitle(a + b + c)
    plt.legend(['Train Perplexity', 'Val Perplexity'], loc='upper right')
    plt.show()
    plt.savefig('../results/'+a+b+c+d+e+'Q3_2_per' + date + '.png')
    plt.close()

def plot_acc_train_valid (train_acc_list, val_acc_list,  hyper_para):
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    date = time.strftime("%Y-%m-%d_%H_%M")
    plt.title('Ngram Neural Network Accuracy vs Epochs')
    # a = 'Para:' + '\tBatch_size=' + str(hyper_para['batch_size'])
    # b =  '\tlr=' + str(hyper_para['learning_rate'])
    # c = '\thidden: ' + str(hyper_para['hidden_layer_size'])
    a =  'bs_' + str(hyper_para['batch_size'])
    b = '_lr_' + str(hyper_para['learning_rate'])
    c = '_hl_' + str(hyper_para['hidden_layer_size'])
    d = '_ev_' + str(hyper_para['embed_size'])
    e = '_ep_' + str(hyper_para['epochs'])

    plt.suptitle(a + b + c + d)
    ax = plt.subplot(111)
    ax.set_xlim(1,train_acc_list.shape[1])
    plt.plot(train_acc_list[0,:])
    plt.plot(val_acc_list[0,:])
    plt.legend(['Train Accuracy', 'Valid Accuray'], loc='upper right')
    plt.show()
    plt.savefig('../results/'+a+b+c+d+e+'/Q3_2_acc' + date + '.png')
    plt.close()

def vis_embedding(param, hyper_para):
    indices = np.random.randint(0, 8000, size=500)
    we_lookup = param['we_lookup']
    w1 = param['w1']
    w2 = param['w2']
    b2 = param['b2']
    b1 = param['b1']
    vocab_dict = param['vocab_dict']
    vocab_dict_inv = param['vocab_dict_inv']

    if hyper_para['embed_size'] == 2:
        y = [2.56422, 3.77284, 3.52623, 3.51468, 3.02199]
        z = [0.15, 0.3, 0.45, 0.6, 0.75]
        n = [58, 651, 393, 203, 123]

        fig, ax = plt.subplots()
        ax.scatter(z, y)

        for i, txt in enumerate(n):
            ax.annotate(txt, (z[i], y[i]))
    else:
        print "Function not designed for embedding size more than 2"