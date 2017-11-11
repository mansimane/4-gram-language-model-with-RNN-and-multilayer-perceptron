import time
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_loss_train_valid (train_p, val_p,train_ce, valid_ce ,  hyper_para):
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    date = time.strftime("%Y-%m-%d_%H_%M")
    plt.title('Ngram Neural Network Loss vs Epochs')
    a = 'Para:' + '\tBatch_size=' + str(hyper_para['batch_size'])
    b =  '\tlr=' + str(hyper_para['learning_rate'])
    c = '\thidden: ' + str(hyper_para['hidden_layer_size'])
    plt.suptitle(a + b + c)
    ax = plt.subplot(111)
    ax.set_xlim(1,train_ce.shape[1])
    plt.plot(train_ce[0,:])
    plt.plot(valid_ce[0,:])
    plt.legend(['Train Cross Loss', 'Valid Cross Loss'], loc='upper right')
    plt.show()
    plt.savefig('../results/Q3_2_error' + date + '.png')
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
    plt.savefig('../results/Q3_2_per' + date + '.png')
    plt.show()

