import time
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_loss_train_valid (train_p, val_p,train_ce, valid_ce ,  hyper_para):
    date = time.strftime("%Y-%m-%d_%H_%M")
    plt.plot(train_ce)
    plt.plot(valid_ce)
    plt.title('Ngram Neural Network Loss vs Epochs')
    a = 'Para:' + '\tBatch_size=' + str(hyper_para['batch_size'])
    b =  '\tlr=' + str(hyper_para['learning_rate'])
    c = '\thidden: ' + str(hyper_para['hidden_layer_size'])
    plt.suptitle(a + b + c)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train Cross Loss', 'Valid Cross Loss'], loc='upper right')
    plt.savefig('../results/Q3_2_error' + date + '.png')
    plt.show()
    plt.close()
    plt.plot(train_p)
    plt.plot(val_p)
    plt.title('Ngram Neural Network Perplexity vs Epochs')
    plt.suptitle(a + b + c)
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend(['Train Perplexity', 'Val Perplexity'], loc='upper right')
    plt.savefig('../results/Q3_2_per' + date + '.png')
    plt.show()

