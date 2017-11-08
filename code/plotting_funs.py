import time
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_ce_train_valid (train_ce, valid_ce, hyper_para):
    date = time.strftime("%Y-%m-%d_%H_%M")
    plt.plot(train_ce)
    plt.plot(valid_ce)
    plt.title('AutoEncoder Error vs Epochs')
    a = 'Para:' + '\tBatch_size=' + str(hyper_para['batch_size'])
    b =  '\tlr=' + str(hyper_para['learning_rate'])
    c = '\tk=' + str(hyper_para['k'])
    d = '\tepochs: ' + str(hyper_para['epochs'])
    plt.suptitle(a + b + c + d)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend(['Train Cross Loss', 'Valid Cross Loss'], loc='upper right')
    plt.savefig('../results/Q3_2_error' + date + '.png')
    plt.show()
