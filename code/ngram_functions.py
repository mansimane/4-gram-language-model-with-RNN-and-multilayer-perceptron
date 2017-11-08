import numpy as np
from generic_functions import *
from ngram_functions import *

def initialize_weights(hyper_para):
    '''
    Initializes all weights

    :param hyper_para:
    :return:
        vocab_size dictionary with words as keys and values: a list, whose first element
        is integer mapped to that word and second element is embedding vector
    '''
    mu = hyper_para['w_init_mu']
    sigma = hyper_para['w_init_sig']
    embed_size = hyper_para['embed_size']
    vocab_size = hyper_para['vocab_size']

    #Lookup table 8000x16, dictionary
    we_lookup = load_obj('vocab')
    i=0
    for key in we_lookup:
        we_lookup[key] = [i,np.random.normal(mu, sigma, (1,embed_size))]
        i = i + 1

    #Weights for activation layer 1, 48x128
    input_layer_size = hyper_para['context_size']*hyper_para['embed_size']
    hidden_layer_size = hyper_para['hidden_layer_size']

    w1 = np.random.normal(mu, sigma, (input_layer_size * hidden_layer_size))
    w1 = w1.reshape(input_layer_size, hidden_layer_size)
    b1 = np.random.normal(mu, sigma, (1, hidden_layer_size))

    #Weights for activation layer 2, 128x8000
    w2 = np.random.normal(mu, sigma, (hidden_layer_size * vocab_size))
    w2 = w2.reshape(hidden_layer_size, vocab_size)
    b2 = np.random.normal(mu, sigma, (1, vocab_size))

    param = (we_lookup, w1, w2, b1, b2)
    return param

def get_word_vec(train_data, hyper_para, param):
    we_lookup, w1, w2, b1, b2 = param
    vocab_size = hyper_para['vocab_size']
    context_size = hyper_para['context_size']
    embed_size = hyper_para['embed_size']
    if not hasattr(get_word_vec, "line_idx"):
        get_word_vec.line_idx = 0
    if not hasattr(get_word_vec, "line_no"):
        get_word_vec.line_no = 0
    if not hasattr(get_word_vec, "max_line_no"):
        get_word_vec.max_line_no = len(train_data)

    x = np.zeros((0, embed_size))
    y_batch = np.zeros((0, vocab_size))

    while (x.shape[0] != hyper_para['batch_size']):
        line = train_data[get_word_vec.line_no]
        words = line.split()
        # Loop over lines
        for i in range(get_word_vec.line_idx, len(words)-3):
            ngram_list = words[i: i+ hyper_para['no_of_grams']]
            #Create vector from ngrams
            for j in range(0, context_size):
                x = np.append(x, we_lookup[ngram_list[i][1]], axis=0)
            y = np.zeros((1, vocab_size))
            y[we_lookup[ngram_list[context_size]][0]] = 1
            y_batch = np.append(y_batch, y, axis=0)
            get_word_vec.line_idx += 1
            if x.shape[0] == hyper_para['batch_size']:
                return x,y

        get_word_vec.line_idx = 0
        get_word_vec.line_no += 1
        #End of file, batch size can be smaller than programmed on for last one
        if get_word_vec.line_no == get_word_vec.max_line_no:
            return x,y

    return x, y

def update_param (param, param_grad, hyper_parameters):
    """Update the parameters with sgd with momentum
  Args:
        param: tuple
        param_grad: tuple
        hyper_parameters: dict
  Returns:
    """
    lr = hyper_parameters['learning_rate']
    for i in range(len(param)):
        param[i] = param[i] - (lr * param_grad[i])

    return param
def update_word_vec(param, ngram_list):

    return param
def grad_calc (param, x, y, hyper_para):
    '''

    :param param:
    :param x: Embedding vecotrs for 3 context words, 1x48
    :param hyper_para:
    :return:
    '''
    we_lookup, w1, w2, b1, b2 = param

    # #### Forward pass
    a1 = act_forward(x, w1, b1) #nx100 = nx48 * 48*128

    a2 = act_forward(a1, w2, b2)    #nx100 = nx128 * 128x8000

    y_pred = softmax_forward(a2)

    ####### Backward Pass
    d3 = softmax_back(y_pred, y)  # nx8000

    [w2_grad, b2_grad, d2] = act_back(a1, a2, d3, w2, b2)

    [w1_grad, b1_grad, d1] = act_back(x, a1, d2, w2, b2)

    we_grad = d1
    param_grad = (we_grad, w1_grad, w2_grad, b1_grad, b2_grad)
    return param_grad

def loss_calc(param,  hyper_para):

    # w1 = param['w1']  # 784*100
    # w2 = param['w2']  # 100x784
    # b1 = param['b1']  # 1x100
    # b2 = param['b2']  # 1x784
    #
    # ###### Drop out during testing
    # mask = np.ones((xtrain.shape)) * (1.0 - hyper_para['drop_out'])
    # xtrain = np.multiply(xtrain, mask)
    #
    # #### Forward pass
    # a1 = act_forward(xtrain, w1, b1)  # nx100 = nx784 * 784*100
    #
    # h1 = sigmoid_forward(a1)  # n x 100 #same as before,
    #
    # a2 = act_forward(h1, w2, b2)  # nx100 = nx100 * 100x100
    #
    # x_hat = sigmoid_forward(a2)  # n x 100 #same as before,
    #
    # loss = (xtrain * np.log(x_hat)) + ((1 - xtrain) * np.log(1 - x_hat))
    # loss = -loss
    # loss = np.sum(loss, axis=0)    #sum across all rows, examples
    # loss = np.sum(loss, axis=0)     #sum across all cols, pixel values
    # loss = loss / xtrain.shape[0]
    train_p = 0
    val_p = 0
    train_loss = 0
    val_loss = 0
    return train_p, val_p, train_loss, val_loss
