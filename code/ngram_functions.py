import numpy as np
from generic_functions import *
from ngram_functions import *


def initialize_weights(hyper_para):
    '''
    Initializes all weights

    :param hyper_para:
    :return:
    '''
    mu = hyper_para['w_init_mu']
    sigma = hyper_para['w_init_sig']
    embed_size = hyper_para['embed_size']
    vocab_size = hyper_para['vocab_size']

    #Lookup table 8000x16, dictionary
    we_lookup = load_obj('vocab')
    for key in we_lookup:
        we_lookup[key] = np.random.normal(mu, sigma, (1,embed_size))


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

def get_word_vec(context,param,context_size):
    we_lookup, w1, w2, b1, b2 = param
    for i in range(0, context_size):
        if i is 0:
            x = we_lookup[context[i]]
        else:
            x = np.append(x, we_lookup[context[i]])
    return x

def act_forward(input, w, b):
    """ Args:
        Returns:
    """
    a = input.dot(w) + b
    return a


def act_back(input, output, grad_prev, w, b):
    """ Args:
        output: a numpy array contains output data
        input: a numpy array contains input data layer, defined in testLeNet.m
        param: parameters, a cell array

        Returns:
        para_grad: a cell array stores gradients of parameters
        input_od: gradients w.r.t input data
    """
    w_grad = (input.T).dot(grad_prev)
    b_grad = np.sum(grad_prev, axis=0, keepdims=True)
    grad_next = grad_prev.dot(w.T)

    return w_grad, b_grad, grad_next

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

def grad_calc (param, x, hyper_para):
    '''

    :param param:
    :param x: Embedding vecotrs for 3 context words, 1x48
    :param hyper_para:
    :return:
    '''
    we_lookup, w1, w2, b1, b2 = param

    # #### Forward pass
    #
    # a1 = act_forward(x, w1, b1) #nx100 = nx784 * 784*100
    #
    # h1 = sigmoid_forward(a1) #n x 100 #same as before,
    #
    # a2 = act_forward(h1, w2, b2)    #nx100 = nx100 * 100x100
    #
    # x_hat = h2 = sigmoid_forward(a2) #n x 100 #same as before,
    #
    # ###### Backward Pass
    # #act_back(input, output, grad_prev, w, b)
    # #sigmoid_back (input, output, grad_prev ):
    #
    # d4 = x_hat - x  # nx10
    #
    # d3 = sigmoid_back(a2, h2, d4)
    #
    # [w2_grad, b2_grad, d2] = act_back(h1, a2, d3, w2, b2)
    #
    # d1 = sigmoid_back(a1, h1, d2)
    #
    # [w1_grad, b1_grad, d0] = act_back(x, a1, d1, w1, b1)    #784*100 = nx784' * nx100
    #
    param_grad = {}
    # param_grad['w1'] = w1_grad
    # param_grad['w2'] = w2_grad
    # param_grad['b1'] = b1_grad
    # param_grad['b2'] = b2_grad

    return param_grad

def loss_calc(param, xtrain, ytrain, hyper_para):

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
