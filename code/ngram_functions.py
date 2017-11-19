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
    mu_c = hyper_para['c_init_mu']
    sig_c = hyper_para['c_init_sig']

    #Lookup table 8000x16, dictionary
    vocab = load_obj('vocab')
    vocab_dict = {}
    i=0
    for key in vocab:
        vocab_dict[key] = i
        i = i+1
#    vocab_dict_inv  = {y:x for x,y in vocab_dict.iteritems()}
    we_lookup = np.random.normal(mu_c, sig_c, (vocab_size*embed_size))
    we_lookup = we_lookup.reshape(vocab_size,embed_size)

    #Weights for activation layer 1, 48x128
    input_layer_size = hyper_para['context_size']*hyper_para['embed_size']
    hidden_layer_size = hyper_para['hidden_layer_size']

    # w1 = np.random.normal(mu, sigma, (input_layer_size * hidden_layer_size))
    # w1 = w1.reshape(input_layer_size, hidden_layer_size)
    # b1 = np.random.normal(mu, sigma, (1, hidden_layer_size))
    #
    # #Weights for activation layer 2, 128x8000
    # w2 = np.random.normal(mu, sigma, (hidden_layer_size * vocab_size))
    # w2 = w2.reshape(hidden_layer_size, vocab_size)
    # b2 = np.random.normal(mu, sigma, (1, vocab_size))

    #implementing Xavier weight initialization
    w1 = np.random.normal(mu, (2.0/(input_layer_size + hidden_layer_size)), (input_layer_size * hidden_layer_size))
    w1 = w1.reshape(input_layer_size, hidden_layer_size)
    b1 = np.random.normal(mu, (2.0/hidden_layer_size), (1, hidden_layer_size))

    #Weights for activation layer 2, 128x8000
    w2 = np.random.normal(mu, (2.0/ (hidden_layer_size + vocab_size)), (hidden_layer_size * vocab_size))
    w2 = w2.reshape(hidden_layer_size, vocab_size)
    b2 = np.random.normal(mu, (2.0/vocab_size), (1, vocab_size))

    param = {}
    param['we_lookup'] = we_lookup
    param['vocab_dict'] = vocab_dict
#    param['vocab_dict_inv'] = vocab_dict_inv
    param['w1'] = w1
    param['w2'] = w2
    param['b1'] = b1
    param['b2'] = b2
    return param

def loss_calc(param, hyper_para, data_x, data_y):
    we_lookup = param['we_lookup']
    w1 = param['w1']
    w2 = param['w2']
    b2 = param['b2']
    b1 = param['b1']

    context_size = hyper_para['context_size']
    embed_size = hyper_para['embed_size']
    vocab_size = hyper_para['vocab_size']
    no_of_samples = data_x.shape[0]

    x = np.zeros((no_of_samples, context_size * embed_size))

    x[:, 0:embed_size] = we_lookup[data_x[:, 0].astype(int)]
    x[:, embed_size:embed_size*2] = we_lookup[data_x[:, 1].astype(int)]
    x[:, embed_size*2:embed_size*3] = we_lookup[data_x[:, 2].astype(int)]

    # #### Forward pass
    a1 = act_forward(x, w1, b1) #nx128 = nx48 * 48*128

    a2 = act_forward(a1, w2, b2)    #nx100 = nx128 * 128x8000

    y_pred = softmax_forward(a2)

#    y_pred_idx = y_pred.argmax(axis=1)

#    per_err = len(np.where(y_pred_idx !=data_y)[0])/float(len(data_y))

    #Forward pass
    prod = y_pred[np.arange(len(y_pred)), data_y.astype(int)]
    loss_arr = -np.log(prod)
    loss = np.sum(loss_arr)/loss_arr.shape[0]

    per = np.power(2.7, loss) #*** 2.7 for natural log
    return per, loss


def update_param (param, param_grad, x_train, hyper_parameters):
    """Update the parameters with sgd with momentum
  Args:
        param: tuple
        param_grad: tuple
        hyper_parameters: dict
  Returns:
    """
    we_grad, w1_grad, w2_grad, b1_grad, b2_grad = param_grad

    we_lookup = param['we_lookup']
    w1 = param['w1']
    w2 = param['w2']
    b2 = param['b2']
    b1 = param['b1']

    batch_size = hyper_parameters['batch_size']
    lr = hyper_parameters['learning_rate']
    embed_size = hyper_parameters['embed_size']
    context_size = hyper_parameters['context_size']
    decay = hyper_parameters['decay']

    #sum and divide gradient
    w2 = w2 - (lr * (w2_grad/batch_size))
    w1 = w1 - (lr * (w1_grad/batch_size))
    b2 = b2 - (lr * (b2_grad/batch_size))
    b1 = b1 - (lr * (b1_grad/batch_size))

    #****Use vectorized implementation
    # for i in range(we_grad.shape[0]):
    #         id0 = x_train[i,0]
    #         id1 = x_train[i,1]
    #         id2 = x_train[i,2]
    #         we_lookup[id0, :] = we_lookup[id0, :] - lr* we_grad[i, 0:16]
    #         we_lookup[id1, :] = we_lookup[id1, :] - lr* we_grad[i, 16:32]
    #         we_lookup[id2, :] = we_lookup[id2, :] - lr* we_grad[i, 32:48]
    id0 = x_train[:, 0]
    id1 = x_train[:, 1]
    id2 = x_train[:, 2]
    id0 = id0.astype(int)
    id1 = id1.astype(int)
    id2 = id2.astype(int)
    we_lookup[id0, :] = we_lookup[id0, :] - lr * we_grad[:, 0:embed_size] - (decay * we_lookup[id0, :])
    we_lookup[id1, :] = we_lookup[id1, :] - lr * we_grad[:, embed_size:embed_size*2] - (decay * we_lookup[id1, :])
    we_lookup[id2, :] = we_lookup[id2, :] - lr * we_grad[:, embed_size*2:embed_size*3] - (decay * we_lookup[id2, :])

    #May be we should divide gradient by no_of_words

    param['we_lookup'] = we_lookup
    param['w1'] = w1
    param['w2'] = w2
    param['b1'] = b1
    param['b2'] = b2
    return param

def grad_calc (param, x_train, y_train, hyper_para):
    '''
    :param param:
    :param x: Embedding vecotrs for 3 context words, 1x48
    :param hyper_para:
    :return:
    '''
    context_size = hyper_para['context_size']
    vocab_size = hyper_para['vocab_size']
    batch_size = hyper_para['batch_size']
    embed_size = hyper_para['embed_size']

    we_lookup = param['we_lookup']
    w1 = param['w1']
    w2 = param['w2']
    b2 = param['b2']
    b1 = param['b1']
    x = np.zeros((batch_size, context_size*embed_size))

    x[:, 0:embed_size] = we_lookup[x_train[:, 0].astype(int)]
    x[:, embed_size:embed_size*2] = we_lookup[x_train[:, 1].astype(int)]
    x[:, embed_size*2:embed_size*3] = we_lookup[x_train[:, 2].astype(int)]

    # #### Forward pass
    a1 = act_forward(x, w1, b1) #nx128 = nx48 * 48*128

    a2 = act_forward(a1, w2, b2)    #nx100 = nx128 * 128x8000

    y_pred = softmax_forward(a2)

    ####### Backward Pass
    #d3 = softmax_back(y_pred, y_train)  # nx8000    #y_pred - y_correct

    #y_pred_cor_cls = y_pred[np.arange(len(y_pred)), y_train.astype(int)]
    prod = y_pred[np.arange(len(y_pred)), y_train.astype(int)]
    loss_arr = -np.log(prod)
    loss = np.sum(loss_arr) / loss_arr.shape[0]

    per = np.power(2.7, loss)  # *** 2.7 for natural log

    y_pred[np.arange(len(y_pred)), y_train.astype(int)] = y_pred[np.arange(len(y_pred)), y_train.astype(int)] - 1

    [w2_grad, b2_grad, d2] = act_back(a1, a2, y_pred, w2, b2)

    [w1_grad, b1_grad, d1] = act_back(x, a1, d2, w1, b1)

    we_grad = d1
    param_grad = (we_grad, w1_grad, w2_grad, b1_grad, b2_grad)

    return param_grad, loss, per

def grad_calc_with_tanh (param, x_train, y_train, hyper_para):
    '''
    :param param:
    :param x: Embedding vecotrs for 3 context words, 1x48
    :param hyper_para:
    :return:
    '''
    context_size = hyper_para['context_size']
    vocab_size = hyper_para['vocab_size']
    batch_size = hyper_para['batch_size']
    embed_size = hyper_para['embed_size']

    we_lookup = param['we_lookup']
    w1 = param['w1']
    w2 = param['w2']
    b2 = param['b2']
    b1 = param['b1']
    x = np.zeros((batch_size, context_size*embed_size))

    x[:, 0:embed_size] = we_lookup[x_train[:, 0].astype(int)]
    x[:, embed_size:embed_size*2] = we_lookup[x_train[:, 1].astype(int)]
    x[:, embed_size*2:embed_size*3] = we_lookup[x_train[:, 2].astype(int)]

    # #### Forward pass
    a1 = act_forward(x, w1, b1) #nx128 = nx48 * 48*128

    h1 = tanh_forward(a1) #nx128

    a2 = act_forward(h1, w2, b2)    #nx100 = nx128 * 128x8000

    y_pred = softmax_forward(a2)
    prod = y_pred[np.arange(len(y_pred)), y_train.astype(int)]
    loss_arr = -np.log(prod)
    loss = np.sum(loss_arr) / loss_arr.shape[0]

    per = np.power(2.7, loss)  # *** 2.7 for natural log
    ####### Backward Pass
    #d3 = softmax_back(y_pred, y_train)  # nx8000    #y_pred - y_correct

    #y_pred_cor_cls = y_pred[np.arange(len(y_pred)), y_train.astype(int)]

    y_pred[np.arange(len(y_pred)), y_train.astype(int)] = y_pred[np.arange(len(y_pred)), y_train.astype(int)] - 1

    [w2_grad, b2_grad, d3] = act_back(a1, a2, y_pred, w2, b2)

    d2 = tanh_back(a1, h1, d3)

    [w1_grad, b1_grad, d1] = act_back(x, a1, d2, w1, b1)

    we_grad = d1
    param_grad = (we_grad, w1_grad, w2_grad, b1_grad, b2_grad)

    return param_grad, loss, per

def predict_word(data_x, param, hyper_para):
    #With or without tanh***

    we_lookup = param['we_lookup']
    w1 = param['w1']
    w2 = param['w2']
    b2 = param['b2']
    b1 = param['b1']
    vocab_dict = param['vocab_dict']
    vocab_dict_inv = param['vocab_dict_inv']

    context_size = hyper_para['context_size']
    embed_size = hyper_para['embed_size']
    vocab_size = hyper_para['vocab_size']

    x = np.zeros((1, context_size * embed_size))

    x[:, 0:embed_size] = we_lookup[data_x[0].astype(int)]
    x[:, embed_size:embed_size * 2] = we_lookup[data_x[1].astype(int)]
    x[:, embed_size * 2:embed_size * 3] = we_lookup[data_x[2].astype(int)]

    # #### Forward pass
    a1 = act_forward(x, w1, b1)  # nx128 = nx48 * 48*128

    h1 = tanh_forward(a1) #nx128

    a2 = act_forward(h1, w2, b2)  # nx100 = nx128 * 128x8000

    y_pred = softmax_forward(a2)

    y_pred_idx = y_pred.argmax(axis=1) #return type is array

    return vocab_dict_inv[y_pred_idx[0]]

def loss_calc_tanh(param, hyper_para, data_x, data_y):
    we_lookup = param['we_lookup']
    w1 = param['w1']
    w2 = param['w2']
    b2 = param['b2']
    b1 = param['b1']

    context_size = hyper_para['context_size']
    embed_size = hyper_para['embed_size']
    vocab_size = hyper_para['vocab_size']
    no_of_samples = data_x.shape[0]

    x = np.zeros((no_of_samples, context_size * embed_size))

    x[:, 0:embed_size] = we_lookup[data_x[:, 0].astype(int)]
    x[:, embed_size:embed_size*2] = we_lookup[data_x[:, 1].astype(int)]
    x[:, embed_size*2:embed_size*3] = we_lookup[data_x[:, 2].astype(int)]

    # #### Forward pass
    a1 = act_forward(x, w1, b1) #nx128 = nx48 * 48*128

    h1 = tanh_forward(a1) #nx128

    a2 = act_forward(h1, w2, b2)    #nx100 = nx128 * 128x8000

    y_pred = softmax_forward(a2)
#    y_pred_idx = y_pred.argmax(axis=1)

#    per_err = len(np.where(y_pred_idx !=data_y)[0])/float(len(data_y))

    #Forward pass
    prod = y_pred[np.arange(len(y_pred)), data_y.astype(int)]
    loss_arr = -np.log(prod)
    loss = np.sum(loss_arr)/loss_arr.shape[0]

    per = np.power(2.7, loss) #*** 2.7 for natural log
    return per, loss

def compute_dist(words, param):
    '''
    Returns nearest 10 words to given word except that word
    :param words:
    :param param:
    :return:
    '''
    we_lookup = param['we_lookup']
    w1 = param['w1']
    w2 = param['w2']
    b2 = param['b2']
    b1 = param['b1']
    vocab_map = param['vocab_dict']

    for word in words:
        idx =  vocab_map[word]
        vec = we_lookup[idx]
        wr = np.linalg.norm(vec - we_lookup)

    clost_lst = []

    return clost_lst