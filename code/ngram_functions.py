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
    vocab_dict_inv  = {y:x for x,y in vocab_dict.iteritems()}
    we_lookup = np.random.normal(mu_c, sig_c, (vocab_size*embed_size))
    we_lookup = we_lookup.reshape(vocab_size,embed_size)

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

    param = {}
    param['we_lookup'] = we_lookup
    param['vocab_dict'] = vocab_dict
    param['vocab_dict_inv'] = vocab_dict_inv
    param['w1'] = w1
    param['w2'] = w2
    param['b1'] = b1
    param['b2'] = b2
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
    #print 'get_word_vec.line_no', get_word_vec.line_no

    x_batch = np.zeros((0, embed_size* context_size))
    y_batch = np.zeros((0, vocab_size))
    ngram_list_batch = []
    while (x_batch.shape[0] != hyper_para['batch_size']):
        line = train_data[get_word_vec.line_no]
        words = line.split()
        # Loop over lines
        for i in range(get_word_vec.line_idx, len(words)-3):
            ngram_list = words[i: i+ hyper_para['no_of_grams']]
            ngram_list_batch.append(ngram_list)
            #Create vector from ngrams
            x = np.zeros((0, embed_size))
            for j in range(0, context_size):
                x = np.append(x, we_lookup[ngram_list[j]][1])
            x = np.reshape(x, (1, embed_size * context_size))
            x_batch = np.append(x_batch, x, axis=0 )

            #Create output vector
            y = np.zeros((1, vocab_size))
            y[0, we_lookup[ngram_list[context_size]][0]] = 1
            y_batch = np.append(y_batch, y, axis=0)
            get_word_vec.line_idx += 1
            if x_batch.shape[0] == hyper_para['batch_size']:
                return ngram_list_batch, x_batch, y_batch

        get_word_vec.line_idx = 0
        get_word_vec.line_no += 1

        #End of file, batch size can be smaller than programmed on for last one
        if get_word_vec.line_no == get_word_vec.max_line_no:
            get_word_vec.line_idx = 0
            get_word_vec.line_no = 0
            return ngram_list_batch, x_batch, y_batch

    return ngram_list_batch, x_batch, y_batch

def get_word_vec_val(val_data, hyper_para, param):
    we_lookup, w1, w2, b1, b2 = param
    vocab_size = hyper_para['vocab_size']
    context_size = hyper_para['context_size']
    embed_size = hyper_para['embed_size']

    if not hasattr(get_word_vec_val, "line_idx"):
        get_word_vec_val.line_idx = 0
    if not hasattr(get_word_vec_val, "line_no"):
        get_word_vec_val.line_no = 0
    if not hasattr(get_word_vec_val, "max_line_no"):
        get_word_vec_val.max_line_no = len(val_data)
    #print 'get_word_vec.line_no', get_word_vec.line_no

    x_batch = np.zeros((0, embed_size* context_size))
    y_batch = np.zeros((0, vocab_size))
    ngram_list_batch = []
    while (x_batch.shape[0] != hyper_para['batch_size']):
        line = val_data[get_word_vec_val.line_no]
        words = line.split()
        # Loop over lines
        for i in range(get_word_vec_val.line_idx, len(words)-3):
            ngram_list = words[i: i+ hyper_para['no_of_grams']]
            ngram_list_batch.append(ngram_list)
            #Create vector from ngrams
            x = np.zeros((0, embed_size))
            for j in range(0, context_size):
                x = np.append(x, we_lookup[ngram_list[j]][1])
            x = np.reshape(x, (1, embed_size * context_size))
            x_batch = np.append(x_batch, x, axis=0 )

            #Create output vector
            y = np.zeros((1, vocab_size))
            y[0, we_lookup[ngram_list[context_size]][0]] = 1
            y_batch = np.append(y_batch, y, axis=0)
            get_word_vec_val.line_idx += 1
            if x_batch.shape[0] == hyper_para['batch_size']:
                return ngram_list_batch, x_batch, y_batch

        get_word_vec_val.line_idx = 0
        get_word_vec_val.line_no += 1

        #End of file, batch size can be smaller than programmed on for last one
        if get_word_vec_val.line_no == get_word_vec_val.max_line_no:
            get_word_vec_val.line_idx = 0
            get_word_vec_val.line_no = 0
            return ngram_list_batch, x_batch, y_batch

    return ngram_list_batch, x_batch, y_batch


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
    train_loss = 0
    train_p = 0
    we_lookup = param['we_lookup']
    w1 = param['w1']
    w2 = param['w2']
    b2 = param['b2']
    b1 = param['b1']
    x = np.zeros((no_of_samples, vocab_size * embed_size))

    x[:, 0:16] = we_lookup[data_x[:, 0]]
    x[:, 16:32] = we_lookup[data_x[:, 1]]
    x[:, 32:48] = we_lookup[data_x[:, 2]]

    # #### Forward pass
    a1 = act_forward(x, w1, b1) #nx128 = nx48 * 48*128

    a2 = act_forward(a1, w2, b2)    #nx100 = nx128 * 128x8000

    y_pred = softmax_forward(a2)




    #Validation DATA loss calculation
    proc_test_file_name = hyper_para['proc_test_file_name']
    total_ngrams_in_val_data = hyper_para['total_ngrams_in_val_data']
    with open(proc_test_file_name) as fd_val:
        val_data = fd_val.readlines()
    no_of_ngram_read = 0
    val_loss = 0
    val_p = 0
    while (no_of_ngram_read <= total_ngrams_in_val_data):
        ngram_list, x, y = get_word_vec_val(val_data, hyper_para, param)
        no_of_samples = x.shape[0]
        #Forward pass
        a1 = act_forward(x, w1, b1)
        a2 = act_forward(a1, w2, b2)
        y_pred = softmax_forward(a2)
        y_corr_idx = np.zeros((y_pred.shape[0]))

        for i in range(y_pred.shape[0]):
            cor_word = ngram_list[i][context_size]
            y_corr_idx[i] = we_lookup[cor_word][0]
        #**You should include other classes as well
        y_corr_idx = y_corr_idx.astype(int)
        y_prob_right = y_pred[range(no_of_samples), y_corr_idx]
        y_pred[range(no_of_samples), y_corr_idx] = 0
        y_prob_wrng = -np.log(1 - y_pred)   #the right ones have zero so log(1-0)= 0 contribution in error

        y_prob_right = -np.log(y_prob_right)
        val_loss += (np.sum(y_prob_right)+ np.sum(np.sum(y_prob_wrng)))/len(y_prob_right)

        val_p += np.power(2.0, val_loss) #*** 2.7 for natural log
        no_of_ngram_read += x.shape[0]

    return train_p, val_p, train_loss, val_loss


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

    #sum and divide gradient
    w2 = w2 - (lr * (w2_grad/batch_size))
    w1 = w1 - (lr * (w1_grad/batch_size))
    b2 = b2 - (lr * (b2_grad/batch_size))
    b1 = b1 - (lr * (b1_grad/batch_size))

    #Use vectorized implementation
    for i in range(we_grad.shape[0]):
            id0 = x_train[i,0]
            id1 = x_train[i,1]
            id2 = x_train[i,2]
            we_lookup[id0,:] = we_lookup[0,:] - lr* we_grad[0,:]
            we_lookup[id1,:] = we_lookup[1,:] - lr* we_grad[1,:]
            we_lookup[id2,:] = we_lookup[2,:] - lr* we_grad[2,:]

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
    x = np.zeros((batch_size, vocab_size*embed_size))

    x[:, 0:16] = we_lookup[x_train[:, 0]]
    x[:, 16:32] = we_lookup[x_train[:, 1]]
    x[:, 32:48] = we_lookup[x_train[:, 2]]

    # #### Forward pass
    a1 = act_forward(x, w1, b1) #nx128 = nx48 * 48*128

    a2 = act_forward(a1, w2, b2)    #nx100 = nx128 * 128x8000

    y_pred = softmax_forward(a2)

    ####### Backward Pass
    d3 = softmax_back(y_pred, y_train)  # nx8000

    [w2_grad, b2_grad, d2] = act_back(a1, a2, d3, w2, b2)

    [w1_grad, b1_grad, d1] = act_back(x, a1, d2, w1, b1)

    we_grad = d1
    param_grad = (we_grad, w1_grad, w2_grad, b1_grad, b2_grad)

    return param_grad

