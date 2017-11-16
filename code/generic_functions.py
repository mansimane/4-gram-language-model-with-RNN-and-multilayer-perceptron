import re
import pickle
import numpy as np
from sklearn.neighbors import *
from sklearn.neighbors import KDTree


def tokenizeDoc(cur_doc):
    return re.findall('\\w+',cur_doc)

def save_obj(obj, name, epoch =-1):
    if epoch is -1:
        epoch = ''
    with open('obj/'+ name + epoch +'.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,epoch =-1):
    if epoch is -1:
        epoch = ''
    with open('obj/' + name + epoch + '.pkl', 'rb') as f:
        return pickle.load(f)

def softmax_forward(a):

    if len(a.shape) == 1:   #if vector
        y = np.exp(a)
        y = y/np.sum(y, keepdims=True)
        assert (len(y) == len(a)), 'Y returned from softmax not match with input array'

    #a = samples x num_of_labels
    else:
        y = np.exp(a)
        y = y / np.sum(y, axis=1, keepdims=True)
    return y

def softmax_back (y_pred, y_correct):
    output = y_pred - y_correct
    return output

def act_forward(input, w, b):
    """ Args:
        Returns:
    """
    a = input.dot(w) + b
    return a

def tanh_forward(z):
    return np.tanh(z)

def tanh_back(input, output, grad_prev):
    grad_next = grad_prev * (1 - np.power(output, 2))
    return grad_next

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

def find_10_nearest_words(words_ch, param, hyper_para):
    #Chosen words
    we_lookup = param['we_lookup']
    vocab_map = param['vocab_dict']
    vocab_map = param['vocab_dict']
    vocab_dict_inv = param['vocab_dict_inv']
    no_of_nbrs = 10
    idx_lst = []
    for word in words_ch.keys():
        idx_lst.append(vocab_map[word])


    kdt = KDTree(we_lookup, leaf_size=30, metric='euclidean')
    query = we_lookup[idx_lst,:]
    nbrs = kdt.query(query, k=no_of_nbrs, return_distance=False)


    for i in range(nbrs.shape[0]):
        word_lst = []
        for j in range(no_of_nbrs):
            word_lst.append(vocab_dict_inv[nbrs[i][j]])
        word = vocab_dict_inv[idx_lst[i]]
        words_ch[word] = word_lst
    return words_ch
