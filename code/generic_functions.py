import re
import pickle
import numpy as np

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
