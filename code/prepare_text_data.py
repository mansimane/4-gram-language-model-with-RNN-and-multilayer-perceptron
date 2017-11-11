from generic_functions import *
from config_3_2 import *
from ngram_functions import *

vocab = load_obj('vocab')
context_size = hyper_para['context_size']
embed_size = hyper_para['embed_size']
vocab_size = hyper_para['vocab_size']

def process (proc_file_name, p):
    '''
        p: Params
    '''

    x_train = np.zeros((0, context_size))
    y_train = np.zeros((0, vocab_size))
    x = np.zeros((1, context_size))
    vocab_dict = p['vocab_dict']
    we_lookup = p['we_lookup']
    line_no = 0
    with open(proc_file_name) as fd_in:  # No need of closing it, as with takes care of it
        for line in fd_in.readlines():
            words = line.split()
            for i in range(len(words) - 3):
                ngram_list = words[i: i + hyper_para['no_of_grams']]

                #Avoinding for loop here  #will work only for 3grams
                x[0, 0] = vocab_dict[ngram_list[0]]
                x[0, 1] = vocab_dict[ngram_list[1]]
                x[0, 2] = vocab_dict[ngram_list[2]]
                idxy = vocab_dict[ngram_list[3]]

                x_train = np.append(x_train, x, axis=0)

                # Create output vector
                y = np.zeros((1, vocab_size))
                y[0, idxy] = 1
                y_train = np.append(y_train, y, axis=0)
            line_no += 1
            print line_no
    return x_train, y_train

def prepare_text_data(param):

    proc_train_file_name = hyper_para['proc_train_file_name']
    x_train, y_train = process(proc_train_file_name, param)

    proc_val_file_name = hyper_para['proc_val_file_name']
    x_val, y_val = process(proc_val_file_name, param)
    return x_train, y_train, x_val, y_val




