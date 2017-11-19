from generic_functions import *
from config_3_2 import *
from ngram_functions import *

vocab = load_obj('vocab')
context_size = hyper_para['context_size']
embed_size = hyper_para['embed_size']
vocab_size = hyper_para['vocab_size']

def process (proc_file_name, p,total_ngrams_in_data, hyper_para):
    '''
        p: Params
    '''


    x_train = np.zeros((total_ngrams_in_data, context_size), dtype=np.int16)
    y_train = np.zeros((total_ngrams_in_data, hyper_para['vocab_size']), dtype=np.int16)
    x = np.zeros((1, context_size))
    vocab_dict = p['vocab_dict']
    line_no = 0

    with open(proc_file_name) as fd_in:  # No need of closing it, as with takes care of it
        ngram_cnt = 0
        for line in fd_in.readlines():
            words = line.split()
            for i in range(len(words) - 3):
                ngram_list = words[i: i + hyper_para['no_of_grams']]

                #Avoinding for loop here  #will work only for 3grams
                x[0, 0] = vocab_dict[ngram_list[0]]
                x[0, 1] = vocab_dict[ngram_list[1]]
                x[0, 2] = vocab_dict[ngram_list[2]]
                idxy = vocab_dict[ngram_list[3]]

                #x_train = np.append(x_train, x, axis=0)
                x_train[ngram_cnt ,:] = x

                # Create output vector
                y = np.zeros((1, vocab_size))
                y[0, idxy] = 1
                #y_train = np.append(y_train, y, axis=0)
                y_train[ngram_cnt,:] = y
                #y_train[ngram_cnt] = idxy
                ngram_cnt += 1
            line_no += 1
            #print line_no
    return x_train, y_train

def prepare_text_data(param):
    total_ngrams_in_tr_data = hyper_para['total_ngrams_in_tr_data']
    proc_train_file_name = hyper_para['proc_train_file_name']
    x_train, y_train = process(proc_train_file_name, param,total_ngrams_in_tr_data, hyper_para)
    #save_obj(x_train, 'x_train')
    np.save('obj/x_train',x_train )
    np.save('obj/vocab_dict',param['vocab_dict'] )
    np.save('obj/vocab_dict_inv',param['vocab_dict_inv'])
    # #pickel can't sav large objs
    np.save('obj/y_train', y_train) #y = np.load('obj/y_train.npy')


    total_ngrams_in_val_data = hyper_para['total_ngrams_in_val_data']
    proc_val_file_name = hyper_para['proc_val_file_name']
    x_val, y_val = process(proc_val_file_name, param, total_ngrams_in_val_data, hyper_para)
    np.save('obj/x_val',x_val )
    np.save('obj/y_val',y_val )
    # save_obj(x_val, 'x_val')
    # save_obj(x_val, 'x_val')

    return x_train, y_train, x_val, y_val

def process_ngram (proc_file_name, p,total_ngrams_in_data, hyper_para):
    '''
        p: Params
    '''


    x_train = np.zeros((total_ngrams_in_data, context_size),dtype=np.int16)
    y_train = np.zeros((total_ngrams_in_data),dtype=np.int16)
    x = np.zeros((1, context_size))
    vocab_dict = p['vocab_dict']
    line_no = 0

    with open(proc_file_name) as fd_in:  # No need of closing it, as with takes care of it
        ngram_cnt = 0
        for line in fd_in.readlines():
            words = line.split()
            for i in range(len(words) - 3):
                ngram_list = words[i: i + hyper_para['no_of_grams']]

                #Avoinding for loop here  #will work only for 3grams
                x[0, 0] = vocab_dict[ngram_list[0]]
                x[0, 1] = vocab_dict[ngram_list[1]]
                x[0, 2] = vocab_dict[ngram_list[2]]
                idxy = vocab_dict[ngram_list[3]]

                #x_train = np.append(x_train, x, axis=0)
                x_train[ngram_cnt ,:] = x

                # Create output vector

                y_train[ngram_cnt] = idxy
                ngram_cnt += 1
            line_no += 1
            #print line_no
    return x_train, y_train

def prepare_text_data_ngram(param):
    total_ngrams_in_tr_data = hyper_para['total_ngrams_in_tr_data']
    proc_train_file_name = hyper_para['proc_train_file_name']
    x_train, y_train = process_ngram(proc_train_file_name, param,total_ngrams_in_tr_data, hyper_para)
    #save_obj(x_train, 'x_train')
    #np.save('obj/x_train',x_train )
    # #pickel can't sav large objs
    np.save('obj/y_train_small', y_train) #y = np.load('obj/y_train.npy')


    total_ngrams_in_val_data = hyper_para['total_ngrams_in_val_data']
    proc_val_file_name = hyper_para['proc_val_file_name']
    x_val, y_val = process_ngram(proc_val_file_name, param, total_ngrams_in_val_data, hyper_para)
    # np.save('obj/x_val',x_val )
    # np.save('obj/y_val',y_val )
    # save_obj(x_val, 'x_val')
    # save_obj(x_val, 'x_val')

    return x_train, y_train, x_val, y_val



def process_trunc (proc_file_name, p,total_ngrams_in_data, hyper_para):
    '''
        p: Params
    '''

    context_size = 2
    x_train = np.zeros((total_ngrams_in_data, context_size), dtype=np.int16)
    y_train = np.zeros((total_ngrams_in_data, hyper_para['vocab_size']), dtype=np.int16)
    x = np.zeros((1, context_size))
    vocab_dict = p['vocab_dict']
    line_no = 0

    with open(proc_file_name) as fd_in:  # No need of closing it, as with takes care of it
        ngram_cnt = 0
        for line in fd_in.readlines():
            words = line.split()
            if ngram_cnt >= total_ngrams_in_data-10:
                break
            for i in range(len(words) - 2):
                ngram_list = words[i: i + hyper_para['no_of_grams']]

                #Avoinding for loop here  #will work only for 3grams
                x[0, 0] = vocab_dict[ngram_list[0]]
                x[0, 1] = vocab_dict[ngram_list[1]]
                # x[0, 2] = vocab_dict[ngram_list[2]]
                idxy = vocab_dict[ngram_list[2]]

                #x_train = np.append(x_train, x, axis=0)
                x_train[ngram_cnt ,:] = x

                # Create output vector
                y = np.zeros((1, vocab_size))
                y[0, idxy] = 1
                #y_train = np.append(y_train, y, axis=0)
                y_train[ngram_cnt,:] = y
                #y_train[ngram_cnt] = idxy
                ngram_cnt += 1
                if ngram_cnt >= total_ngrams_in_data:
                    break
            line_no += 1
            #print line_no
    return x_train, y_train

def prepare_text_data_trunc(param):
    total_ngrams_in_tr_data = hyper_para['total_ngrams_in_tr_data']
    proc_train_file_name = hyper_para['proc_train_file_name']
    x_train, y_train = process_trunc(proc_train_file_name, param,total_ngrams_in_tr_data, hyper_para)
    #save_obj(x_train, 'x_train')
    np.save('obj/x_train_t',x_train )
    # #pickel can't sav large objs
    np.save('obj/y_train_t', y_train) #y = np.load('obj/y_train.npy')


    total_ngrams_in_val_data = hyper_para['total_ngrams_in_val_data']
    proc_val_file_name = hyper_para['proc_val_file_name']
    x_val, y_val = process_trunc(proc_val_file_name, param, total_ngrams_in_val_data, hyper_para)
    np.save('obj/x_val_t',x_val )
    np.save('obj/y_val_t',y_val )
    # save_obj(x_val, 'x_val')
    # save_obj(x_val, 'x_val')

    return x_train, y_train, x_val, y_val
