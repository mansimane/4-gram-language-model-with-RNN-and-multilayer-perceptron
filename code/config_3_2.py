hyper_para = {}
hyper_para['epochs'] = 3

hyper_para['batch_size'] = 256

hyper_para['hidden_layer_size'] = 128   # 100 hidden units

hyper_para['drop_out'] = 0.0      # weight decay

hyper_para['embed_size'] = 16   # Size of embedding vector

hyper_para['vocab_size'] = 8000   # size of vocabulary

hyper_para['no_of_grams'] = 4

hyper_para['learning_rate'] = 0.01
hyper_para['context_size'] = hyper_para['no_of_grams'] - 1

hyper_para['w_init_mu'] = 0
hyper_para['w_init_sig'] = 0.1 # mean and standard deviation

# train and test files
proc_train_file_name = '../data/train_with_tags.txt'
hyper_para['proc_test_file_name'] = '../data/val_with_tags.txt'

total_ngrams_in_tr_data = 86402
hyper_para['total_ngrams_in_val_data'] = 10360