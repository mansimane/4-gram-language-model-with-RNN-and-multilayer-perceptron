hyper_para = {}
hyper_para['non_lin'] = '_lin_'   # Size of embedding vector
hyper_para['epochs'] = 100

hyper_para['batch_size'] = 512

hyper_para['hidden_layer_size'] = 128   # 100 hidden units

hyper_para['decay'] = 0.00005      # weight decay

hyper_para['embed_size'] = 16   # Size of embedding vector

hyper_para['vocab_size'] = 8000   # size of vocabulary

hyper_para['no_of_grams'] = 4

hyper_para['learning_rate'] = 0.02
hyper_para['context_size'] = hyper_para['no_of_grams'] - 1

hyper_para['w_init_mu'] = 0
hyper_para['w_init_sig'] = 0.01 # mean and standard deviation

hyper_para['c_init_mu'] = 0
hyper_para['c_init_sig'] = 0.015 # mean and standard deviation

# train and test files
hyper_para['proc_train_file_name'] = '../data/train_with_tags.txt'
hyper_para['proc_val_file_name'] = '../data/val_with_tags.txt'
# hyper_para['proc_train_file_name'] = '../data/train_small.txt'
# hyper_para['proc_val_file_name'] = '../data/val_small.txt'

#Not a hyperpara, but just to make evident that it came from here
hyper_para['total_ngrams_in_tr_data'] = 86402
hyper_para['total_ngrams_in_val_data'] = 10360

#starts at 6 for lr  = 0.1, bs 512, winit = xavier, cinit = 0.015
