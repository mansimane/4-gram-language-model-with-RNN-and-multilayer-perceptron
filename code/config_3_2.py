hyper_para = {}
hyper_para['epochs'] = 100

hyper_para['batch_size'] = 100

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
prec_test_file_name = '../data/val_with_tags.txt'