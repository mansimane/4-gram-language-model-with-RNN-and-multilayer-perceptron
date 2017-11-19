from prepare_text_data import *
from config_3_2 import *
from ngram_functions import *
from keras.models import Sequential
from keras.layers import Recurrent, LSTM, Dense, Embedding, SimpleRNN
from keras import optimizers
from keras import backend as K
from generic_functions import *
import matplotlib.pyplot as plt
import time
import numpy as np
import numpy as np



param = initialize_weights(hyper_para)
#x_train, y_train, x_val, y_val = prepare_text_data(param)
y_train = np.load('obj/y_train.npy')
print("a")
y_val = np.load('obj/y_val.npy')
x_val = np.load('obj/x_val.npy')
x_train = np.load('obj/x_train.npy')

print(x_train.shape, y_train.shape)
print(x_train.shape, y_train.shape)

#Build the model

model = Sequential()
no_of_words = 8000
word_embedding_size = hyper_para['embed_size']
RNN_length = 128
batch_size = 16
epochs = hyper_para['epochs']
learning_rate = 0.01

model.add(Embedding(no_of_words, word_embedding_size, input_length=3))
#model.add(LSTM(RNN_length,activation='tanh', return_sequences = False))
#keras.layers.SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
#model.add(SimpleRNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False))
model.add(SimpleRNN(RNN_length, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False))


#model.add(Dropout(0.5))
model.add(Dense(no_of_words, activation='softmax'))


#input_array = np.random.randint(1000, size=(32, 10))
sgd = optimizers.SGD(lr=learning_rate)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy')
history = model.fit(x_train, y_train, validation_data = (x_val, y_val), batch_size = batch_size, epochs = epochs)

#output_array = model.predict(input_array)
d = '_ev_' + str(word_embedding_size)
e = '_ep_' + str(epochs)
a = '_hh_' + str(RNN_length)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('RNN loss: '+ d+ e)
plt.ylabel('Cross entropy loss')
plt.xlabel('Epochs')
date = time.strftime("%Y-%m-%d_%H_%M")
plt.legend(['train', 'test'], loc='upper right')
fig = plt.gcf()
fig.savefig('../results/'+'/Q3_6_1_err'+ a+d + e + '_' + date + '.png')
#plt.show()

tr_per = np.zeros((1,len(history.history['loss'])))
val_per = np.zeros((1,len(history.history['val_loss'])))

for i in range(epochs):
    tr_per[0,i] = np.power(2.7, history.history['loss'][i])
    val_per[0,i] = np.power(2.7, history.history['val_loss'][i])

plt.close()
plt.plot(tr_per[0,:])
plt.plot(val_per[0,:])
plt.title('RNN Perplexity: '+ d+ e)
plt.ylabel('Perplexity')
plt.xlabel('Epochs')
date = time.strftime("%Y-%m-%d_%H_%M")
plt.legend(['train', 'test'], loc='upper right')
fig = plt.gcf()
fig.savefig('../results/'+'/Q3_6_1_per'+a +d + e + '_' + date + '.png')
#plt.show()
np.save('../results/train_loss'+a+ d+e, history.history['loss'])
np.save('../results/val_loss'+a+ d+e, history.history['val_loss'])
