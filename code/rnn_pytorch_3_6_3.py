import numpy as np
import torch
import torch.nn as nn
import torch.nn as nn
from torch.autograd import Variable
import random

batch_size = 16
n_hidden = 128
xtrain = np.load('obj/x_train.npy')
ytrain = np.load('obj/y_train_small.npy')
xval = np.load('obj/x_val.npy')
yval = np.load('obj/y_val.npy')
vocab_dict = np.load('obj/vocab_dict.npy')#word:int
vocab_dict_inv = np.load('obj/vocab_dict_inv.npy')#int:word
vocab_dict = vocab_dict.item()
vocab_dict_inv = vocab_dict_inv.item()
n_letters = 8000
n_categories = 8000
no_of_train_examples = 8640

xtrain = xtrain[0:no_of_train_examples,:]
ytrain = ytrain[0:no_of_train_examples]
print xtrain.shape
print ytrain.shape

def lineToTensor(line):
    batch_size = line.shape[0]
    tensor = torch.zeros(line.shape[1], batch_size, n_letters)
    for i in range (batch_size):
        one_line = line[i,:]
        for li, letter in enumerate(one_line):
            tensor[li][i][one_line[li]] = 1


    return tensor
p = lineToTensor(xtrain[1:9,:])
print p.shape


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):

        combined = torch.cat((input, hidden), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(batch_size, self.hidden_size))

rnn = RNN(n_letters, n_hidden, n_categories)

criterion = nn.NLLLoss()
learning_rate = 0.0005

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
    category_i = top_i[0][0]
    print category_i
    return vocab_dict_inv[category_i], category_i

def randomTrainingExample():
    idx = np.random.randint(no_of_train_examples, size=batch_size)

    category_tensor = Variable(torch.LongTensor((ytrain[idx].astype(int))))

    line_tensor = Variable(lineToTensor(xtrain[idx,:]))

    return category_tensor, line_tensor

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]

import time
import math

n_iters = 10000
print_every = 100
plot_every = 1000



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        print('%d %d%% (%s) %.4f  / %s ' % (iter, iter / n_iters * 100, timeSince(start), loss, guess))


    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

#
# category_tensor, line_tensor = randomTrainingExample()
# output, loss = train(category_tensor, line_tensor)
# current_loss += loss
# print current_loss