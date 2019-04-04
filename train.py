import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import math
from data import *
from models import *
from nltk.translate.bleu_score import corpus_bleu
import random


MAX_LENGTH = 10


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence if word in lang.word2index]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, model, optimizer, criterion):
    encoder_hidden = model.init_hidden()
    encoder_cell_state = model.init_hidden()

    # input_tensor = input_tensor.flip([0])

    optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    loss = 0
    # print("Input", input_tensor)
    # print("Target", target_tensor)

    for ei in range(input_length):
        encoder_output, encoder_hidden, encoder_cell_state = model.encode(input_tensor[ei], (encoder_hidden, encoder_cell_state))

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    decoder_cell_state = model.init_hidden()

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_cell_state = model.decode(decoder_input, (decoder_hidden, decoder_cell_state))
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]

    loss.backward()
    optimizer.step()
    # print('done')

    return loss.item() / target_length

def validate(input_tensor, target_tensor, model, criterion, max_length=MAX_LENGTH):
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    loss = 0
    encoder_hidden = model.init_hidden()
    encoder_cell_state = model.init_hidden()

    # input_tensor = input_tensor.flip([0])

    for ei in range(input_length):
        encoder_output, encoder_hidden, encoder_cell_state = model.encode(input_tensor[ei], (encoder_hidden, encoder_cell_state))

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    decoder_cell_state = model.init_hidden()


    # use k-beam search
    k=10
    predictions = []

    decoder_output, decoder_hidden, decoder_cell_state = model.decode(decoder_input, (decoder_hidden, decoder_cell_state))
    loss += criterion(decoder_output, target_tensor[di])
    topv, topi = decoder_output.data.topk(k)
    decoder_input = topi.squeeze().detach()

    for di in range(target_length):
        temp = []
        for i in range(k):
        decoder_output, decoder_hidden, decoder_cell_state = model.decode(decoder_input, (decoder_hidden, decoder_cell_state))
        loss += criterion(decoder_output, target_tensor[di])
        topv, topi = decoder_output.data.topk(k)
        decoder_input = topi.squeeze().detach()

    return loss.item() / target_length

def translate(input_tensor, model, max_length=MAX_LENGTH):
    input_length = input_tensor.size(0)
    target_length = int(input_length * 1.25)
    encoder_hidden = model.init_hidden()
    encoder_cell_state = model.init_hidden()

    # input_tensor = input_tensor.flip([0])

    for ei in range(input_length):
        encoder_output, encoder_hidden, encoder_cell_state = model.encode(input_tensor[ei], (encoder_hidden, encoder_cell_state))

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    decoder_cell_state = model.init_hidden()
    decoder_words = []

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_cell_state = model.decode(decoder_input, (decoder_hidden, decoder_cell_state))
        topv, topi = decoder_output.data.topk(1)
        if topi.item() == EOS_token:
            break
        else:
            decoder_words.append(output_lang.index2word[topi.item()])
        decoder_input = topi.squeeze().detach()
    return decoder_words


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# data = {"in":input_lang, "out": output_lang, "in_e":input_embd,
# "out_e": output_embd, 'p': pairs, 'tp': training_pairs}

import pickle
f = open('preprocess_en_de.pickle', 'rb')
data = pickle.load(f)
f.close()
input_lang =  data['in']
output_lang = data['out']
pairs = data['p']
input_embd = data['in_e']
output_embd = data['out_e']
training_pairs = data['tp']

# # datafile = 'data/eng_ger_train.txt'
# datafile = 'data/valdata.enhi.txt'
# print('Data File:', datafile)
# input_lang, output_lang, pairs = prepare_data(datafile, 'eng', 'hindi')
# input_embd = input_lang.embeddings('data/eng.vec')
# output_embd = output_lang.embeddings('data/hindi.vec')
# print("Pretrained embedding loaded")
# training_pairs = [tensorsFromPair(pairs[i]) for i in range(len(pairs))]
# # training_pairs = training_pairs[0:1000]

print('Tranining Pairs:', len(training_pairs))

val_data_file = 'data/eng_ger_valid.txt'
print('Validation File:', val_data_file)
val_pairs = read_test_file(val_data_file)
validation_pairs =[tensorsFromPair(val_pairs[i]) for i in range(len(val_pairs))]
# validation_pairs = validation_pairs[0:1000]
print('Validation Pairs:', len(validation_pairs))

hidden_size = 256

model = Seq2Seq(input_lang.n_words, hidden_size, output_lang.n_words, input_embd, output_embd, device).to(device)

epochs = 4

learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()

print_loss_total = 0
n_iters = len(training_pairs)

for ep in range(epochs):
    start = time.time()
    for iter in range(0, n_iters):
        training_pair = training_pairs[iter]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        # print(pairs[iter])
        # print("input", input_tensor)
        # print("output", target_tensor)

        loss = train(input_tensor, target_tensor, model, optimizer, criterion)
        print_loss_total += loss

        if (iter+1) % 500 == 0:
            print_loss_avg = print_loss_total / 500
            print_loss_total = 0
            # print('%s (%d %d%%) %.4f' % (timeSince(start, (iter+1) / n_iters), (iter+1), (iter+1)/n_iters * 100, print_loss_avg))
            print("Epoch: {}, Step: {}, Training Loss: {}, Time Left: {}".format(ep, iter, print_loss_avg, timeSince(start, (iter+1) / n_iters)))

        if (iter+1) % 20000 == 0:
            with torch.no_grad():
                vloss = 0
                actual = []
                predicted = []
                j = 0
                for i in range(len(validation_pairs)):
                    val_in = validation_pairs[i][0]
                    val_out = validation_pairs[i][1]
                    vloss += validate(val_in, val_out, model, criterion)
                    actual.append(val_pairs[i][1])
                    predicted.append(translate(val_in, model))
                    if j < 2 and random.random() > 0.75:
                        print('S:', val_pairs[i][0])
                        print('O:', val_pairs[i][1])
                        print('P:', translate(val_in, model))
                        j += 1
                print("Epoch: {}, validation Loss: {} ".format(ep, vloss/len(validation_pairs)))
                # calculate BLEU score
                print('BLEU-1:', corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))

    torch.save(model, 'model_eh_'+str(ep))

# translate
