import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, input_embd, output_embd, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.encoder_embedding = nn.Embedding.from_pretrained(input_embd, freeze=True)
        self.encoder_lstm = nn.LSTM(300, hidden_size)#, bidirectional=True)

        self.decoder_embedding = nn.Embedding.from_pretrained(output_embd, freeze=True)
        self.decoder_lstm = nn.LSTM(300, hidden_size)#, bidirectional=True)
        self.time_dep = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def encode(self, input, hidden):
        input_embed = self.encoder_embedding(input).view(1, 1, -1)
        output, (hidden, cell_state) = self.encoder_lstm(input_embed, hidden)
        return output, hidden, cell_state

    def decode(self, input, hidden):
        input_embed = self.decoder_embedding(input).view(1 ,1, -1)
        # output = F.relu(input_embed)
        output, (hidden, cell_state) = self.decoder_lstm(input_embed, hidden)
        output = self.softmax(self.time_dep(output[0]))
        return output, hidden, cell_state

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class Seq2Seq_attn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, input_embd, output_embd, attn_type, device):
        super(Seq2Seq_attn, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.encoder_embedding = nn.Embedding.from_pretrained(input_embd, freeze=True)
        self.encoder_lstm = nn.LSTM(300, hidden_size)

        self.decoder_embedding = nn.Embedding.from_pretrained(output_embd, freeze=True)
        self.decoder_lstm = nn.LSTM(300, hidden_size)
        self.time_dep = nn.Linear(hidden_size*2, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.attn_type = attn_type
        self.multiplicative_W = torch.randn(hidden_size, hidden_size, device=self.device, requires_grad=True)

        self.additive_W1 = torch.randn(200, hidden_size, device=self.device, requires_grad=True)
        self.additive_W2 = torch.randn(200, hidden_size, device=self.device, requires_grad=True)
        self.additive_v = torch.randn(200, device=self.device, requires_grad=True)

    def encode(self, input, hidden):
        input_embed = self.encoder_embedding(input).view(1, 1, -1)
        output, (hidden, cell_state) = self.encoder_lstm(input_embed, hidden)
        return output, hidden, cell_state

    def additive_attn(self, encoder_outputs, hidden):
        Wh = torch.matmul(self.additive_W1, encoder_outputs.t())
        Ws = torch.matmul(self.additive_W2, hidden.view(-1, 1))
        attn_outs = torch.matmul(self.additive_v, torch.tanh(Wh + Ws))
        scores = torch.softmax(attn_outs, dim=0)
        weighted = torch.matmul(scores, encoder_outputs).view(1, -1)
        return weighted

    def multiplicative_attn(self, encoder_outputs, hidden):
        attn_outs = torch.matmul(encoder_outputs, torch.matmul(self.multiplicative_W, hidden))
        scores = torch.softmax(attn_outs, dim=0)
        weighted = torch.matmul(scores, encoder_outputs).view(1, -1)
        return weighted

    def dot_product_attn(self, encoder_outputs, hidden):
        attn_outs = torch.matmul(encoder_outputs, hidden)
        scores = torch.softmax(attn_outs, dim=0)
        weighted = torch.matmul(scores, encoder_outputs).view(1, -1)
        return weighted

    def decode(self, input, hidden, encoder_outputs):
        input_embed = self.decoder_embedding(input).view(1 ,1, -1)
        output, (hidden, cell_state) = self.decoder_lstm(input_embed, hidden)
        if self.attn_type == 'dot_product':
            weighted = self.dot_product_attn(encoder_outputs, hidden[0][0])
        elif self.attn_type == 'multiplicative':
            weighted = self.multiplicative_attn(encoder_outputs, hidden[0][0])
        elif self.attn_type == 'additive':
            weighted = self.additive_attn(encoder_outputs, hidden[0][0])
        output = self.softmax(self.time_dep(torch.cat((output[0], weighted), dim=1)))
        return output, hidden, cell_state

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
