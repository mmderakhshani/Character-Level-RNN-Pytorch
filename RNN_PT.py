import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

# RNN for Language Modeling
class RNN_LM(nn.Module):
    def __init__(self, args, vocab_size, generation = False):
        super(RNN_LM, self).__init__()
        
        self.generation = generation
        self.args = args
        self.device = args.device
        self.cell_type = args.cell_type
        self.state_size = args.state_size
        self.embedding_size = args.embedding_size
        self.keep_prob = args.keep_prob
        self.num_layers = args.num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.net = self.rnn_cell()

        if generation:
            self.batch_size = 1
        else:
            self.batch_size = args.batch_size

        self.h0 = Variable(torch.randn(args.num_layers, self.batch_size, self.state_size))
        if self.cell_type == 'lstm':
            self.c0 = Variable(torch.randn(args.num_layers, self.batch_size, self.state_size))

        self.logits = nn.Linear(in_features = self.state_size,out_features = vocab_size)
        self.preds = nn.LogSoftmax()

    def forward(self, inputs):
        if not self.generation:
            embedding = self.embedding(inputs)
            embedding = embedding.permute(1,0,2)
            if self.cell_type == 'rnn':
                output, hn = self.net(embedding, self.h0)
            elif self.cell_type == 'gru':
                output, hn = self.net(embedding, self.h0)
            elif self.cell_type == 'lstm':
                output, hn = self.net(embedding, (self.h0, self.c0))
            output = output.contiguous().view(-1, self.state_size)
            output = self.logits(output)
            return output
        else:
            embedding = self.embedding(inputs)
            embedding = embedding.permute(1,0,2)
            if self.cell_type == 'rnn':
                output, hn = self.net(embedding, self.h0)
                self.h0 = hn
            elif self.cell_type == 'gru':
                output, hn = self.net(embedding, self.h0)
                self.h0 = hn
            elif self.cell_type == 'lstm':
                output, hn = self.net(embedding, (self.h0, self.c0))
                self.h0 = hn[0]
                self.c0 = hn[1]
            output = output.contiguous().view(-1, self.state_size)
            output = self.logits(output)
            output = self.preds(output)
            return output
    
    def rnn_cell(self):
        if self.cell_type is 'rnn':
            net = nn.RNN(input_size=self.embedding_size, 
                hidden_size=self.state_size, 
                nonlinearity='relu',
                dropout=1 - self.keep_prob,
                num_layers=self.num_layers)

        elif self.cell_type is 'gru':
            net = nn.GRU(input_size=self.embedding_size, 
                hidden_size=self.state_size, 
                dropout=1 - self.keep_prob,
                num_layers=self.num_layers)
        elif self.cell_type is 'lstm':
            net = nn.LSTM(input_size=self.embedding_size, 
                hidden_size=self.state_size, 
                dropout=1 - self.keep_prob,
                num_layers=self.num_layers)
        else:
            net = None
        return net