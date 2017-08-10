import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import argparse
from RNN_PT import RNN_LM
from utils import Textdataset
import pdb

torch.manual_seed(1234)

def main():
    parser = argparse.ArgumentParser()
    # Number of Layers
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of stacked RNN layers')
    # Cell Type
    parser.add_argument('--cell_type', type=str, default='gru',
                        help='rnn, lstm, gru')
    # State Size
    parser.add_argument('--state_size', type=int, default=100,
                        help='Number of hidden neurons of RNN cells')
    #Embedding Size
    parser.add_argument('--embedding_size', type=int, default=10,
                        help='learning rate')
    # 1-Drop out
    parser.add_argument('--keep_prob', type=int, default=0.9,
                        help='keeping probability(1-dropout)')
    
    # Length of Unrolled RNN (Sequence Length)
    parser.add_argument('--seq_length', type=int, default=200,
                        help='maximum sequences considered for backprop')
    # Number of Training Epoch
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of epochs')
    # Learning Rate
    parser.add_argument('--lr', type=int, default=0.01,
                        help='learning rate')
    # Training Device
    parser.add_argument('--device', type=str, default='cpu',
                        help='for cpu: \'cpu\', for gpu: \'gpu\'')
    # Batch Size
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Size of batches for training')

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default='./save/model',
                        help='Name of saved model.')

    args = parser.parse_args()
    train(args)

def train(args):
    dataset = Textdataset(args.batch_size, args.seq_length)
    RNN_model = RNN_LM(args, dataset.vocab_size)
    RNN_model.train(True)
    optimizer, loss = optim.Adam(RNN_model.parameters(), lr=args.lr), nn.CrossEntropyLoss()
    if args.device == 'gpu' and torch.cuda.is_available():
        RNN_model = RNN_model.cuda()

    # Training
    start_process = time.time()
    for epoch in range(args.num_epochs):
        start_epoch = time.time()
        avg_loss = 0
        total_batches = dataset.total_batches
        # Loop over batches
        for i in range(total_batches):
            optimizer.zero_grad()
            batch_x, batch_y = dataset.next_batch()

            if args.device == 'gpu' and torch.cuda.is_available():
                batch_x = Variable(torch.from_numpy(batch_x).type('torch.LongTensor')).cuda()
                batch_y = Variable(torch.from_numpy(batch_y).type('torch.LongTensor')).contiguous().view(-1,1).squeeze(-1).cuda()
            else:    
                batch_x = Variable(torch.from_numpy(batch_x).type('torch.LongTensor'))
                batch_y = Variable(torch.from_numpy(batch_y).type('torch.LongTensor')).contiguous().view(-1,1).squeeze(-1)
            
            output = RNN_model(batch_x)
            batch_loss = loss(output, batch_y)
            batch_loss.backward()
            optimizer.step()        
            avg_loss += batch_loss.data[0]/args.batch_size
        end_epoch = time.time()
        print("Epoch:", epoch+1, "Train Loss:",avg_loss/total_batches,"in:", int(end_epoch - start_epoch), "sec")
        torch.save(RNN_model.state_dict(), '%s/checkpoint.pth' % (args.checkpoint))
    end_process = time.time()
    print("Train completed in:",int(end_process - start_process), "sec")


if __name__ == '__main__':
    main()