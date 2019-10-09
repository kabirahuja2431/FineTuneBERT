import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import time
from src.model import SentimentClassifier
from src.dataloader import SSTDataset
import pdb
import glob

def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc

def evaluate(net, criterion, dataloader, args):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for seq, attn_masks, labels in dataloader:
            seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)
            logits = net(seq, attn_masks)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1

    return mean_acc / count, mean_loss / count



def train(net, criterion, opti, train_loader, val_loader, args):

    best_acc = 0
    for ep in range(args.max_eps):
        
        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            #Clear gradients
            opti.zero_grad()  
            #Converting these to cuda tensors
            seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)

            #Obtaining the logits from the model
            logits = net(seq, attn_masks)

            #Computing loss
            loss = criterion(logits.squeeze(-1), labels.float())

            #Backpropagating the gradients
            loss.backward()

            #Optimization step
            opti.step()

            if it % args.print_every == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(it, ep, loss.item(), acc))

        
        val_acc, val_loss = evaluate(net, criterion, val_loader, args)
        print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(ep, val_acc, val_loss))
        if val_acc > best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
            best_acc = val_acc
            torch.save(net.state_dict(), 'Models/sstcls_{}_freeze_{}.dat'.format(ep, args.freeze_bert))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type = int, default = 0)
    parser.add_argument('-freeze_bert', action='store_true')
    parser.add_argument('-maxlen', type = int, default= 25)
    parser.add_argument('-batch_size', type = int, default= 32)
    parser.add_argument('-lr', type = float, default = 2e-5)
    parser.add_argument('-print_every', type = int, default= 100)
    parser.add_argument('-max_eps', type = int, default= 5)
    args = parser.parse_args()

    #Instantiating the classifier model
    print("Building model! (This might take time if you are running this for first time)")
    st = time.time()
    net = SentimentClassifier(args.freeze_bert)
    net.cuda(args.gpu) #Enable gpu support for the model
    print("Done in {} seconds".format(time.time() - st))

    print("Creating criterion and optimizer objects")
    st = time.time()
    criterion = nn.BCEWithLogitsLoss()
    opti = optim.Adam(net.parameters(), lr = args.lr)
    print("Done in {} seconds".format(time.time() - st))

    #Creating dataloaders
    print("Creating train and val dataloaders")
    st = time.time()
    train_set = SSTDataset(filename = 'data/SST-2/train.tsv', maxlen = args.maxlen)
    val_set = SSTDataset(filename = 'data/SST-2/dev.tsv', maxlen = args.maxlen)
    train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers = 5)
    val_loader = DataLoader(val_set, batch_size = args.batch_size, num_workers = 5)
    print("Done in {} seconds".format(time.time() - st))

    print("Let the training begin")
    st = time.time()
    train(net, criterion, opti, train_loader, val_loader, args)
    print("Done in {} seconds".format(time.time() - st))


