import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LinearDiscriminator(nn.Module):
    """docstring for LinearDiscriminator"""
    def __init__(self, args, encoder):
        super(LinearDiscriminator, self).__init__()
        self.args = args

        self.encoder = encoder
        if args.freeze_enc:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.linear = nn.Linear(args.nz, args.ncluster)
        self.dropout = nn.Dropout(0.5)
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def get_performance(self, batch_data, batch_labels):
        """ Return loss of discriminator that uses SAMPLES (not mean)"""
        z, _ = self.encoder.encode(batch_data, 1)
        z = z.squeeze(1)
        if not self.args.freeze_enc:
            z = self.dropout(z)
        logits = self.linear(z)
        loss = self.loss(logits, batch_labels)
        _, pred = torch.max(logits, dim=1)
        #correct = torch.eq(pred, batch_labels).float().sum().item()

        return loss, pred
        #_, pred = torch.max(logits, dim=1)
        #correct = torch.eq(pred, batch_labels).float().sum().item()

        #return loss, correct


class MLPDiscriminator(nn.Module):
    """docstring for LinearDiscriminator"""
    def __init__(self, args, encoder):
        super(MLPDiscriminator, self).__init__()
        self.args = args

        self.encoder = encoder
        if args.freeze_enc:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        self.feats = nn.Sequential(
            nn.Linear(args.nz, args.nz*10),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(args.nz*10, args.ncluster),
        )
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def get_performance(self, batch_data, batch_labels):
        """ Return loss of discriminator that uses SAMPLES (not mean)"""
        z, _ = self.encoder.encode(batch_data, 1)
        z = z.squeeze(1)
        logits = self.feats(z)
        loss = self.loss(logits, batch_labels)

        _, pred = torch.max(logits, dim=1)
        #correct = torch.eq(pred, batch_labels).float().sum().item()

        return loss, pred
