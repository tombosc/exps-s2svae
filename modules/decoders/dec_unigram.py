import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np

from .decoder import DecoderBase
from .decoder_helper import BeamSearchNode

class UnigramDecoder(DecoderBase):
    """Unigram decoder"""
    def __init__(self, args, vocab, model_init, emb_init):
        super(UnigramDecoder, self).__init__()
        self.ni = args.ni
        self.nh = args.dec_nh
        self.nz = args.nz
        self.vocab = vocab
        self.device = args.device

        # no padding when setting padding_idx to -1
        #self.embed = nn.Embedding(len(vocab), args.ni, padding_idx=-1)

        self.dropout = nn.Dropout(args.dec_dropout)

        # for initializing hidden state and cell
        self.first_layer = nn.Linear(args.nz, args.dec_nh, bias=True)

        pred_linear_bias = getattr(args, 'pred_linear_bias', False)
        # prediction layer
        self.pred_layer = nn.Linear(args.dec_nh, len(vocab), bias=pred_linear_bias)

        vocab_mask = torch.ones(len(vocab))
        # vocab_mask[vocab['<pad>']] = 0
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        #emb_init(self.embed.weight)

    def sample_text(self, input, z, EOS, device):
        raise NotImplementedError()

    def decode(self, input, z):
        """
        Args:
            input: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        """

        # not predicting start symbol
        # sents_len -= 1

        batch_size, n_sample, _ = z.size()
        seq_len = input.size(1)

        # (batch_size, seq_len, ni)
        #word_embed = self.dropout(word_embed)

        if n_sample == 1:
            z_ = z.expand(batch_size, seq_len, self.nz)
        else:
            word_embed = word_embed.unsqueeze(1).expand(batch_size, n_sample, seq_len, self.ni) \
                                   .contiguous()

            # (batch_size * n_sample, seq_len, ni)
            word_embed = word_embed.view(batch_size * n_sample, seq_len, self.ni)

            z_ = z.unsqueeze(2).expand(batch_size, n_sample, seq_len, self.nz).contiguous()
            z_ = z_.view(batch_size * n_sample, seq_len, self.nz)
        h = self.first_layer(z_)
        h = F.relu(self.dropout(h))
        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.pred_layer(h)

        return output_logits

    def reconstruct_error(self, x, z, sum_over_len=True):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """

        #remove end symbol
        src = x[:, :-1]

        # remove start symbol
        tgt = x[:, 1:]

        batch_size, seq_len = src.size()
        n_sample = z.size(1)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.decode(src, z)

        if n_sample == 1:
            tgt = tgt.contiguous().view(-1)
        else:
            # (batch_size * n_sample * seq_len)
            tgt = tgt.unsqueeze(1).expand(batch_size, n_sample, seq_len) \
                     .contiguous().view(-1)

        # (batch_size * n_sample * seq_len)
        loss = self.loss(output_logits.view(-1, output_logits.size(2)),
                         tgt)


        if sum_over_len:
            # (batch_size, n_sample)
            return loss.view(batch_size, n_sample, -1).sum(-1)
        else:
            # bs, n_sample, len
            return loss.view(batch_size, n_sample, -1)


    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """

        return -self.reconstruct_error(x, z)

    def beam_search_decode(self, z, K=5):
        """beam search decoding, code is based on
        https://github.com/pcyin/pytorch_basic_nmt/blob/master/nmt.py

        the current implementation decodes sentence one by one, further batching would improve the speed

        Args:
            z: (batch_size, nz)
            K: the beam width

        Returns: List1
            List1: the decoded word sentence list
        """
        raise NotImplementedError()

    def greedy_decode(self, z):
        raise NotImplementedError()

    def sample_decode(self, z, greedy=False):
        """sample/greedy decoding from z
        Args:
            z: (batch_size, nz)

        Returns: List1
            List1: the decoded word sentence list
        """
        raise NotImplementedError()
