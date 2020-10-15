from itertools import chain
import math
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .gaussian_encoder import GaussianEncoderBase
from ..utils import log_sum_exp

def compute_ifoc_gates_rnn(rnn, dim, input, prev_hidden, prev_cell):
    #print("DBG alt LAYER={}".format(i))
    gates_preac = torch.mm(rnn.weight_ih, input) + rnn.bias_ih.unsqueeze(1) + \
                  torch.mm(rnn.weight_hh, prev_hidden) + rnn.bias_hh.unsqueeze(1)
    input_gate  = torch.sigmoid(gates_preac[:dim])
    forget_gate = torch.sigmoid(gates_preac[dim:2*dim])
    cell_gate   = torch.tanh(gates_preac[2*dim:3*dim])
    output_gate = torch.sigmoid(gates_preac[3*dim:4*dim])
    new_cell = (cell_gate * input_gate + forget_gate * prev_cell).t()
    new_hid = (output_gate * torch.tanh(new_cell.t())).t()
    #assert(approx_equal(new_cell, cell))
    #assert(approx_equal(new_hid, hidden))
    return input_gate, forget_gate, output_gate, cell_gate

class GaussianPoolEncoder(GaussianEncoderBase):
    def __init__(self, args, vocab_size, model_init, emb_init, enc_type, skip_first_word):
        super(GaussianPoolEncoder, self).__init__()
        self.ni = args.ni
        self.nz = args.nz
        self.args = args

        self.embed = nn.Embedding(vocab_size, args.ni)
        self.enc_type = enc_type
        if enc_type == 'max_avg_pool':
            in_size = 2 * args.ni
        elif enc_type in ['max_pool', 'avg_pool']:
            in_size = args.ni
        else:
            raise NotImplementedError()
        if args.enc_nh == -1:
            self.transform = nn.Linear(in_size, 2 * args.nz, bias=False)
        else:
            # Could be interesting to experiment w/ non-linear extractors in
            # the future
            raise NotImplementedError('MLP not yet implemented')

        self.skip_first_word = skip_first_word

        self.reset_parameters(model_init, emb_init)

    def compute_local_features(self, input):
        # (batch_size, seq_len-1, args.ni)
        return self.embed(input)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, input):
        """
        Args:
            x: (batch_size, seq_len)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """

        # (batch_size, seq_len-1, args.ni)
        word_embed = self.embed(input)
        if self.skip_first_word:
            word_embed = word_embed[:,2:] # skip BOS and first word
        if self.enc_type == 'max_avg_pool':
            mean = word_embed.mean(dim=1)
            max_, _ = word_embed.max(dim=1)
            sentence_repr = torch.cat([mean, max_], dim=1)
        elif self.enc_type == 'max_pool':
            sentence_repr, _ = word_embed.max(dim=1)
        elif self.enc_type == 'avg_pool':
            mean = word_embed.mean(dim=1)
            sentence_repr = mean
        mean, logvar = self.transform(sentence_repr).chunk(2, -1)

        # fix variance as a pre-defined value
        if self.args.fix_var > 0:
            logvar = mean.new_tensor([[[math.log(self.args.fix_var)]]]).expand_as(mean)
        return mean, logvar

class GaussianLSTMEncoder(GaussianEncoderBase):
    """Gaussian LSTM Encoder with constant-length input"""
    def __init__(self, args, vocab_size, model_init, emb_init, pooling=None):
        super(GaussianLSTMEncoder, self).__init__()
        if not hasattr(args, 'enc_reverse'):
            args.enc_reverse = False
        self.ni = args.ni
        self.nh = args.enc_nh
        self.nz = args.nz
        self.args = args
        self.reverse_input = args.enc_reverse

        self.embed = nn.Embedding(vocab_size, args.ni)

        self.lstm = nn.LSTM(input_size=args.ni,
                            hidden_size=args.enc_nh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0)
        assert(pooling in [None, 'max', 'avg', 'max_avg'])
        self.pooling = pooling
        if pooling == 'max_avg':
            in_linear = args.enc_nh * 2
        else:
            in_linear = args.enc_nh
        self.linear = nn.Linear(in_linear, 2 * args.nz, bias=False)
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        # for name, param in self.lstm.named_parameters():
        #     # self.initializer(param)
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #         # model_init(param)
        #     elif 'weight' in name:
        #         model_init(param)

        # model_init(self.linear.weight)
        # emb_init(self.embed.weight)
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def compute_local_features(self, input):
        # (batch_size, seq_len-1, args.ni)
        if self.reverse_input:
            input = input.flip((1,))#[:,1:-1]
        word_embed = self.embed(input)
        hidden_states, _ = self.lstm(word_embed)
        return hidden_states

    def forward(self, input):
        """
        Args:
            x: (batch_size, seq_len)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """
        # (batch_size, seq_len-1, args.ni)
        if self.reverse_input:
            # the following condition is because of a bug in pytorch
            # can't flip tensors with dim (1, n)
            # returns DIFFERENT tensor of same shape
            if input.size(0) > 1:
                input = input.flip((1,))#[:,1:-1]
            else:
                raise NotImplementedError()
        word_embed = self.embed(input)
        hidden_states, (last_state, _) = self.lstm(word_embed)
        if self.pooling == 'avg':
            sentence_repr = hidden_states.mean(dim=1).unsqueeze(0) # batch first LSTM
        elif self.pooling == 'max':
            sentence_repr = hidden_states.max(dim=1)[0].unsqueeze(0) # batch first LSTM
        elif self.pooling == 'max_avg':
            mean = hidden_states.mean(dim=1).unsqueeze(0) # batch first LSTM
            max_ = hidden_states.max(dim=1)[0].unsqueeze(0) # batch first LSTM
            sentence_repr = torch.cat((mean, max_), dim=2)
        else:
            sentence_repr = last_state

        mean, logvar = self.linear(sentence_repr).chunk(2, -1)

        # fix variance as a pre-defined value
        if self.args.fix_var > 0:
            logvar = mean.new_tensor([[[math.log(self.args.fix_var)]]]).expand_as(mean)
        return mean.squeeze(0), logvar.squeeze(0)

    def unroll_cell(self, input, lstm_cell):
        if self.reverse_input:
            input = input.flip((1,))#[:,1:-1]
        word_embed = self.embed(input).transpose(1,0) #L, bs, d
        _, bs, embed_dim = word_embed.size()
        device=word_embed.device
        h = torch.zeros(bs, self.nh, device=device)
        c = torch.zeros(bs, self.nh, device=device)
        input_gates = []
        forget_gates = []
        hidden_sqr_norms = []
        hidden_states = []
        hidden_states_diff = []
        for w in word_embed:
            new_h, c = lstm_cell(w, (h,c))
            h_diff = ((h - new_h)**2)
            hidden_states_diff.append(h_diff)
            h = new_h
            ig, fg, og, cg = compute_ifoc_gates_rnn(lstm_cell, self.nh, w.t(), h.t(), c.t()) 
            input_gates.append(ig)
            forget_gates.append(fg)
            hidden_sqr_norms.append(h_diff)
            mean, logvar = self.linear(h).chunk(2, -1)
            hidden_states.append(h)
        hidden_states = torch.stack(hidden_states) # L, bs, d 
        hidden_states_diff = torch.stack(hidden_states_diff) # L, bs, d 
        diff_of_diff_nor = ((hidden_states_diff[1:] - hidden_states_diff[:-1])**2).sum(2).mean(1)
        print("DIFF OF DIFF", diff_of_diff_nor)
        print(((hidden_states_diff[1:] - hidden_states_diff[:-1])**2).mean(1)[-4])
        print("topk", hidden_states_diff[-5][0].topk(50))
        print("Norms", (hidden_states**2).sum(2).mean(1)[0:10])
        # Take the first 20 dimensions of the hidden states
        # Which dimensions are maximal on the entire sequence?
        idx_max = hidden_states.max(0)[1] # bs, d
        # We expect that some dimensions will always be max with the same timestep
        #print("analysis", idx_max.size(0), (idx_max == idx_max[0]).sum(0).topk(50))
        # Calculer le nombre d'éléments qui sont égaux à batch_size
        n_equal_to_bs = ((idx_max == idx_max[0]).sum(0) == idx_max.size(0)).sum()
        print("FULL", n_equal_to_bs)
        # Ensuite, calculer le nombre qui sont attribués à l'état 0, 1, 2, ...

        # I could store: 
        # All hidden states
        stacked_input_gates = torch.stack(input_gates).permute(2,0,1) # bs, L, dim
        stacked_forget_gates = torch.stack(forget_gates).permute(2,0,1) # bs, L, dim
        #stacked_sqr_norms = torch.stack(hidden_sqr_norms).permute(1,0) # bs, L
        return stacked_input_gates, stacked_forget_gates, stacked_sqr_norms
        
    # def eval_inference_mode(self, x):
    #     """compute the mode points in the inference distribution
    #     (in Gaussian case)
    #     Returns: Tensor
    #         Tensor: the posterior mode points with shape (*, nz)
    #     """

    #     # (batch_size, nz)
    #     mu, logvar = self.forward(x)


class VarLSTMEncoder(GaussianLSTMEncoder):
    """Gaussian LSTM Encoder with variable-length input"""
    def __init__(self, args, vocab_size, model_init, emb_init):
        super(VarLSTMEncoder, self).__init__(args, vocab_size, model_init, emb_init)


    def forward(self, input):
        """
        Args:
            input: tuple which contains x and sents_len
                    x: (batch_size, seq_len)
                    sents_len: long tensor of sentence lengths

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """

        input, sents_len = input
        # (batch_size, seq_len, args.ni)
        word_embed = self.embed(input)

        packed_embed = pack_padded_sequence(word_embed, sents_len.tolist(), batch_first=True)

        _, (last_state, last_cell) = self.lstm(packed_embed)

        mean, logvar = self.linear(last_state).chunk(2, -1)

        return mean.squeeze(0), logvar.squeeze(0)

    def encode(self, input, nsamples, return_parameters=False):
        """perform the encoding and compute the KL term
        Args:
            input: tuple which contains x and sents_len

        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]

        """

        # (batch_size, nz)
        mu, logvar = self.forward(input)

        # (batch, nsamples, nz)
        z = self.reparameterize(mu, logvar, nsamples)
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        if return_parameters:
            return (mu, logvar), KL

        return z, KL



