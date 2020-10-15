import math
import torch
import torch.nn as nn

from .utils import log_sum_exp
from .lm import LSTM_LM


class VAE(nn.Module):
    """VAE with normal prior"""
    def __init__(self, encoder, decoder, args, encode_length):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.args = args

        self.nz = args.nz

        loc = torch.zeros(self.nz, device=args.device)
        scale = torch.ones(self.nz, device=args.device)

        self.prior = torch.distributions.normal.Normal(loc, scale)
        self.encode_length = encode_length

    def encode(self, x, nsamples=1, return_parameters=False):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """
        return self.encoder.encode(x, nsamples, return_parameters)

    def encode_stats(self, x):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the mean of latent z with shape [batch, nz]
            Tensor2: the logvar of latent z with shape [batch, nz]
        """

        return self.encoder.encode_stats(x)

    def decode(self, z, strategy, K=10, no_unk=False):
        """generate samples from z given strategy

        Args:
            z: [batch, nsamples, nz]
            strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter

        Returns: List1
            List1: a list of decoded word sequence
        """

        if strategy == "beam":
            return self.decoder.beam_search_decode(z, K=K, no_unk=no_unk)
        elif strategy == "greedy":
            return self.decoder.greedy_decode(z, no_unk)
        elif strategy == "sample":
            return self.decoder.sample_decode(z, no_unk)
        else:
            raise ValueError("the decoding strategy is not supported")


    def reconstruct(self, x, decoding_strategy="greedy", K=5, force_length=-1, no_unk=False):
        """reconstruct from input x

        Args:
            x: (batch, *)
            decoding_strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter

        Returns: List1
            List1: a list of decoded word sequence
        """
        z = self.sample_from_inference(x)
        bs = z.size(0)
        if self.encode_length in [1,2]:
            if force_length > 0:
                n_words = force_length
            else:
                n_words = x.size(1)
            z = self.encode_len(z, n_words, bs, 1)
            #print("VAE recon", z.size())
            #length_input = length_input.squeeze(1)
            #length_input = length_input.to(x.device).expand(bs, 3)
        # squeeze nsamples dimension again
        z = z.squeeze(1) 
        return self.decode(z, decoding_strategy, K, no_unk)

    def encode_len(self, z, n, bs, nsamples):
        """ Return a vector indicating the length of the desired sequence,
        ready to be concatenated to the input of the decoder.
        
        Returns tensor of size bs x nsamples x n_words x dim_code_len or 
        bs x nsamples x dim_code_len if the code is constant throughout the
        sequence
        dim_code_len depends on type_code
        """
        if self.encode_length == 1:
            hundreds = n // 100
            deci = (n - hundreds*100) // 10
            units = (n - hundreds*100 - deci*10)
            code = torch.Tensor([hundreds, deci, units]) / 10
            code = code.expand(bs, nsamples, 3) # bs, n_s, 3
            z = z.expand(nsamples, bs, -1).permute(1,0,2)
        elif self.encode_length == 2:
            code = - torch.log(torch.arange(1,n+1).float().flip(0)) + 2
            code = code.expand(bs, nsamples, 1, n).transpose(2,3) # bs, n_s, n, 1  
            z = z.expand(1, n, bs, -1).permute(2,0,1,3)
        z = torch.cat([z, code.to(z.device)], -1)
        #print("ENCODE", z.size())
        return z

    def loss(self, x, kl_weight, nsamples=1, sum_over_len=True):
        """
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list

        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: total loss [batch]
            Tensor2: reconstruction loss shape [batch]
            Tensor3: KL loss shape [batch]
        """
        z, KL = self.encode(x, nsamples)
        bs = z.size(0)
        n_words = x.size(1)
        if self.encode_length in [1,2]:
            z = self.encode_len(z.transpose(0,1), n_words, bs, nsamples)
            #z = z.expand(nsamples, n_words, bs, -1).permute(2,0,1,3)
            #z = torch.cat([z, length_input.to(x.device)], -1)
        reconstruct_err = self.decoder.reconstruct_error(x, z, sum_over_len)
        # average over nsamples
        reconstruct_err = reconstruct_err.mean(dim=1)
        # (batch) or (batch, len) if sum_over_len=False
        if sum_over_len:
            return reconstruct_err + kl_weight * KL, reconstruct_err, KL
        else:
            return reconstruct_err + kl_weight * KL.unsqueeze(1), reconstruct_err, KL


    def loss_iw(self, x, kl_weight, nsamples=50, ns=10):
        """
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list

        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: total loss [batch]
            Tensor2: reconstruction loss shape [batch]
            Tensor3: KL loss shape [batch]
        """
        # TODO remove: identical to nll_iw?
        mu, logvar = self.encoder.forward(x)

        ##################
        # compute KL
        ##################
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        tmp = []
        reconstruct_err_sum = 0

        #import pdb

        for _ in range(int(nsamples / ns)):

            # (batch, nsamples, nz)
            z = self.encoder.reparameterize(mu, logvar, ns)

            ##################
            # compute qzx
            ##################
            nz = z.size(2)

            # (batch_size, 1, nz)
            _mu, _logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
            var = _logvar.exp()

            # (batch_size, nsamples, nz)
            dev = z - _mu

            # (batch_size, nsamples)
            # compute (positive) log likelihood of the prior and posterior
            log_two_pi = math.log(2*math.pi)
            log_pz = (-0.5 * log_two_pi - z**2 / 2).sum(dim=-1)
            log_qzx = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                0.5 * (nz * log_two_pi + _logvar.sum(-1))



            ##################
            # compute reconstruction loss
            ##################
            reconstruct_err = self.decoder.reconstruct_error(x, z) # bs, nsamples
            reconstruct_err_sum += reconstruct_err.cpu().detach().sum(dim=1)
            # Sum over all samples

            #pdb.set_trace()

            tmp.append(reconstruct_err + kl_weight * (log_qzx - log_pz))

        nll_iw = log_sum_exp(torch.cat(tmp, dim=-1), dim=-1) - math.log(nsamples)

        return nll_iw, reconstruct_err_sum / nsamples, KL


    def nll_iw(self, x, nsamples, ns=100):
        """compute the importance weighting estimate of the log-likelihood
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list
            nsamples: Int
                the number of samples required to estimate marginal data likelihood
        Returns: Tensor1
            Tensor1: the estimate of log p(x), shape [batch]
        """

        # compute iw every ns samples to address the memory issue
        # nsamples = 500, ns = 100
        # nsamples = 500, ns = 10

        # TODO: note that x is forwarded twice in self.encoder.sample(x, ns) and self.eval_inference_dist(x, z, param)
        #.      this problem is to be solved in order to speed up
        #assert(ns > nsamples and nsamples % ns == 0)
        tmp = []
        #print(nsamples, ns, nsamples/ns)
        for _ in range(int(nsamples / ns)):
            # [batch, ns, nz]
            # param is the parameters required to evaluate q(z|x)
            z, param = self.encoder.sample(x, ns)

            # [batch, ns]
            log_comp_ll = self.eval_complete_ll(x, z)
            log_infer_ll = self.eval_inference_dist(x, z, param)

            tmp.append(log_comp_ll - log_infer_ll)
        #print(len(tmp))

        ll_iw = log_sum_exp(torch.cat(tmp, dim=-1), dim=-1) - math.log(nsamples)

        return -ll_iw

    def KL(self, x):
        _, KL = self.encode(x, 1)

        return KL

    def eval_prior_dist(self, zrange):
        """perform grid search to calculate the true posterior
        Args:
            zrange: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/space
        """

        # (k^2)
        return self.prior.log_prob(zrange).sum(dim=-1)

    def eval_complete_ll(self, x, z):
        """compute log p(z,x)
        Args:
            x: Tensor
                input with shape [batch, seq_len]
            z: Tensor
                evaluation points with shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log p(z,x) Tensor with shape [batch, nsamples]
        """

        # [batch, nsamples]
        log_prior = self.eval_prior_dist(z)
        log_gen = self.eval_cond_ll(x, z)

        return log_prior + log_gen

    def eval_cond_ll(self, x, z):
        """compute log p(x|z)
        """

        return self.decoder.log_probability(x, z)

    def eval_log_model_posterior(self, x, grid_z):
        """perform grid search to calculate the true posterior
         this function computes p(z|x)
        Args:
            grid_z: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/pace

        Returns: Tensor
            Tensor: the log posterior distribution log p(z|x) with
                    shape [batch_size, K^2]
        """
        try:
            batch_size = x.size(0)
        except:
            batch_size = x[0].size(0)

        # (batch_size, k^2, nz)
        grid_z = grid_z.unsqueeze(0).expand(batch_size, *grid_z.size()).contiguous()

        # (batch_size, k^2)
        log_comp = self.eval_complete_ll(x, grid_z)

        # normalize to posterior
        log_posterior = log_comp - log_sum_exp(log_comp, dim=1, keepdim=True)

        return log_posterior

    def sample_from_inference(self, x, nsamples=1):
        """perform sampling from inference net
        Returns: Tensor
            Tensor: samples from infernece nets with
                shape (batch_size, nsamples, nz)
        """
        z, _ = self.encoder.sample(x, nsamples)

        return z


    def sample_from_posterior(self, x, nsamples):
        """perform MH sampling from model posterior
        Returns: Tensor
            Tensor: samples from model posterior with
                shape (batch_size, nsamples, nz)
        """

        # use the samples from inference net as initial points
        # for MCMC sampling. [batch_size, nsamples, nz]
        cur = self.encoder.sample_from_inference(x, 1)
        cur_ll = self.eval_complete_ll(x, cur)
        total_iter = self.args.mh_burn_in + nsamples * self.args.mh_thin
        samples = []
        for iter_ in range(total_iter):
            next = torch.normal(mean=cur,
                std=cur.new_full(size=cur.size(), fill_value=self.args.mh_std))
            # [batch_size, 1]
            next_ll = self.eval_complete_ll(x, next)
            ratio = next_ll - cur_ll

            accept_prob = torch.min(ratio.exp(), ratio.new_ones(ratio.size()))

            uniform_t = accept_prob.new_empty(accept_prob.size()).uniform_()

            # [batch_size, 1]
            mask = (uniform_t < accept_prob).float()

            mask_ = mask.unsqueeze(2)

            cur = mask_ * next + (1 - mask_) * cur
            cur_ll = mask * next_ll + (1 - mask) * cur_ll

            if iter_ >= self.args.mh_burn_in and (iter_ - self.args.mh_burn_in) % self.args.mh_thin == 0:
                samples.append(cur.unsqueeze(1))


        return torch.cat(samples, dim=1)

    def calc_model_posterior_mean(self, x, grid_z):
        """compute the mean value of model posterior, i.e. E_{z ~ p(z|x)}[z]
        Args:
            grid_z: different z points that will be evaluated, with
                    shape (k^2, nz), where k=(zmax - zmin)/pace
            x: [batch, *]

        Returns: Tensor1
            Tensor1: the mean value tensor with shape [batch, nz]

        """

        # [batch, K^2]
        log_posterior = self.eval_log_model_posterior(x, grid_z)
        posterior = log_posterior.exp()

        # [batch, nz]
        return torch.mul(posterior.unsqueeze(2), grid_z.unsqueeze(0)).sum(1)

    def calc_infer_mean(self, x):
        """
        Returns: Tensor1
            Tensor1: the mean of inference distribution, with shape [batch, nz]
        """

        mean, logvar = self.encoder.forward(x)

        return mean



    def eval_inference_dist(self, x, z, param=None):
        """
        Returns: Tensor
            Tensor: the posterior density tensor with
                shape (batch_size, nsamples)
        """
        return self.encoder.eval_inference_dist(x, z, param)

    def calc_mi_q(self, x):
        """Approximate the mutual information between x and z
        under distribution q(z|x)

        Args:
            x: [batch_size, *]. The sampled data to estimate mutual info
        """

        return self.encoder.calc_mi(x)
