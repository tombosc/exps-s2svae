import os
import time
import importlib
import argparse
import random
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, optim

from data import MonoTextData, VocabEntry
from modules.plot_tools import export_avg_loss_per_ts

from exp_utils import create_exp_dir, load_args
from utils import uniform_initializer, xavier_normal_initializer, calc_iwnll, calc_iw_elbo, calc_mi, calc_au, sample_sentences, visualize_latent, reconstruct, create_model, modify_params

clip_grad = 5.0
decay_epoch = 2
lr_decay = 0.5
max_decay = 4

logging = None


def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')

    parser.add_argument('--ignore_load_errors', action='store_true', help='ignore load error to pre-train using different decoders')
    parser.add_argument('--freeze_encoder', action='store_true', help='freeze encoder')
    parser.add_argument('--freeze_encoder_exc', action='store_true', help='freeze encoder except for variational parameter layer')
    parser.add_argument('--load_encoder', type=str, required=False, help='load pretrained model as encoder')
    parser.add_argument('--load_encoder_and_decoder', type=str, required=False, help='load pretrained model as encoder and decoder')
    # model hyperparameters
    parser.add_argument('--dataset', type=str, required=True, help='dataset to use')
    parser.add_argument('--model-type', type=str, default='lstm', help='model type')
    # optimization parameters
    parser.add_argument('--momentum', type=float, default=0, help='sgd momentum')
    parser.add_argument('--opt', type=str, choices=["sgd", "adam"], default="sgd", help='sgd momentum')

    parser.add_argument('--nsamples', type=int, default=1, help='number of samples for training')
    parser.add_argument('--iw_nsamples', type=int, default=500,
                         help='number of samples to compute importance weighted estimate')

    # force model params (override config file)
    parser.add_argument('--encode_length', default=None, help='Encode length (-1 or 2)')
    parser.add_argument('--nz', default=None, help='Size of latent variable')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--eval_iw_elbo', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--eval_valid_elbo', action='store_true', default=False, help='compute elbo on validation set')
    parser.add_argument('--load-args-from-logs', action='store_true', default=False, help='Load args from logs.txt')
    parser.add_argument('--export-avg-loss-per-ts', default=None, help='Export average loss per timestep')
    parser.add_argument('--study-pooling', action='store_true', default=False, help='Study pooling')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--reconstruct_batch_size', type=int, default=32)

    # decoding
    parser.add_argument('--force_absolute_length', type=int, default=-1, help="for models with encode_length=True, force the length of decoded sequence")
    parser.add_argument('--reconstruct_from', type=str, default='', help="the model checkpoint path")
    parser.add_argument('--reconstruct_to', type=str, default="decoding.txt", help="save file")
    parser.add_argument('--reconstruct_add_labels_to_source', action='store_true', help="add the label to the decoded source")
    parser.add_argument('--reconstruct_max_examples', type=int, default=-1, help="Maximum number of examples to decode")
    parser.add_argument('--decoding_strategy', type=str, choices=["greedy", "beam", "sample"], default="greedy")
    parser.add_argument('--no-unk', action='store_true',
        help="Disable sampling of unks")

    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=10, help="number of annealing epochs. warm_up=0 means not anneal")
    parser.add_argument('--kl_start', type=float, default=1.0, help="starting KL weight")


    # inference parameters
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')

    # output directory
    parser.add_argument('--exp_dir', default=None, type=str,
                         help='experiment directory.')
    parser.add_argument("--save_ckpt", type=int, default=0,
                        help="save checkpoint every epoch before this number")
    parser.add_argument("--save_latent", type=int, default=0)

    # new
    parser.add_argument("--fix_var", type=float, default=-1)
    parser.add_argument("--reset_dec", action="store_true", default=False)
    parser.add_argument("--load_best_epoch", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1.)

    parser.add_argument("--fb", type=int, default=0,
                         help="0: no fb; 1: fb; 2: max(target_kl, kl) for each dimension")
    parser.add_argument("--target_kl", type=float, default=-1,
                         help="target kl of the free bits trick")

    args = parser.parse_args()

    # set args.cuda
    args.cuda = torch.cuda.is_available()

    # set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params
    modify_params(params, args.model_type)
    # some params can override configuration files, in which case
    # we remove them from the config
    for p, type_ in [('nz', int), ('encode_length', int)]:
        value_p = getattr(args, p, None)
        if value_p is not None:
            if params.get(p, None) is not None:
                del params[p]
            # convert
            setattr(args, p, type_(value_p))
        else:
            delattr(args, p)
    args = argparse.Namespace(**vars(args), **params)
    args, args_model = load_args(args, parser)
    load_str = "_load" if args.load_path != "" else ""
    if args.fb == 0:
        fb_str = ""
    elif args.fb == 1:
        fb_str = "_fb"
    elif args.fb == 2:
        fb_str = "_fbdim"
    elif args.fb == 3:
        fb_str = "_fb3"

    # set load and save paths
    if args.exp_dir == None:
        args.exp_dir = "exp_{}{}/{}_warm{}_kls{:.1f}{}_tr{}".format(args.dataset,
            load_str, args.dataset, args.warm_up, args.kl_start, fb_str, args.target_kl)
    if len(args.load_path) <= 0 and (args.eval or args.eval_valid_elbo):
        args.load_path = os.path.join(args.exp_dir, 'model.pt')

    args.save_path = os.path.join(args.exp_dir, 'model.pt')

    # set args.label
    if 'label' in params:
        args.label = params['label']
    else:
        args.label = False

    return args, args_model


def study_pooling(model, batch, args, min_doc_size=-1):
    max_L = 250
    if hasattr(model.encoder, 'nh'):
        nh = model.encoder.nh
    else:
        nh = args.ni

    counters_max_component = torch.zeros((nh, max_L))
    # same but backwards (negative indices)
    counters_max_component_b = torch.zeros((nh, max_L))
    n_docs = 0
    for b in batch:
        sent_len = b.size(1)
        if sent_len < min_doc_size:
            continue
        n_docs += b.size(0)
        features = model.encoder.compute_local_features(b)  # bs, L, d
        idx_argmax = features.max(1)[1]  # bs, d
        r = range(nh)
        for argmaxes in idx_argmax:  # d
            counters_max_component[r, argmaxes] += 1
            counters_max_component_b[r, sent_len - argmaxes] += 1
        if n_docs > 1000:
            break
    print(f"{n_docs} documents.")

    def process(counter):
        counter /= n_docs
        # find the position most often activated for each component
        maxima, argmaxima = counter.max(1)  
        # filter only those which are selected more often than X
        for thresh in [0.5, 0.8]:
            large_proportion = maxima > thresh
            print(f"{large_proportion.sum()} / {maxima.size(0)} components are"
                  f" activated at the same position more than {thresh*100}% of"
                  f" the time.")
            # find out which position maximizes these
            print("Favorite positions:",
                  Counter(argmaxima[large_proportion].numpy().tolist()))
        # sort by their frequencies
        maxima_sorted, argsort = maxima.sort()
        return counter[argsort, :].t()[:50]
    plt.figure(figsize=(15, 15))
    plt.subplot(211)
    plt.imshow(process(counters_max_component))
    plt.colorbar()
    plt.subplot(212)
    plt.imshow(process(counters_max_component_b))
    plt.colorbar()
    plt.tight_layout()
    plt.show()



def test(model, test_data_batch, mode, args, verbose=True):
    global logging

    report_kl_loss = report_rec_loss = report_loss = 0
    report_num_words = report_num_sents = 0
    for i in np.random.permutation(len(test_data_batch)):
        batch_data = test_data_batch[i]
        batch_size, sent_len = batch_data.size()

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size
        report_num_sents += batch_size
        loss, loss_rc, loss_kl = model.loss(batch_data, 1.0, nsamples=args.nsamples)
        assert(not loss_rc.requires_grad)

        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()
        loss = loss.sum()

        report_rec_loss += loss_rc.item()
        report_kl_loss += loss_kl.item()
        if args.warm_up == 0 and args.kl_start < 1e-6:
            report_loss += loss_rc.item()
        else:
            report_loss += loss.item()

    mutual_info = calc_mi(model, test_data_batch)

    test_loss = report_loss / report_num_sents

    nll = (report_kl_loss + report_rec_loss) / report_num_sents
    kl = report_kl_loss / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)
    if verbose:
        logging('%s --- avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f, nll: %.4f, ppl: %.4f' % \
               (mode, test_loss, report_kl_loss / report_num_sents, mutual_info,
                report_rec_loss / report_num_sents, nll, ppl))
        #sys.stdout.flush()

    return test_loss, nll, kl, ppl, mutual_info


def main(args, args_model):
    global logging
    eval_mode = (args.reconstruct_from != "" or args.eval or args.eval_iw_elbo or args.eval_valid_elbo or args.export_avg_loss_per_ts or args.study_pooling) # don't make exp dir for reconstruction
    logging = create_exp_dir(args.exp_dir, scripts_to_save=None, debug=eval_mode)

    if args.cuda:
        logging('using cuda')
    logging(str(args))

    opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}

    vocab = {}
    if getattr(args, 'vocab_file', None):
        with open(args.vocab_file, 'r', encoding='utf-8') as fvocab:
            for i, line in enumerate(fvocab):
                vocab[line.strip()] = i

        vocab = VocabEntry(vocab)

    train_data = MonoTextData(args.train_data, label=args.label, vocab=vocab)

    vocab = train_data.vocab
    vocab_size = len(vocab)

    val_data = MonoTextData(args.val_data, label=args.label, vocab=vocab)
    test_data = MonoTextData(args.test_data, label=args.label, vocab=vocab)

    logging('Train data: %d samples' % len(train_data))
    logging('finish reading datasets, vocab size is %d' % len(vocab))
    logging('dropped sentences: %d' % train_data.dropped)
    #sys.stdout.flush()

    log_niter = max((len(train_data)//args.batch_size)//10, 1)

    device = torch.device("cuda" if args.cuda else "cpu")
    vae = create_model(vocab, args, args_model, logging, eval_mode)

    if args.eval:
        logging('begin evaluation')
        vae.eval()
        with torch.no_grad():
            test_data_batch = val_data.create_data_batch(batch_size=1,
                                                          device=device,
                                                          batch_first=True)
            nll, ppl = calc_iwnll(vae, test_data_batch, args, ns=250)
            logging('iw nll: %.4f, iw ppl: %.4f' % (nll, ppl))
        return

    if args.eval_iw_elbo:
        logging('begin evaluation')
        vae.load_state_dict(torch.load(args.load_path))
        vae.eval()
        with torch.no_grad():
            test_data_batch = test_data.create_data_batch(batch_size=1,
                                                          device=device,
                                                          batch_first=True)
            nll, ppl = calc_iw_elbo(vae, test_data_batch, args)
            logging('iw ELBo: %.4f, iw PPL*: %.4f' % (nll, ppl))
        return



    if args.eval_valid_elbo:
        logging('begin evaluation on validation set')
        vae.load_state_dict(torch.load(args.load_path))
        vae.eval()
        val_data_batch = val_data.create_data_batch(batch_size=args.batch_size,
                                                    device=device,
                                                    batch_first=True)

        with torch.no_grad():
            loss, nll, kl, ppl, mi = test(vae, val_data_batch, "VAL", args)
            logging('nll: %.4f, iw ppl: %.4f' % (nll, ppl))
        return


    if args.study_pooling:
        vae.load_state_dict(torch.load(args.load_path))
        vae.eval()
        with torch.no_grad():
            data_batch = train_data.create_data_batch(batch_size=args.batch_size,
                                                          device=device,
                                                          batch_first=True)
            model_dir = os.path.dirname(args.load_path)
            archive_npy = os.path.join(model_dir, 'pooling.npy')
            random.shuffle(data_batch)
            #logs = study_pooling(vae, data_batch, "TRAIN", args, min_doc_size=16)
            logs = study_pooling(vae, data_batch, args, min_doc_size=4)
            logs['exp_dir'] = model_dir
            np.save(archive_npy, logs)
        return

    if args.export_avg_loss_per_ts:
        print("MODEL")
        print(vae)
        export_avg_loss_per_ts(
            vae,
            train_data,
            device,
            args.batch_size,
            args.load_path,
            args.export_avg_loss_per_ts,
        )
        return

    if args.reconstruct_from != "":
        print("begin decoding")
        vae.load_state_dict(torch.load(args.reconstruct_from))
        vae.eval()
        with torch.no_grad():
            if args.reconstruct_add_labels_to_source:
                test_data_batch, test_labels_batch = test_data.create_data_batch_labels(batch_size=args.reconstruct_batch_size,
                                                  device=device,
                                                  batch_first=True,
                                                  deterministic=True)
                c = list(zip(test_data_batch, test_labels_batch))
                #random.shuffle(c)
                test_data_batch, test_labels_batch = zip(*c)
            else:
                test_data_batch = test_data.create_data_batch(batch_size=args.reconstruct_batch_size,
                                                          device=device,
                                                          batch_first=True)
                test_labels_batch = None
                #random.shuffle(test_data_batch)
            # test(vae, test_data_batch, "TEST", args)
            reconstruct(vae, test_data_batch, vocab, args.decoding_strategy, args.reconstruct_to, test_labels_batch, args.reconstruct_max_examples, args.force_absolute_length, args.no_unk)

        return

    if args.freeze_encoder_exc:
        assert args.enc_type == 'lstm'
        enc_params = vae.encoder.linear.parameters()
    else:
        enc_params = vae.encoder.parameters()
    dec_params = vae.decoder.parameters()
    if args.opt == 'sgd':
        optimizer_fn = optim.SGD
    elif args.opt == 'adam':
        optimizer_fn = optim.Adam
    else:
        raise ValueError("optimizer not supported")

    def optimizer_fn_(params):
        return optimizer_fn(params, lr=args.lr, momentum=args.momentum)

    enc_optimizer = optimizer_fn_(enc_params)
    dec_optimizer = optimizer_fn_(dec_params)

    iter_ = decay_cnt = 0
    best_loss = 1e4
    best_kl = best_nll = best_ppl = 0
    vae.train()
    start = time.time()

    kl_weight = args.kl_start
    if args.warm_up > 0:
        anneal_rate = (1.0 - args.kl_start) / (args.warm_up * (len(train_data) / args.batch_size))
    else:
        anneal_rate = 0

    dim_target_kl = args.target_kl / float(args.nz)

    train_data_batch = train_data.create_data_batch(batch_size=args.batch_size,
                                                    device=device,
                                                    batch_first=True)

    val_data_batch = val_data.create_data_batch(batch_size=args.batch_size,
                                                device=device,
                                                batch_first=True)

    test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                  device=device,
                                                  batch_first=True)


    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(args.epochs):
            report_kl_loss = report_rec_loss = report_loss = 0
            report_num_words = report_num_sents = 0

            for i in np.random.permutation(len(train_data_batch)):

                batch_data = train_data_batch[i]
                batch_size, sent_len = batch_data.size()

                # not predict start symbol
                report_num_words += (sent_len - 1) * batch_size
                report_num_sents += batch_size

                kl_weight = min(1.0, kl_weight + anneal_rate)
                
                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                
                if args.fb == 0:
                    loss, loss_rc, loss_kl = vae.loss(batch_data, kl_weight, nsamples=args.nsamples)
                elif args.fb == 1:
                    loss, loss_rc, loss_kl = vae.loss(batch_data, kl_weight, nsamples=args.nsamples, sum_over_len = False)
                    kl_mask = (loss_kl > args.target_kl).float()
                    loss_rc = loss_rc.sum(-1)
                    loss = loss_rc + kl_mask * kl_weight * loss_kl 
                elif args.fb == 2:
                    mu, logvar = vae.encoder(batch_data)
                    z = vae.encoder.reparameterize(mu, logvar, args.nsamples)
                    loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
                    kl_mask = (loss_kl > dim_target_kl).float()
                    fake_loss_kl = (kl_mask * loss_kl).sum(dim=1)
                    loss_rc = vae.decoder.reconstruct_error(batch_data, z).mean(dim=1)
                    loss = loss_rc + kl_weight * fake_loss_kl
                loss = loss.mean(dim=-1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

                loss_rc = loss_rc.sum()
                loss_kl = loss_kl.sum()
                
                if not args.freeze_encoder:
                    enc_optimizer.step()
                dec_optimizer.step()

                report_rec_loss += loss_rc.item()
                report_kl_loss += loss_kl.item()
                report_loss += loss_rc.item() + loss_kl.item()

                if iter_ % log_niter == 0:
                    train_loss = report_loss / report_num_sents

                    logging('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, recon: %.4f,' \
                           'time %.2fs, kl_weight %.4f' %
                           (epoch, iter_, train_loss, report_kl_loss / report_num_sents,
                           report_rec_loss / report_num_sents, time.time() - start, kl_weight))

                    report_rec_loss = report_kl_loss = report_loss = 0
                    report_num_words = report_num_sents = 0
                iter_ += 1

            logging('kl weight %.4f' % kl_weight)
            logging('lr {}'.format(opt_dict["lr"]))

            vae.eval()
            with torch.no_grad():
                loss, nll, kl, ppl, mi = test(vae, val_data_batch, "VAL", args)
                au, au_var = calc_au(vae, val_data_batch)
                logging("%d active units" % au)

            if args.save_ckpt > 0 and epoch <= args.save_ckpt:
                logging('save checkpoint')
                torch.save(vae.state_dict(), os.path.join(args.exp_dir, f'model_ckpt_{epoch}.pt'))

            if loss < best_loss:
                logging('update best loss')
                best_loss = loss
                best_nll = nll
                best_kl = kl
                best_ppl = ppl
                torch.save(vae.state_dict(), args.save_path)

            if loss > opt_dict["best_loss"]:
                opt_dict["not_improved"] += 1
                if opt_dict["not_improved"] >= decay_epoch and epoch >= args.load_best_epoch:
                    opt_dict["best_loss"] = loss
                    opt_dict["not_improved"] = 0
                    opt_dict["lr"] = opt_dict["lr"] * lr_decay
                    vae.load_state_dict(torch.load(args.save_path))
                    logging('new lr: %f' % opt_dict["lr"])
                    decay_cnt += 1
                    enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
                    dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)

            else:
                opt_dict["not_improved"] = 0
                opt_dict["best_loss"] = loss

            if decay_cnt == max_decay:
                break

            if args.save_latent > 0 and epoch <= args.save_latent:
                visualize_latent(args, epoch, vae, "cuda", test_data)

            vae.train()

    except KeyboardInterrupt:
        logging('-' * 100)
        logging('Exiting from training early')

    # compute importance weighted estimate of log p(x)
    vae.load_state_dict(torch.load(args.save_path))

    vae.eval()
    with torch.no_grad():
        loss, nll, kl, ppl, _ = test(vae, test_data_batch, "TEST", args)
        au, au_var = calc_au(vae, test_data_batch)

if __name__ == '__main__':
    args, args_model = init_config()
    main(args, args_model)
