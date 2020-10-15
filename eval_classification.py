""" Evaluate models on the semi-supervised learning task with
repeated-stratified K-Fold cross-val.
"""
import os
import importlib
import argparse
import json
import glob
import warnings

from sklearn import clone
from sklearn.metrics import f1_score, log_loss, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

from data import MonoTextData, VocabEntry
from modules import (
    VAE, GaussianLSTMEncoder, LSTMDecoder, GaussianPoolEncoder, UnigramDecoder
)

from exp_utils import get_logger_existing_dir, load_args

logging = None


def augment_cross_val(y, n_resample, n_splits, n_repeats):
    skf = RepeatedStratifiedKFold(n_splits, random_state=1,
                                  n_repeats=n_repeats)

    def augm(l):
        ext = [list(range(e*n_resample, (e+1)*n_resample)) for e in l]
        return [item for sublist in ext for item in sublist]

    for train_index, test_index in skf.split(y, y):
        yield augm(train_index), augm(test_index)


def augment_dataset(X, n_resample, split, model):
    """ Given X a matrix of size n, 2d where [:,:d] are means and [:,d:] are std
        or n, d where each row is a probability distribution,
        Returns a matrix of samples either from Gaussian (first case) or
        Gumbel-Softmax (second case)
        """
    d = X.shape[1]//2
    X2 = []
    for row in X:
        if split:
            mu, logvar = row[:d], row[d:]
            mu = mu.view((1, -1))
            logvar = logvar.view((1, -1))
            with torch.no_grad():
                X2.append(model.encoder.reparameterize(mu, logvar, nsamples=n_resample))
        else:
            row = row.view((1,-1))
            with torch.no_grad():
                X2.append(model.encoder.reparameterize(row, nsamples=n_resample))
    X2 = torch.cat(X2).squeeze(1)
    return X2


def refit_and_eval(score_name, clf, cv_results, train_codes, train_labels,
                   test_scaled_codes, test_labels, scorer):
    rank = cv_results['rank_test_' + score_name]
    id_best = rank.argmin()
    print("Score name", score_name, rank, id_best)
    best_params = cv_results['params'][id_best]
    best_crossval = cv_results['mean_test_' + score_name][id_best]
    print(cv_results['mean_test_' + score_name],
          np.mean(cv_results['mean_test_' + score_name]))
    print(cv_results['std_test_' + score_name])
    print("best params", best_params)
    best_estimator = clone(clf.estimator).set_params(**best_params)
    best_estimator.fit(train_codes, train_labels)
    test_loss = scorer(best_estimator, test_scaled_codes, test_labels)
    return best_crossval, test_loss


def init_config():
    parser = argparse.ArgumentParser(description='Evaluate SSL performance')

    # model hyperparameters
    parser.add_argument('--dataset', type=str, required=True, help='dataset to use')
    # optimization parameters
    parser.add_argument('--nsamples', type=int, default=1, help='number of samples for training')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--load-args-from-logs', action='store_true', default=False, help='Load args from logs.txt')

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
    parser.add_argument("--load_best_epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.)

    parser.add_argument("--classify_using_samples", action="store_true", default=False)
    parser.add_argument("--resample", type=int, default=-1)
    parser.add_argument("--use_log_loss", action="store_true", default=False)
    parser.add_argument("--fb", type=int, default=0,
                         help="0: no fb; 1: fb; 2: max(target_kl, kl) for each dimension")
    parser.add_argument("--target_kl", type=float, default=-1,
                         help="target kl of the free bits trick")
    parser.add_argument("--update_every", type=int, default=1,
                         help="target kl of the free bits trick")
    parser.add_argument("--num_label_per_class", type=int,
                        help='Num label per class or -1 all training examples and validate on validation set')
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--freeze_enc", action="store_true", default=False)
    parser.add_argument("--discriminator", type=str, default="linear")

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
    n = args.num_label_per_class

    args = argparse.Namespace(**vars(args), **params)
    args, args_model = load_args(args, parser)

    load_str = "_load" if args.load_path != "" else ""
    if args.fb == 0:
        fb_str = ""
    elif args.fb == 1:
        fb_str = "_fb"
    elif args.fb == 2:
        fb_str = "_fbdim"

    opt_str = "_adam" if args.opt == "adam" else "_sgd"

    if len(args.load_path.split("/")) > 2:
        load_path_str = args.load_path.split("/")[1]
    else:
        load_path_str = args.load_path.split("/")[0]

    model_str = "_{}".format(args.discriminator)
    # set load and save paths
    if args.exp_dir == None:
        args.exp_dir = "exp_{}{}_evaln/{}{}{}".format(args.dataset,
            load_str, load_path_str, model_str, opt_str)


    if len(args.load_path) <= 0:# and args.eval:
        raise ValueError()
    else:
        args.exp_dir = os.path.dirname(args.load_path)
    args.save_path = os.path.join(args.exp_dir, 'classifier.pt')

    # set args.label
    if 'label' in params:
        args.label = params['label']
    else:
        args.label = False

    return args, args_model

def main(args, args_model):
    global logging
    logging = get_logger_existing_dir(os.path.dirname(args.load_path), 'log_classifier.txt')

    if args.cuda:
        logging('using cuda')
    logging(str(args))

    opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}

    vocab = {}
    if getattr(args, 'vocab_file', None) is not None:
        with open(args.vocab_file) as fvocab:
            for i, line in enumerate(fvocab):
                vocab[line.strip()] = i

        vocab = VocabEntry(vocab)

    filename_glob = args.train_data + '.seed_*.n_' + str(args.num_label_per_class)
    train_sets = glob.glob(filename_glob)
    print("Train sets:", train_sets)

    main_train_data = MonoTextData(args.train_data, label=args.label, vocab=vocab)
    vocab = main_train_data.vocab
    vocab_size = len(vocab)

    logging('finish reading datasets, vocab size is %d' % len(vocab))
    #sys.stdout.flush()

    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)

    #device = torch.device("cuda" if args.cuda else "cpu")
    device = "cuda" if args.cuda else "cpu"
    args_model.device = device

    if args_model.enc_type == 'lstm':
        args_model.pooling = getattr(args_model, 'pooling', None)
        encoder = GaussianLSTMEncoder(args_model, vocab_size, model_init, emb_init,
                                      pooling=args_model.pooling, )

    elif args_model.enc_type in ['max_avg_pool', 'max_pool', 'avg_pool']:
        args_model.skip_first_word = getattr(args_model, 'skip_first_word', None)
        encoder = GaussianPoolEncoder(args_model, vocab_size, model_init, emb_init, enc_type=args_model.enc_type, skip_first_word=args_model.skip_first_word)
        #args.enc_nh = args.dec_nh
    else:
        raise ValueError("the specified encoder type is not supported")

    args_model.encode_length = getattr(args_model, 'encode_length', None)
    if args_model.dec_type == 'lstm':
        decoder = LSTMDecoder(args_model, vocab, model_init, emb_init, args_model.encode_length)
    elif args_model.dec_type == 'unigram':
        decoder = UnigramDecoder(args_model, vocab, model_init, emb_init)

    vae = VAE(encoder, decoder, args_model, args_model.encode_length).to(device)

    if args.load_path:
        print("load args!")
        print(vae)
        loaded_state_dict = torch.load(args.load_path)
        vae.load_state_dict(loaded_state_dict)
        logging("%s loaded" % args.load_path)

    vae.eval()

    def preprocess(data_fn):
        codes, labels = read_dataset(data_fn, vocab, device, vae, args.classify_using_samples)
        if args.classify_using_samples:
            is_gaussian_enc = codes.shape[1] == (vae.encoder.nz*2)
            codes = augment_dataset(codes, 1, is_gaussian_enc, vae) # use only 1 sample for test
        codes = codes.cpu().numpy()
        labels = labels.cpu().numpy()
        return codes, labels

    test_codes, test_labels = preprocess(args.test_data)

    test_f1_scores = []
    average_f1 = 'macro'
    f1_scorer = make_scorer(f1_score, average=average_f1, labels=np.unique(test_labels), greater_is_better=True)
    # log loss: negative log likelihood. We should minimize that, so greater_is_better=False
    log_loss_scorer = make_scorer(log_loss , needs_proba=True, greater_is_better=False)
    warnings.filterwarnings('ignore')
    results = {
        'n_samples_per_class': args.num_label_per_class,
    }
    n_repeats = args.n_repeats

    n_splits = min(args.num_label_per_class, 5)
    for i, fn in enumerate(train_sets):
        codes, labels = preprocess(fn)
        if args.resample > 1:
            # going to augment the training set by sampling
            # then create a new cross validation function to get the correct indices
            cross_val = augment_cross_val(labels, args.resample, n_splits, n_repeats)
            labels = np.repeat(labels, args.resample)
        else:
            cross_val = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

        scaler = StandardScaler()
        codes = scaler.fit_transform(codes)
        scaled_test_codes = scaler.transform(test_codes)
        gridsearch = GridSearchCV(
            LogisticRegression(solver='sag', multi_class='auto'),
            {
                "penalty": ['l2'],
                "C": [0.01, 0.1, 1, 10, 100],
            },
            cv=cross_val,
            scoring={
                "f1": f1_scorer,
                "log": log_loss_scorer,
            },
            refit=False,
        )
        clf = gridsearch
        clf.fit(codes, labels)
        crossval_f1, test_f1 = refit_and_eval(
            'f1', clf, clf.cv_results_, codes, labels, scaled_test_codes,
            test_labels, f1_scorer,
        )
        crossval_log, test_log_loss = refit_and_eval(
            'log', clf, clf.cv_results_, codes, labels, scaled_test_codes,
            test_labels, log_loss_scorer,
        )
        results[i] = {
            "F1": {'crossval': crossval_f1, 'test': test_f1},
            "log": {'crossval': crossval_log, 'test': test_log_loss},
        }
        print(results[i])

    if args.classify_using_samples:
        n_per_class = str(args.num_label_per_class)
        resample = 1 if args.resample == -1 else args.resample
        output_fn = os.path.join(args.exp_dir,
                                 'results_sample_' + str(resample) + '_' +
                                 n_per_class + '.json')
    else:
        output_fn = os.path.join(args.exp_dir,
                                 'results_' + n_per_class + '.json')
    with open(output_fn, 'w') as f:
        json.dump(results, f)


def read_dataset(fn, vocab, device, model, classify_using_samples):
    """ Read dataset in file fn with vocab and return (codes, labels)
    """
    data = MonoTextData(fn, label=True, vocab=vocab)
    data_batch, labels_batch = data.create_data_batch_labels(
        batch_size=1, device=device, batch_first=True)
    with torch.no_grad():
        labels, codes = [], []
        for i in np.random.permutation(len(data_batch)):
            batch_data = data_batch[i]
            batch_labels = labels_batch[i]
            batch_labels = [int(x) for x in batch_labels]
            labels_ = torch.tensor(batch_labels, dtype=torch.long, requires_grad=False, device=device)
            batch_size, sent_len = batch_data.size()
            if classify_using_samples:
                # to use samples, we can't simply encode more samples
                # need to keep the correspondance between z and x
                # so that we can cross-validate without leaks if we want to resample
                params, KL = model.encode(batch_data, 1, return_parameters=True)
                codes_ = torch.cat(params, dim=1)
            else:
                params, KL = model.encode(batch_data, 1,
                                          return_parameters=True)
                if type(params) == tuple:
                    codes_ = params[0]  # mean
                else:  # Gumbel Softmax
                    codes_ = params
            labels.append(labels_)
            codes.append(codes_)
        codes = torch.cat(codes, 0)
        labels = torch.stack(labels).reshape((-1,))
    return (codes, labels)


if __name__ == '__main__':
    args, args_model = init_config()
    main(args, args_model)
