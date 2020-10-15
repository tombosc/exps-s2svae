import glob
import json
import pickle
import argparse
from exp_utils import load_args_from_logs
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def retrieve_loss(log_fn):
    """ Parse log file log_fn and return a dict with various "val_*" and
    "test_*" information: "val_kl", "val_recon", "val_nll", "val_ppl", ... and
    "test_*" equivalents.
    """
    def parse_loss_line(l, prefix):
        l = l.split(' ')[2:]
        keys = [l[i][:-1] for i in range(0, len(l), 2)]
        values = [float(l[i][:-1]) for i in range(1, len(l), 2)]
        return {prefix + k: v for k, v in zip(keys, values)}

    with open(log_fn, 'r') as f:
        lines = f.readlines()
        loss = parse_loss_line(lines[-4], 'val_')
        test_loss = parse_loss_line(lines[-1], 'test_')
        loss.update(test_loss)
        return loss


def retrieve_nll_estimate(log_fn):
    with open(log_fn, 'r') as f:
        lines = f.readlines()
        estimate_nll = float(lines[-1].split(' ')[2][:-1])
        estimate_ppl = float(lines[-1].split(' ')[-1][:-1])
        return estimate_nll, estimate_ppl


def aggregate_cross_val(df, average_test_metric, loss, keep_rows):
    rows = []
    len_df = len(df)
    n_splits = 5 # TODO remove that hardcoding
    for i in range(n_splits):
        crossval_metric = str(i) + '_crossval_' + loss
        # test_metric = str(i) + '_test_' + loss
        sorted_res = df.sort_values(crossval_metric, ascending=False)
        best_row = sorted_res.iloc[0]
        rows.append(best_row)
    rows = pd.DataFrame(rows)
    if not keep_rows:
        # agg = rows.agg(['mean', 'std'])
        agg = rows.agg({
            'test_elbo': ['mean'],
            'test_kl': ['mean'],
            'test_ppl': ['mean'],
            # 'warm_up': ['mean'],
            # 'target_kl': ['mean'],
            'f1_decoding_b': ['mean'],
            'f1_decoding_g': ['mean'],
            average_test_metric: ['mean'],
        })
        agg['counts'] = len_df
        agg['crossval'] = best_row[crossval_metric]
        return agg
    return rows


def aggregate_best_elbo(df):
    sorted_by_elbo = df.sort_values(['val_elbo'], ascending=[True])
    return sorted_by_elbo.iloc[0][['val_elbo', 'val_kl', 'warm_up', 'target_kl', average_test_metric]]


def read_decoding_f1(fn_dir, fn_json):
    full_fn = os.path.join(fn_dir, fn_json)
    if os.path.exists(full_fn):
        with open(full_fn, 'r') as f:
            decoded_accuracy = json.load(f)
            f1_decoded = decoded_accuracy['macro_F1_gt_hyp']
            return f1_decoded
    return -1


def read_decoding_stats(fn_dir, fn_json):
    full_fn = os.path.join(fn_dir, fn_json)
    if os.path.exists(full_fn):
        with open(full_fn, 'r') as f:
            stats = json.load(f)
            retrieve_acc = stats['retrieve_pc']
            return retrieve_acc, stats['correct_len_pc']
    # TODO do not hardcode that it contains 20 positions?
    return [-1,]*20, -1


def read_ssl_results(full_fn):
    """ Semi-supervised learning evaluation result.

    There are two file formats:
        - one line for cross-validation
        - two lines for full train set, with validation and test.
    """
    with open(full_fn, 'r') as f:
        n_lines = len(f.readlines())
        f.seek(0)  # go back to beginning of file
        if n_lines == 1:
            # Cross-validation format
            results_ = json.load(f)
            new_results = aggregate_limited_samples(results_)
            n_samples_per_class = results_['n_samples_per_class']
        elif n_lines == 2:
            # Validation + test format
            new_results = aggregate_full_data(f)
            n_samples_per_class = 'all'
        else:
            raise ValueError(f"Unknown file format in file {full_fn}")
    return new_results, n_samples_per_class


def aggregate_limited_samples(results):
    """ Change format of the dict and compute averages"""
    new_results = {}
    crossval_results_F1, test_results_log, test_results_F1 = [], [], []
    for k, v in results.items():
        if not isinstance(v, dict):
            continue
        new_results[k + '_crossval_F1'] = v['F1']['crossval']
        new_results[k + '_crossval_log'] = v['log']['crossval']
        new_results[k + '_test_F1'] = v['F1']['test']
        new_results[k + '_test_log'] = v['log']['test']
        test_results_log.append(v['log']['test'])
        test_results_F1.append(v['F1']['test'])
        crossval_results_F1.append(v['F1']['crossval'])
    if not test_results_log:
        return {}
    n_samples_per_class = results['n_samples_per_class']
    str_n = str(n_samples_per_class)
    new_results[str_n + '_average_test_log'] = np.mean(test_results_log)
    new_results[str_n + '_average_test_F1'] = np.mean(test_results_F1)
    new_results[str_n + '_average_crossval_F1'] = np.mean(crossval_results_F1)
    return new_results


def aggregate_full_data(opened_file):
    valid, test = opened_file.readlines()
    valid = json.loads(valid)
    test = json.loads(test)
    new_results = {
        'all_average_crossval_log': valid['avg_loss'],
        'all_average_crossval_F1': valid['macro_f1'],
        'all_average_test_log': test['avg_loss'],
        'all_average_test_F1': test['macro_f1'],
        '0_crossval_F1': valid['macro_f1'],
        '0_test_F1': test['macro_f1'],
    }
    return new_results


def aggregate_results(root_dirs, in_fn):
    """ Aggregate results from [root_dirs]/*/[in_fn] into a pandas Dataframe.

    Args:
        root_dirs: list of subdirs containing in_fn and other JSONs (see below)
        in_fn: filename for SSL evaluation

    The "other JSONs" are results from the text generation evaluation. These
    filenames are hardcoded, unlike in_fn. See the function implem to see which
    files are necessary.
    """
    results = []
    for root_dir in root_dirs:
        glob_select_json = os.path.join(root_dir, '*', in_fn)
        for fn in glob.glob(glob_select_json):
            dirname = os.path.dirname(fn)
            f1_decoded_b = read_decoding_f1(dirname, 'decoded_f1_beam.json')
            f1_decoded_g = read_decoding_f1(dirname, 'decoded_f1_greedy.json')
            acc_b, correct_len_pc_b = read_decoding_stats(
                dirname, 'decoded_test_beam_stats.json'
            )
            acc_g, correct_len_pc_g = read_decoding_stats(
                dirname, 'decoded_test_greedy_stats.json'
            )
            new_results, n_samples_per_class = read_ssl_results(fn)
            new_results.update({
                'n_samples_per_class': n_samples_per_class,
                'f1_decoding_b': f1_decoded_b,
                'f1_decoding_g': f1_decoded_g,
                'word_1_retrieve_pc_b': acc_b[0],
                'word_1_retrieve_pc_g': acc_g[0],
                'correct_len_pc_b': correct_len_pc_b,
                'correct_len_pc_g': correct_len_pc_g,
            })
            results_ = new_results

            logs_fn = os.path.join(os.path.dirname(fn), 'log.txt')
            ppl_eval_fn = os.path.join(os.path.dirname(fn), 'test_eval.txt')
            try:
                losses = retrieve_loss(logs_fn)
            except:
                losses = {}
            try:
                estimated_nll, estimated_ppl = retrieve_nll_estimate(ppl_eval_fn)
            except:
                estimated_nll = -1.
                estimated_ppl = -1.

            args_and_metrics = vars(load_args_from_logs(None, logs_fn))
            #args_and_metrics['vanilla_ae'] = (args_and_metrics['kl_start'] == 0) & (args_and_metrics['target_kl'] == -1)
            args_and_metrics['pre'] = (args_and_metrics['load_path'] != '')
            args_and_metrics['unidec'] = ('unidec' in
                                          args_and_metrics['exp_dir'])
            args_and_metrics.update(losses)
            args_and_metrics.update(results_)
            args_and_metrics['iw_nll'] = estimated_nll
            args_and_metrics['iw_ppl'] = estimated_ppl
            results.append(args_and_metrics)
    return results


def fill_default_value(df, field, default):
    if field not in df:
        df[field] = default
    else:
        df[field] = df[field].fillna(default)
    return df


def rename_colnames_values_for_print(R):
    R = R.rename(columns={
        'enc_type': 'Enc.',
        'dec_type': 'Dec.',
        'pooling': '$r$',
        'pre': 'Pre.',
        'encode_length': 'Len.',
    })
    if 'Enc.' in R:
        R['Enc.'] = R['Enc.'].replace({'lstm': 'LSTM', '-': 'BoW'})
    if 'Dec.' in R:
        R['Dec.'] = R['Dec.'].replace({'lstm': 'LSTM', 'unigram': 'Uni'})
    if '$r$' in R:
        R['$r$'] = R['$r$'].replace({'-': 'last', 'max': 'max'})
    if 'Pre.' in R:
        R['Pre.'] = R['Pre.'].replace({True: 'PreAE', False: '-'})
    return R


def df_from_pickle(fn): 
    if not os.path.exists(fn):
        print("File does not exist")
        exit(1)
    res = pickle.load(open(fn, 'rb'))
    df = pd.DataFrame(res['data'])
    print("File prefix:", res['file_prefix'])
    print("N samples per class:", res['n_samples_per_class'])
    print("Len dataframe", len(df))
    # some preprocessing
    df['val_elbo'] = df['val_kl'] + df['val_recon']
    df['test_elbo'] = df['test_kl'] + df['test_recon']
    df.loc[df['enc_type'] == 'avg_pool', 'pooling'] = 'avg'
    df.loc[df['enc_type'] == 'avg_pool', 'enc_type'] = '-'
    df.loc[df['enc_type'] == 'max_pool', 'pooling'] = 'max'
    df.loc[df['enc_type'] == 'max_pool', 'enc_type'] = '-'
    df.loc[df['enc_type'] == 'max_avg_pool', 'pooling'] = 'max_avg'
    df.loc[df['enc_type'] == 'max_avg_pool', 'enc_type'] = '-'
    df['vanilla_ae'] = (df['kl_start'] == 0) & (df['target_kl'] == -1)
    # print("Unique freeze enc", df['freeze_encoder'].unique())
    df['pooling'] = df['pooling'].fillna('-')
    df = fill_default_value(df, 'skip_first_word', False)
    # df = fill_default_value(df, 'encode_length', False)
    df = fill_default_value(df, 'pred_linear_bias', False)
    df = fill_default_value(df, 'reconstruct_partial', False)
    df = fill_default_value(df, 'reconstruct_partial_from', -1)
    df = fill_default_value(df, 'reconstruct_random_words', -1)
    df = fill_default_value(df, 'encode_length', -1)
    df = fill_default_value(df, 'enc_reverse', False)
    df = fill_default_value(df, 'freeze_encoder', False)
    df = fill_default_value(df, 'freeze_encoder_exc', False)
    df.loc[(df['pre'] is True) & (df['freeze_encoder'] == -1),
           'freeze_encoder'] = False
    df.loc[(df['encode_length'] == 0), 'encode_length'] = -1
    df.loc[(df['enc_reverse'] == False), 'enc_reverse'] = False
    if 'gumbel_T' in df:
        df['discrete'] = ~df['gumbel_T'].isna()  # ~: negation
    else:
        df['discrete'] = False
    return df.fillna(-1), res['n_samples_per_class']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Pool --in_fn files produced by eval_classification.py from
        subdirectories list given in --aggregate.
        Store dataframe in given pkl_dump npy archive for future analysis."""
    )
    parser.add_argument("pkl_dump", type=str,
                        default='npy_archives/results_fs.npy')
    parser.add_argument("--aggregate", nargs='+',  
                        help='Subdirectories to aggregate into pkl_dump')
    parser.add_argument("--n_samples_per_class", type=int)
    parser.add_argument("--in_fn", type=str, default='',
                        help="Filename containing result of eval")

    args = parser.parse_args()

    if args.n_samples_per_class is None:
        parser.error('Need to specify --n_samples_per_class')
    if args.in_fn is None:
        parser.error('Need to specify --in_fn')
    results = aggregate_results(args.aggregate, args.in_fn)
    res = {
        'data': results,
        'file_prefix': args.in_fn,
        'n_samples_per_class': args.n_samples_per_class,
    }
    if len(results) > 0:
        pickle.dump(res, open(args.pkl_dump, 'wb'))
    else:
        print("No results were written.")
        exit(1)
