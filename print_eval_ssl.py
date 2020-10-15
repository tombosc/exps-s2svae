""" Print tables showing semi-supervised learning performance.
"""
import argparse
from pool_results import (
    df_from_pickle, rename_colnames_values_for_print
)
from modules.hyperparameters import all_hp_except_seed, check_correct_groupby
import pandas as pd
import numpy as np


def flatten(df):
    df.columns = ['_'.join(col).strip('_') for col in df.columns.values]


def aggregate_CV(df, average_test_metric, loss, possible_n_splits=[1,5], output_KL=True):
    """ 1-factor ANOVA with a random effect with model selection on HPs.
    """
    df = df.set_index(all_hp_except_seed)
    mean_over_seeds = df.groupby(level=all_hp_except_seed).agg('mean')
    cv_scores = [s for s in df.columns if s.endswith('_crossval_F1') and 'average' not in s]
    test_scores = [s for s in df.columns if s.endswith('_test_F1') and 'average' not in s]
    assert(len(cv_scores) == len(test_scores) and len(cv_scores) in possible_n_splits)
    # do model selection and rename all the different subsample fields
    # (which are subsample specific right now, eg. 1_crossval_F1 for the first
    # subsample) into one unique name
    runs = []
    c = 0
    for test_split_score, cv_split_score in zip(test_scores,cv_scores):
        # for each split, get the index of best model
        top_model_idx_split = mean_over_seeds[cv_split_score].idxmax()
        # store the CV and test score of the best architecture accross seeds
        scores_out = [cv_split_score, test_split_score, 'f1_decoding_b',
                      'f1_decoding_g', 'test_kl', 'seed']
        # double brackets with df.loc are important! 
        # if len(df) = 1, df.loc will return a pd.Series, else a pd.DataFrame
        # with [[]] we force it to be a DataFrame (even if one row)
        scores = df.loc[[top_model_idx_split]]
        scores = scores[scores_out]
        scores = scores.rename(columns={
            cv_split_score: ('cv_F1'),
            test_split_score: ('test_F1'),
            'test_kl': ('test_KL'),
        })
        scores['subsample'] = c
        c += 1
        runs.append(scores)
    runs = pd.concat(runs)
    # print(runs[['test_F1', 'subsample', 'seed']].to_string(index=False))
    n_seeds = len(runs['seed'].unique())
    n_runs = len(runs)
    n_splits = len(cv_scores)
    # TODO: not great to count like that, this assumes factorial design
    # better to use 'count' in agg?
    runs = runs.reset_index().set_index(all_hp_except_seed + ['seed', 'subsample'])
    print("N runs, n seeds, n splits:", n_runs, n_seeds, n_splits)
    fields = ['cv_F1', 'test_F1', 'test_KL']
    # global mean
    mu = runs.mean()
    # Option 1: Seed is a factor
    # seed specific mean
    alpha_i = runs.groupby('seed').agg({k: 'mean' for k in fields})
    # standard deviation of first random-effect factor 
    std_1 = np.sqrt(n_splits) * (mu - alpha_i).std(ddof=1)
    # standard deviation of replicates
    std_2 = (runs - alpha_i)[fields].std(ddof=n_seeds)
    # Sanity check: SS = SS_T + SS_E (treatment, error)
    # - MS_T = SS_T/(g-1) (g: number of different levels for the factor)
    # - MS_E = SS_E / N-g (N: total number of points)
    # MS are estimates of variances
    # Therefore we have the following relation:
    # (g-1) MS_T + (N-g) MS_E = (N-1) SS 
    #assert((n_seeds - 1) * std_1['test_F1']**2 + (n_runs - n_seeds) * std_2['test_F1']**2, (n_runs - 1) * runs['test_F1'].var(ddof=1))

    # Option 2: Split is a factor
    ## Merge these dataframes into mu
    ## split specific mean
    #alpha_i = runs.groupby('subsample').agg({k: 'mean' for k in fields})
    ## standard deviation of first random-effect factor 
    #std_1 = n_seeds * (mu - alpha_i).std(ddof=1)
    ## standard deviation of replicates
    #std_2 = (runs - alpha_i)[fields].std(ddof=n_splits)

    mu = pd.DataFrame(mu).transpose()
    # Create hierarchical index
    multi = pd.MultiIndex.from_tuples([(k, 'mean') for k in mu.columns.values])
    mu.columns = multi
    for k in fields:
        mu[(k, 'std_seed')] = std_1[k]
        mu[(k, 'std')] = std_2[k]
    return mu

def aggreg(results, grouping_variables, suffixes, method, loss):
    aggregated = None
    i = 0
    for df, suffix in zip(results, suffixes):
        check_correct_groupby(df, all_hp_except_seed, ['seed'])

        average_test_metric = suffix + '_average_test_' + loss
        grouped = df.groupby(grouping_variables)

        if method == 'crossval':
            agg_by_crossval = grouped.apply(lambda x: aggregate_CV(x, average_test_metric, loss))
            agg_by_crossval = agg_by_crossval.rename(columns = {
                average_test_metric: 'F1',
            })
        elif method == 'mean':
            agg_by_crossval = grouped.agg('mean')
            F1_col = 'F1_' + str(suffix)
            agg_by_crossval = agg_by_crossval.rename(columns = {
                average_test_metric: F1_col,
            })

        if i == 0:
            aggregated = agg_by_crossval
        else:
            print("Before agg")
            print(aggregated.columns.values, agg_by_crossval.columns.values)
            new_col_names = {}
            for name in agg_by_crossval.columns.levels[0]:
                new_col_names[name] = name + '_' + suffixes[i]
            print(new_col_names)
            agg_by_crossval = agg_by_crossval.rename(columns=new_col_names)
            aggregated = pd.concat((aggregated, agg_by_crossval), axis=1)#on=grouping_variables)
        print("EOL", len(aggregated))
        i += 1
    return aggregated

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_dumps", type=str, nargs='+')
    parser.add_argument("n_samples", type=str)
    args = parser.parse_args()
    loss = 'F1'

    suffixes = [n for n in args.n_samples.split(',')]
    results = [df_from_pickle(dump)[0] for dump in args.pkl_dumps]

    def pre_filter(df):
        return df[(
            df['target_kl'].isin([2, 8]) &
            df['nz'].isin([4, 16]) &
            (df['vanilla_ae'] == False) & 
            (df['skip_first_word'] == False)
        )]

    results = [pre_filter(df) for df in results]

    grouping_variables = [
        'enc_type', 'dec_type', 'warm_up', 'pooling',
        'reset_dec', 'pre', 'fb', 'encode_length', 'freeze_encoder',
    ]

    aggreg_method = 'crossval'

    results_cv = aggreg(
        results, grouping_variables, suffixes, aggreg_method, loss,
    )

    results_cv = results_cv.rename(
        columns = {
        'val_elbo': 'ppl*',
        'val_kl': 'KL',
        'pooling': 'pool',
        'enc_type': 'encoder',
    })

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Rename F1 scores
    rename_F1 = {'F1_' + s: 'F1(' + s +')' for s in suffixes[0:]}
    rename_F1['F1'] = 'F1(' + suffixes[0] + ')'
    results_cv = results_cv.rename(columns = rename_F1)
    F1_to_show = ['F1(' + s + ')' for s in suffixes]
    print("F1", F1_to_show)
    to_show = F1_to_show
    #if aggreg_method == 'crossval':
    #    to_show += ['counts']
    #results_cv = results_cv[to_show]
    #results_cv = results_cv.xs((0, -1), level=('warm_up', 'encode_length'))
    unigram = True
    results_cv = results_cv.xs((0, 1), level=('warm_up', 'fb'))
    if unigram:
        results_cv = results_cv.reset_index().set_index(['enc_type', 'dec_type', 'pooling', 'pre', 'reset_dec', 'encode_length'])
        print("Results cross-val")
        R = results_cv.loc[[
            ('lstm', 'lstm', '-', False, False, -1),
            ('lstm', 'lstm', '-', True, True, -1),
            ('lstm', 'lstm', 'max', False, False, -1),
            ('lstm', 'lstm', 'max', True, True, -1),
            ('-', 'lstm', 'max', False, False, -1),
            ('lstm', 'unigram', 'max', False, False, -1),
            ('lstm', 'unigram', '-', False, False, -1),
            ('-', 'unigram', 'max', False, False, -1),
            ('lstm', 'lstm', 'avg', False, True, -1),
        ]].reset_index(['reset_dec'])
        R = R.drop(columns=['reset_dec'])#, 'test_elbo_10', 'test_elbo_50'])
        R = R.reset_index().set_index(['enc_type', 'pooling', 'dec_type', 'pre'])
    else:
        results_cv = results_cv.xs((-1, 'lstm'), level=('encode_length', 'dec_type'))
        results_cv = results_cv.reset_index().set_index(['enc_type', 'pooling', 'pre', 'reset_dec'])
        print("Results cross-val")
        R = results_cv.loc[[
            ('lstm', '-', False, False),
            ('lstm', '-', True, True),
            ('lstm', 'max', False, False),
            ('lstm', 'max', True, True),
            ('-', 'max', False, False),
            ('lstm', 'avg', False, True),
        ]].reset_index(['reset_dec'])
        R = R.drop(columns=['reset_dec'])#, 'test_elbo_10', 'test_elbo_50'])
        R = R.reset_index().set_index(['enc_type', 'pooling', 'pre'])
    print("Attention")
    print(R.to_string(
        float_format="{:0.2f}".format,
        #index=False,
    ))
    print(R.reset_index().to_string(
        float_format="{:0.3f}".format,
        index=False,
    ))
    R.drop(inplace=True, columns=['freeze_encoder'])
    # Drop useless columns:
    suffixes = ['_' + s for s in suffixes]
    suffixes[0] = ''
    # now suffixes = ['', '_5', '_10']
    print(suffixes)
    for s in suffixes:
        R['r_mean'] = R[('test_F1' + s, 'mean')].mul(100).round(1).astype(str)
        R['r_std1'] = R[('test_F1' + s, 'std_seed')].mul(100).round(1).astype(str)
        R['r_std2'] = R[('test_F1' + s, 'std')].mul(100).round(1).astype(str)
        # If there is only one split:
        R['r_std2'] = R['r_std2'].str.replace('nan', '-')
        R['F1'+s] = '$' + R['r_mean'] + '\pm^{' + R['r_std2'] + '}_{' + R['r_std1'] + '}$'

    R = R[['F1' + s for s in suffixes]]
    R = R.reset_index()
    R = rename_colnames_values_for_print(R)
    floats = R.select_dtypes(include=['float64'])
    print(R.to_latex(
        index=False,
        escape=False,
    ))
