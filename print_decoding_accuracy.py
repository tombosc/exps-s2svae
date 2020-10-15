import argparse
from pool_results import df_from_pickle, rename_colnames_values_for_print
from modules.hyperparameters import all_hp_except_seed, check_correct_groupby
import pandas as pd


def flatten(df):
    df.columns = ['_'.join(col).strip('_') for col in df.columns.values]


def aggreg(results, grouping_variable, suffixes):
    aggregated = None
    i = 0
    for df, suffix in zip(results, suffixes):
        check_correct_groupby(df, all_hp_except_seed, ['seed'])
        average_test_metric = suffix + '_average_test_' + loss
        grouped = df.groupby(grouping_variables)
        suffix_decoding = '_b'
        fields_to_aggregate = [
            'f1_decoding', 'correct_len_pc', 'word_1_retrieve_pc'
        ]
        fields_to_aggregate = [f + suffix_decoding for f in fields_to_aggregate]
        fields_to_aggregate += ['iw_nll', 'iw_ppl']
        processed_group = grouped.agg({
            f: ['mean', 'std', 'count'] for f in fields_to_aggregate
        })
        df.groupby(grouping_variables).apply(lambda X: print(X['iw_nll']))
        F1_col = 'F1_' + str(suffix)
        processed_group = processed_group.rename(columns={
            average_test_metric: F1_col,
        })

        if i == 0:
            aggregated = processed_group
        else:
            suffixes_ = ('', '_' + suffixes[i])
            aggregated = pd.merge(
                aggregated, processed_group, on=grouping_variables,
                suffixes=suffixes_,
            )
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
    # First, we are going to compare models for any 'target_kl' value 
    # There should be 4 target_kl values and 3 seeds, so 12 runs per aggregate
    print("Check that the filter is correct")

    def pre_filter(df):
        return df[(
            df['target_kl'].isin([8]) &
            df['nz'].isin([16]) &
            (df['vanilla_ae'] == False) & 
            (df['dec_type'] == 'lstm') &
            (df['skip_first_word'] == False) &
            (df['encode_length'] == -1) &
            (df['fb'] == 1) &
            (df['warm_up'] == 0) &
            (df['f1_decoding_b'] != -1) &
            (df['reconstruct_partial'] == False) &
            (df['reconstruct_partial_from'] == -1) &
            (df['enc_reverse'] == False) &
            (df['reconstruct_random_words'] == -1)
        )]

    results = [pre_filter(df) for df in results]

    grouping_variables = [
        'enc_type', 'dec_type', 'warm_up', 'pooling', 'reset_dec', 'pre', 'fb',
        'encode_length', 'freeze_encoder', 'unidec'
    ]

    results_cv = aggreg(results, grouping_variables, suffixes)
    results_cv = results_cv.rename(columns={
        'val_elbo': 'ppl*',
        'val_kl': 'KL',
        'pooling': 'pool',
        'enc_type': 'encoder',
    })

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Rename F1 scores
    rename_F1 = {'F1_' + s: 'F1(' + s + ')' for s in suffixes[0:]}
    rename_F1['F1'] = 'F1(' + suffixes[0] + ')'
    results_cv = results_cv.rename(columns=rename_F1)
    results_cv = results_cv.reset_index()
    results_cv = results_cv.set_index(
        ['enc_type', 'pooling', 'pre', 'reset_dec', 'unidec']
    )
    R = results_cv.loc[[
        ('lstm', '-', True, True, False),
        ('lstm', 'max', True, True, False),
        ('-', 'max', False, False, False),
        ('-', 'max', True, False, True),
        ('lstm', 'max', True, False, True),
        ('lstm', 'avg', False, True, False),
    ]].reset_index(['reset_dec'])
    R = R.drop(columns=['warm_up', 'encode_length', 'fb', 'freeze_encoder', 'reset_dec'])
    R = R.reset_index().set_index(['enc_type', 'pooling', 'pre'])
    R.drop(columns=[('unidec', '')], inplace=True)
    metrics = set([col[0] for col in R.columns if col[1] != ''])
    for metric in metrics:
        mean = R[(metric, 'mean')]
        std = R[(metric, 'std')]
        if metric not in ['iw_nll', 'iw_ppl']:
            mean = mean.mul(100)
            std = std.mul(100)
        mean = mean.round(1).astype(str)
        std = std.round(1).astype(str)
        R.drop([(metric, 'mean'), (metric, 'std'), (metric, 'count')], inplace=True, axis='columns')
        R[metric] = '$' + mean.str.cat(std, sep='\pm') + '$'

    R = R.reset_index()
    R = rename_colnames_values_for_print(R)
    R = R.set_index(['Enc.', '$r$', 'Pre.'])
    print(R[['f1_decoding_b', 'word_1_retrieve_pc_b', 'correct_len_pc_b', 'iw_ppl']])
    # Latex for paper
    print(R[['f1_decoding_b', 'word_1_retrieve_pc_b', 'correct_len_pc_b', 'iw_ppl']].to_latex(
        sparsify=False,
        escape=False,
    ))
